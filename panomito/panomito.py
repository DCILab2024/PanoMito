from torch.nn import functional as F
import torch

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, Instances

from maskdino.maskdino import MaskDINO
from maskdino.modeling.matcher import HungarianMatcher

from panomito.criterion import MySetCriterion

@META_ARCH_REGISTRY.register()
class PanoMito(MaskDINO):
    def __init__(self, cfg):
        super().__init__(cfg)

        if 'SemLoss' in cfg.PANOMITO:
            self.SemLoss = cfg.PANOMITO.SemLoss
            try:
                self.SemLossWeight = cfg.PANOMITO.SemLossWeight
            except:
                self.SemLossWeight = 10.
            try:
                self.SemLossSigmoid = cfg.PANOMITO.SemLossSigmoid
            except:
                self.SemLossSigmoid = False
        else:
            self.SemLoss = False

        if 'AllSemLoss' in cfg.PANOMITO:
            self.AllSemLoss = cfg.PANOMITO.AllSemLoss
        else:
            self.AllSemLoss = False
    @classmethod
    def from_config(cls, cfg):
        maskdino_dict = super().from_config(cfg)

        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT 
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #

        connectivity_weight = cfg.MODEL.MaskDINO.CONNECTIVITY_WEIGHT

        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT



        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )
        weight_dict = {"loss_ce": class_weight}
        if mask_weight>0 and dice_weight>0:
            weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        if box_weight>0 and giou_weight>0:
            weight_dict.update({"loss_bbox": box_weight, "loss_giou": giou_weight})

        if connectivity_weight>0:
            weight_dict.update({"loss_connectivity": connectivity_weight})

        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)

        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k != "loss_mask" and k != "loss_dice"})
            dn_losses = ["labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses = ["labels", "masks", "boxes"]
        else:
            dn_losses = []
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}

            for i in range(dec_layers+1): 
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks", "boxes"]
        else:
            losses = ["labels", "masks"]
        if connectivity_weight>0:
            losses.append("connectivity")

        criterion = MySetCriterion(
            num_classes=maskdino_dict['sem_seg_head'].num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
        )

        return {
            "backbone": maskdino_dict['backbone'],
            "sem_seg_head": maskdino_dict['sem_seg_head'],
            "criterion": criterion,  
            "num_queries": maskdino_dict['num_queries'],
            "object_mask_threshold": maskdino_dict['object_mask_threshold'],
            "overlap_threshold": maskdino_dict['overlap_threshold'],
            "metadata": maskdino_dict['metadata'],
            "size_divisibility": maskdino_dict['size_divisibility'],
            "sem_seg_postprocess_before_inference": maskdino_dict['sem_seg_postprocess_before_inference'],
            "pixel_mean": maskdino_dict['pixel_mean'],
            "pixel_std": maskdino_dict['pixel_std'],
            "semantic_on": maskdino_dict['semantic_on'],
            "instance_on": maskdino_dict['instance_on'],
            "panoptic_on": maskdino_dict['panoptic_on'],
            "test_topk_per_image": maskdino_dict['test_topk_per_image'],
            "data_loader": maskdino_dict['data_loader'],
            "focus_on_box": maskdino_dict['focus_on_box'],
            "transform_eval": maskdino_dict['transform_eval'],
            "pano_temp": maskdino_dict['pano_temp'],
            "semantic_ce_loss": maskdino_dict['semantic_ce_loss']
        }
    
    def forward(self, batched_inputs):

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        if self.training:

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                if 'detr' in self.data_loader:
                    targets = self.prepare_targets_detr(gt_instances, images)
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            outputs,mask_dict = self.sem_seg_head(features,targets=targets)
            losses = self.criterion(outputs, targets,mask_dict)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] = losses[k] * self.criterion.weight_dict[k]
                else:
                    losses.pop(k)

            if "sr" in batched_inputs[0]:
                sr = [x["sr"].to(self.device) for x in batched_inputs]
                sr = [(x - self.pixel_mean) / self.pixel_std for x in sr]
                sr = ImageList.from_tensors(sr, self.size_divisibility) 
                targets[0]['sr'] = sr.tensor
            
            if self.SemLoss:                    

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                mask_box_results = outputs["pred_boxes"]
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                sem_gts = []
                for idx, per_image in enumerate(batched_inputs):
                    gt_instances = per_image["instances"].to(self.device)
                    gt_classes = gt_instances.gt_classes  
                    gt_masks = gt_instances.gt_masks.float() 

                    N = len(gt_classes)
                    mask_cls_onehot = torch.zeros(N, 80, device=self.device) 
                    mask_cls_onehot[range(N), gt_classes] = 1.
                    sems_gt = torch.einsum("qc,qhw->chw", mask_cls_onehot, gt_masks)
                    
                    sem_gts.append(sems_gt[0])
                sem_gts = torch.stack(sem_gts)


                sems_preds = []
                for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
                ):  
                    height = input_per_image['image'].shape[-2]  
                    width = input_per_image['image'].shape[-1] 
                    new_size = mask_pred_result.shape[-2:]  

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    mask_cls_result = mask_cls_result.sigmoid()
                    mask_pred_result = mask_pred_result.sigmoid()
                    r = torch.einsum("qc,qhw->chw", mask_cls_result, mask_pred_result)
                    sems_preds.append(r[0])
                sems_preds = torch.stack(sems_preds)

                if self.SemLossSigmoid:
                    loss_sem = F.l1_loss(sems_preds.sigmoid(), sem_gts.sigmoid(), reduction='mean')
                else:
                    loss_sem = F.l1_loss(sems_preds, sem_gts, reduction='mean')

                losses['loss_sem'] = loss_sem*self.SemLossWeight                

                if self.AllSemLoss:
                    for kk in range(len(outputs['aux_outputs'])):
                        mask_cls_results = outputs['aux_outputs'][kk]["pred_logits"]
                        mask_pred_results = outputs['aux_outputs'][kk]["pred_masks"]
                        mask_box_results = outputs['aux_outputs'][kk]["pred_boxes"]
                        sems_preds = []
                        for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                            mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
                        ):  
                            height = input_per_image['image'].shape[-2] 
                            width = input_per_image['image'].shape[-1] 
                            new_size = mask_pred_result.shape[-2:]

                            if self.sem_seg_postprocess_before_inference:
                                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                                    mask_pred_result, image_size, height, width
                                )
                                mask_cls_result = mask_cls_result.to(mask_pred_result)
                          
                            mask_cls_result = mask_cls_result.sigmoid()
                            mask_pred_result = mask_pred_result.sigmoid()
                            r = torch.einsum("qc,qhw->chw", mask_cls_result, mask_pred_result)
                            sems_preds.append(r[0])
                        sems_preds = torch.stack(sems_preds)

                        if self.SemLossSigmoid:
                            loss_sem = F.l1_loss(sems_preds.sigmoid(), sem_gts.sigmoid(), reduction='mean')
                        else:
                            loss_sem = F.l1_loss(sems_preds, sem_gts, reduction='mean')

                        losses[f'loss_sem_{kk}'] = loss_sem*self.SemLossWeight                

            return losses
        else:
            outputs, _ = self.sem_seg_head(features)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
            ): 
                height = input_per_image.get("height", image_size[0])  
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = mask_pred_result.shape[-2:]  


                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r


                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r



                if self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    mask_box_result = self.box_postprocess(mask_box_result, height, width)

                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results
        
    def instance_inference(self, mask_cls, mask_pred, mask_box_result):

        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid() 
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
   
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)

        result.pred_masks = (mask_pred > 0).float()
 
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)

        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
        
def test():
    import tifffile
    for i in range(0,1000):
        mask = outputs[0]['instances']._fields['pred_masks'][i].detach().cpu().numpy()
        if mask.sum() > 8:
            tifffile.imwrite('mask.tif', mask)
            bbox =  outputs[0]['instances']._fields['pred_boxes'][i]
            break
    print(mask.sum())
