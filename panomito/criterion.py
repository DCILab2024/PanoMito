import cv2, numpy as np
import torch
from torch.nn import functional as F
from maskdino.modeling.criterion import SetCriterion, calculate_uncertainty, sigmoid_ce_loss_jit, dice_loss_jit
from maskdino.utils.misc import nested_tensor_from_tensor_list, nested_tensor_from_tensor_list
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
class MySetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, dn="no", dn_losses=[], panoptic_on=False,semantic_ce_loss=False):
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses,
                         num_points, oversample_ratio, importance_sample_ratio,dn,dn_losses, panoptic_on, semantic_ce_loss)


    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,  # latter
            'masks': self.loss_masks,
            'boxes': self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes,  # latter
            'connectivity': self.loss_connectivity,
            'masks_routine': self.loss_masks_routine,
            'sr': self.loss_sr,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def loss_sr(self, outputs, targets, indices, num_masks):
        try:
            src_sr = outputs['sr']
            tgt_sr = targets[0]['sr']
            l1loss = torch.nn.L1Loss()
            loss = l1loss(src_sr, tgt_sr[:,0:1,:,:])
            return {"loss_sr": loss}
        except:
            tgt_sr = targets[0]['sr']
            return {"loss_sr": torch.tensor(0.0, device=tgt_sr.device, requires_grad=True)}

    def loss_connectivity(self, outputs, targets, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]  
        if len(src_masks) == 0:
            
            return {"loss_connectivity": torch.tensor(0.0, device=src_masks.device, requires_grad=True)}
        
        pred_probs = src_masks.sigmoid()

        loss = 0.0
        for pred_prob in pred_probs:
            
            connectivity_loss = self.penalize_small_regions(pred_prob, k=3)
            loss += connectivity_loss

        return {"loss_connectivity": loss / len(pred_probs)}

    def penalize_small_regions(self, pred_prob, k=2):
        
        with torch.no_grad():  
           
            binary_mask_np = (pred_prob > 0.5).float().cpu().numpy()
            
            num_labels, labels = cv2.connectedComponents(binary_mask_np.astype(np.uint8))
            if num_labels <= 2: 
                return torch.tensor(0.0, device=pred_prob.device)
            if num_labels > 2:
                
                max_label = np.argmax([np.sum(labels == i) for i in range(1, num_labels)]) + 1
                small_regions_mask = (labels != 0) & (labels != max_label)
                
                kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
                dilated_mask = cv2.dilate(
                    small_regions_mask.astype(np.uint8),
                    kernel,
                    iterations=1
                ).astype(bool)

        
        dx = torch.abs(pred_prob[:, 1:] - pred_prob[:, :-1])
        dy = torch.abs(pred_prob[1:, :] - pred_prob[:-1, :])

        
        penalty_dx = dx[dilated_mask[:, :-1]]
        penalty_dy = dy[dilated_mask[:-1, :]]

        
        loss = (penalty_dx.mean() + penalty_dy.mean()) * 10

        return loss


    def loss_masks_routine(self, outputs, targets, indices, num_masks):
      
        src_idx = self._get_src_permutation_idx(indices)  
        tgt_idx = self._get_tgt_permutation_idx(indices)  
        src_masks = outputs["pred_masks"]  
        src_masks = src_masks[src_idx]  
        masks = [t["masks"] for t in targets]  
        
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]  

        
        pred_masks = F.interpolate(
            src_masks.unsqueeze(1),  
            scale_factor=4,  
            mode="bilinear",  
            align_corners=False,  
        ).squeeze(1)

      
        if target_masks.numel() == 0:
            
            return {}
        if target_masks.numel() != 0:
            target_masks.min() >= 0 and target_masks.max() <= 1


        
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, target_masks, reduction="mean"
        ) 

       
        pred_masks = torch.sigmoid(pred_masks)  
        smooth = 1e-6  
        intersection = (pred_masks * target_masks).sum(dim=(1, 2))  
        union = pred_masks.sum(dim=(1, 2)) + target_masks.sum(dim=(1, 2))
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score.mean()


        losses = {
            "loss_mask_routine": bce_loss,
            "loss_dice_routine": dice_loss,
            }
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

       
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses