import torch
from torch.nn import functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.structures import BitMasks

from maskdino.modeling.transformer_decoder.maskdino_decoder import MaskDINODecoder
from maskdino.modeling.transformer_decoder.maskdino_decoder import TRANSFORMER_DECODER_REGISTRY
from maskdino.utils.utils import MLP, gen_encoder_output_proposals, inverse_sigmoid
from maskdino.utils import box_ops

from .myconfig import get_panomito_cfg

@TRANSFORMER_DECODER_REGISTRY.register()
class MaskDINODecoderDynamicHead(MaskDINODecoder):
    @configurable
    def __init__(self, **kwargs):

        super().__init__(**kwargs)        


        self.mask_dim = 256

        self.mask_head = DynamicConvHead(
            hidden_dim=self.mask_dim,
            channels=self.mask_dim,  
            num_bases=16,
            kernel_size=3,
        )

    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  
        
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            outputs_mask = self.mask_head(decoder_output, mask_features)

        return outputs_class, outputs_mask

@TRANSFORMER_DECODER_REGISTRY.register()
class HDMaskDINODecoder(MaskDINODecoder):
    @configurable
    def __init__(self, **kwargs):

        mycfg = get_panomito_cfg()
        if 'ALLHD' in mycfg:
            self.allhd = mycfg['allhd']
        else:
            self.allhd = False
        
        if self.allhd:
            kwargs['dec_layers'] = 3
        
        super().__init__(**kwargs)
        
        self.mask_dim = 256
        self.query_embed = nn.Embedding(kwargs['num_queries'], 4)

        if 'MYHEAD' in mycfg.PANOMITO:
            self.myhead = mycfg['myhead']
        else:
            self.myhead = False
        if self.myhead:
            self.up_conv = torch.nn.Sequential(

                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                torch.nn.Conv2d(self.mask_dim+1, 128, kernel_size=3, padding=1),

                torch.nn.GELU(),
                torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),

                

                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),

                torch.nn.GELU(),
                torch.nn.Conv2d(32, 1, kernel_size=3, padding=1),

            ).apply(self._init_weights)
        if 'ISHD' in mycfg.PANOMITO:
            self.ishd = mycfg['ishd']
        else:
            self.ishd = False
        if self.ishd:
            self.up_conv = torch.nn.Sequential(
               
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                torch.nn.Conv2d(self.mask_dim, self.mask_dim, kernel_size=3, padding=1),
                
                torch.nn.GELU(),
                torch.nn.Conv2d(self.mask_dim, self.mask_dim, kernel_size=3, padding=1),

                
                
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                torch.nn.Conv2d(self.mask_dim, self.mask_dim, kernel_size=3, padding=1),
                
                torch.nn.GELU(),
                torch.nn.Conv2d(self.mask_dim, self.mask_dim, kernel_size=3, padding=1),

            ).apply(self._init_weights)


    
    def forward(self, x, mask_features, masks, targets=None):

        if self.allhd:
            mask_features_detached = mask_features
            up_mask_features = self.up_conv(mask_features_detached)
           
            interpolated_mask_features = F.interpolate(mask_features_detached, size=up_mask_features.shape[-2:], 
                                                    mode='bilinear', align_corners=False)
            res_mask_features = up_mask_features + interpolated_mask_features 
            out, mask_dict = self.forward_allhd(x, mask_features, res_mask_features, masks, targets)
            return out, mask_dict
        
        if self.myhead:
            out, mask_dict = self.forward_myhead(x, mask_features,masks, targets)
            return out, mask_dict
        
        assert len(x) == self.num_feature_levels
        device = x[0].device
        size_list = []
        
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  
        mask_flatten = torch.cat(mask_flatten, 1) 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            enc_outputs_coord_unselected = self._bbox_embed(
                output_memory) + output_proposals  
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  
            

            outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features)
            
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs=dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':
                
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  
                    refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(device)
                elif self.initialize_box_type == 'mask2box': 
                    refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
                else:
                    assert NotImplementedError
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h],
                                                                                              dtype=torch.float).to(device)
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

       
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(tgt.transpose(0, 1), mask_features, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
        else:
            out_boxes = self.pred_box(references, hs)

        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), mask_features, self.training or (i == len(hs)-1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if mask_dict is not None:
            predictions_mask=torch.stack(predictions_mask)
            predictions_class=torch.stack(predictions_class)
            predictions_class, out_boxes,predictions_mask=\
                self.dn_post_process(predictions_class,out_boxes,mask_dict,predictions_mask)
            predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        elif self.training:  
            predictions_class[-1] += 0.0*self.label_enc.weight.sum()

        aux_out = self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask,out_boxes
            )
        
        if not self.ishd:
          
            out = {
                'pred_logits':predictions_class[-1],
                'pred_masks':predictions_mask[-1],
                'pred_boxes':out_boxes[-1],
                'aux_outputs': aux_out
            }

            if self.two_stage:
                out['interm_outputs'] = interm_outputs
            return out, mask_dict    
            
        else:
            mask_features_detached = mask_features.detach()
            up_mask_features = self.up_conv(mask_features_detached)
            
            interpolated_mask_features = F.interpolate(mask_features_detached, size=up_mask_features.shape[-2:], 
                                                    mode='bilinear', align_corners=False)
            res_mask_features = up_mask_features + interpolated_mask_features  
            
            outputs_class2, outputs_mask2 = self.forward_prediction_heads(hs[-1].transpose(0, 1), res_mask_features, True)
                       
            
            if mask_dict is not None:
                outputs_class2 = outputs_class2[:, mask_dict['pad_size']:,:]
                outputs_mask2 = outputs_mask2[:, mask_dict['pad_size']:,:]
               
                
            out = {
                'pred_logits':outputs_class2,
                'pred_masks':outputs_mask2, 
                'pred_boxes':out_boxes[-1],
                'aux_outputs': aux_out
            }

            if self.two_stage:
                out['interm_outputs'] = interm_outputs
            return out, mask_dict

    def forward_allhd(self, x, mask_features, sr_mask_features, masks, targets=None):
    
        assert len(x) == self.num_feature_levels
        device = x[0].device
        size_list = []
        
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  
        mask_flatten = torch.cat(mask_flatten, 1) 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            enc_outputs_coord_unselected = self._bbox_embed(
                output_memory) + output_proposals  
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
            

            outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), sr_mask_features)
            
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs=dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':
                
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  
                    refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(device)
                elif self.initialize_box_type == 'mask2box': 
                    refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
                else:
                    assert NotImplementedError
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h],
                                                                                              dtype=torch.float).to(device)
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

       
        if self.initial_pred:
            
            outputs_class, outputs_mask = self.forward_prediction_heads(tgt.transpose(0, 1),sr_mask_features, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
        else:
            out_boxes = self.pred_box(references, hs)

        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1).detach(), sr_mask_features, self.training or (i == len(hs)-1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if mask_dict is not None:
            predictions_mask=torch.stack(predictions_mask)
            predictions_class=torch.stack(predictions_class)
            predictions_class, out_boxes,predictions_mask=\
                self.dn_post_process(predictions_class,out_boxes,mask_dict,predictions_mask)
            predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        elif self.training:  
            predictions_class[-1] += 0.0*self.label_enc.weight.sum()

        aux_out = self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask,out_boxes
            )
        

       
        out = {
            'pred_logits':predictions_class[-1],
            'pred_masks':predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'aux_outputs': aux_out
        }

        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict    

    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask
    
   

    def forward_myhead(self, x, mask_features, masks, targets=None):
        assert len(x) == self.num_feature_levels
        device = x[0].device
        size_list = []
       
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  
        mask_flatten = torch.cat(mask_flatten, 1) 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            enc_outputs_coord_unselected = self._bbox_embed(
                output_memory) + output_proposals 
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) 
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  
            

            outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features)
            
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs=dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':
                
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  
                    refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(device)
                elif self.initialize_box_type == 'mask2box':  
                    refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
                else:
                    assert NotImplementedError
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h],
                                                                                              dtype=torch.float).to(device)
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

        
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(tgt.transpose(0, 1), mask_features, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
        else:
            out_boxes = self.pred_box(references, hs)

        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), mask_features, self.training or (i == len(hs)-1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if mask_dict is not None:
            predictions_mask=torch.stack(predictions_mask)
            predictions_class=torch.stack(predictions_class)
            predictions_class, out_boxes,predictions_mask=\
                self.dn_post_process(predictions_class,out_boxes,mask_dict,predictions_mask)
            predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        elif self.training: 
            predictions_class[-1] += 0.0*self.label_enc.weight.sum()

        aux_out = self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask,out_boxes
            )
        
        out = {
            'pred_logits':predictions_class[-1],
            'pred_masks':predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'aux_outputs': aux_out
        }

        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict    

        
    def refine_head(self, masks, mask_features):
        up_mask_features = self.up_conv(mask_features)
        
        interpolated_mask_features = F.interpolate(mask_features, size=up_mask_features.shape[-2:], mode='bilinear', align_corners=True)
        res_mask_features = up_mask_features + interpolated_mask_features 
        
        up_masks = F.interpolate(masks, size=up_mask_features.shape[-2:],mode='bilinear', align_corners=False)
        return up_masks
    
    def _bbox_to_spatial_grid(self, bbox, h, w, device):
        
        y, x = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device))
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        
        cx, cy, bw, bh = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]  # (b, q, 1)    
        bw = bw.unsqueeze(-1).unsqueeze(-1) 
        bh = bh.unsqueeze(-1).unsqueeze(-1)
        cx = cx.unsqueeze(-1).unsqueeze(-1)
        cy = cy.unsqueeze(-1).unsqueeze(-1)

        
        x_dist = (x - cx) ** 2 / (2 * bw ** 2 + 1e-8)
        y_dist = (y - cy) ** 2 / (2 * bh ** 2 + 1e-8)
        grid = torch.exp(-(x_dist + y_dist))  
        return grid 

   

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            module.weight.data.fill_(0)
            if module.bias is not None:
                module.bias.data.fill_(0)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.fill_(0)
            module.bias.data.fill_(0)


@TRANSFORMER_DECODER_REGISTRY.register()
class SrMaskDINODecoder(MaskDINODecoder):
    @configurable
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        self.mask_dim = 256


        self.up_conv = torch.nn.Sequential(
            
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(self.mask_dim, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            
            
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
        ).apply(self._init_weights)

        self.sr_output =  torch.nn.Sequential(           
              
            torch.nn.Conv2d(64, 1, kernel_size=1)
        ).apply(self._init_weights)

        self.down_conv = torch.nn.Sequential(
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

           
            torch.nn.Conv2d(256, self.mask_dim, kernel_size=1)
        ).apply(self._init_weights)

    def forward(self, x, mask_features, masks, targets=None):
    
        up_mask_features = self.up_conv(mask_features)
        down_mask_features = self.down_conv(up_mask_features) 
        mask_features = mask_features + down_mask_features       
        predictions = super(SrMaskDINODecoder,self).forward(x, mask_features, masks, targets)        
              
        high_res_mask_features = self.sr_output(up_mask_features)
        predictions[0]['sr'] = torch.sigmoid(high_res_mask_features)
        return predictions

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            module.weight.data.fill_(0)
            if module.bias is not None:
                module.bias.data.fill_(0)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.fill_(0)
            module.bias.data.fill_(0)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return x
    
class DynamicConvHead(torch.nn.Module):
    def __init__(self, hidden_dim=256, channels=256, num_bases=64, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.num_bases = num_bases
        self.kernel_size = kernel_size

        self.basis_module = torch.nn.Conv2d(channels, num_bases, 1)
        weight_dim = num_bases * kernel_size * kernel_size
        self.weight_predictor = MLP(hidden_dim, hidden_dim, weight_dim, 3)
        self.projection = torch.nn.Conv2d(num_bases, 1, 1)


        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)

    def forward(self, x, mask_features):
        B, Q, C = x.shape
        H, W = mask_features.shape[-2:]

        bases = self.basis_module(mask_features) 

        weights = self.weight_predictor(x).view(B, Q, self.num_bases, self.kernel_size**2)  
        bases_unfold = F.unfold(bases, kernel_size=self.kernel_size, padding=1)
        bases_unfold = bases_unfold.view(B, self.num_bases, -1, H*W).permute(0,3,1,2).reshape(B*H*W, self.num_bases, -1)

        weights = weights.view(B, Q, self.num_bases, -1)
        bases_unfold = bases_unfold.view(B, H*W, self.num_bases, -1)

        out = torch.einsum("bqkr,blkr->bql", weights, bases_unfold)  
        out = out.view(B, Q, H, W)

        return out  