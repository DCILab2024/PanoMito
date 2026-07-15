from typing import Dict
from torch import nn
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.config import configurable
from detectron2.layers import ShapeSpec

from maskdino.modeling.pixel_decoder.maskdino_encoder import MaskDINOEncoder



@SEM_SEG_HEADS_REGISTRY.register()
class HighResMaskDINOEncoder(MaskDINOEncoder):
    @configurable
    def __init__(self, input_shape: Dict[str, ShapeSpec], **kwargs):
       
        super().__init__(input_shape, **kwargs)
        self.conv_dim = 256
        
        self.high_res_conv = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, self.conv_dim),
            nn.ReLU(inplace=True)
        )

        self.high_res_conv.to("cuda")

        
        for layer in self.high_res_conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    
    def forward_features(self, features, masks):
       
        mask_features, high_res_feature, multi_scale_features = super(HighResMaskDINOEncoder, self).forward_features(features, masks)

        
        high_res_feature_branch = self.high_res_conv(high_res_feature)

       
        multi_scale_features.append(high_res_feature_branch)


       
        return mask_features, high_res_feature, multi_scale_features