
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.config import get_cfg, CfgNode
from panomito.myconfig import set_panomito_cfg
from maskdino import MaskDINO
from panomito.panomito import PanoMito

try:
    from ..maskdino.config import add_maskdino_config
except:
    from maskdino.config import add_maskdino_config

class PanomitoPredictor:
    def __init__(self, 
                 model_path, 
                 confidence_thresh=0.3, 
                 max_det=1000, 
                 device='cuda', 
                 config_path='./configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'):
        self.model_path = model_path
        self.confidence_thresh = confidence_thresh
        self.config_path = config_path
        self.max_det = max_det
        self.device = device
        self.setup()
        
        self.predictor = DefaultPredictor(self.cfg)

    def setup(self):
        
        self.cfg = get_cfg()

        
        add_deeplab_config(self.cfg)
        add_maskdino_config(self.cfg)
        self.cfg.merge_from_file(self.config_path)

        self.cfg.MODEL.IN_CHANS = 3
        self.cfg.SOLVER.AMP.ENABLED = False
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.MODEL.DEVICE = self.device

        self.cfg.INPUT.IMAGE_SIZE = 512
        self.cfg.INPUT.MIN_SIZE_TEST = 512
        self.cfg.INPUT.MAX_SIZE_TEST = 2000

        self.cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = self.max_det
        self.cfg.TEST.DETECTIONS_PER_IMAGE = self.max_det
        
        self.cfg.MODEL.META_ARCHITECTURE = "PanoMito"
        self.cfg.MODEL.MaskDINO.CONNECTIVITY_WEIGHT = 0.0
        
        self.cfg.PANOMITO = CfgNode()
        self.cfg.PANOMITO.SemLoss = True    
        self.cfg.PANOMITO.AllSemLoss = True
        self.cfg.PANOMITO.SemLossWeight = 0
        self.cfg.PANOMITO.SemLossSigmoid = False
        set_panomito_cfg(self.cfg)
        
        self.cfg.freeze()

    def predict(self, image):
        
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        confident_detections = instances[instances.scores > self.confidence_thresh]

        rst = []
        mask_array = confident_detections.pred_masks.numpy().copy()
        num_instances = mask_array.shape[0]
        output = np.zeros(mask_array.shape[1:])

        for i in range(num_instances):
            output[mask_array[i,:,:]==True] = i+1

        output = output.astype(int)

       
        del outputs, instances, confident_detections
        torch.cuda.empty_cache()
        
        return output

