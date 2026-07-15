import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch, numpy as np
from torch.optim.optimizer import Optimizer
import copy, logging, weakref, time, itertools
from typing import Any, Dict, List, Set
from collections import OrderedDict
import sys
sys.path.append(os.getcwd())

from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, create_ddp_model, AMPTrainer, SimpleTrainer, default_setup, default_argument_parser
from detectron2.evaluation import verify_results, COCOEvaluator, COCOPanopticEvaluator, CityscapesSemSegEvaluator, SemSegEvaluator, LVISEvaluator, CityscapesInstanceEvaluator, DatasetEvaluators
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

from maskdino import add_maskdino_config, COCOPanopticNewBaselineDatasetMapper, SemanticSegmentorWithTTA

from panomito.mymapper import MyMapper
from panomito.panomito import PanoMito
from panomito.panomitoseg_encoder  import HighResMaskDINOEncoder
from panomito.panomitoseg_decoder  import SrMaskDINODecoder, MaskDINODecoderDynamicHead, HDMaskDINODecoder
from detectron2.modeling import META_ARCH_REGISTRY, SEM_SEG_HEADS_REGISTRY
from maskdino.modeling.transformer_decoder.maskdino_decoder import TRANSFORMER_DECODER_REGISTRY

# META_ARCH_REGISTRY.get("MyMaskDINO")
# SEM_SEG_HEADS_REGISTRY.get("HighResMaskDINOEncoder")
# TRANSFORMER_DECODER_REGISTRY.get("SrMaskDINODecoder")
# TRANSFORMER_DECODER_REGISTRY.get("MaskDINODecoderDynamicHead")
# TRANSFORMER_DECODER_REGISTRY.get("HDMaskDINODecoder")

class MyTrainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else MySimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        # TODO: release model conversion checkpointer from DINO to MaskDINO
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        # TODO: release GPU cluster submit scripts based on submitit for multi-node training

    @classmethod
    def build_train_loader(cls, cfg):
        # coco instance segmentation lsj new baseline
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)

        mapper = MyMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     """
    #     Returns:
    #         iterable

    #     It now calls :func:`detectron2.data.build_detection_test_loader`.
    #     Overwrite it if you'd like a different data loader.
    #     """
    #     # return build_detection_test_loader(cfg, dataset_name)
    #     return build_detection_test_loader(cfg, dataset_name, mapper=Test_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator = COCOEvaluator(dataset_name, output_dir=output_folder, max_dets_per_image=1000)

        return DatasetEvaluators([evaluator])

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        # accumation step
        accumulation_steps = cfg.SOLVER.ACCUMULATION_STEPS
        optimizer = GradientAccumulationOptimizer(optimizer, accumulation_steps)
        return optimizer
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
class MySimpleTrainer(SimpleTrainer):

    def __init__(self,model,data_loader,optimizer,gather_metric_period=1,zero_grad_before_forward=False,async_write_metrics=False):
        super().__init__(model,data_loader,optimizer,gather_metric_period,zero_grad_before_forward,async_write_metrics)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

class MyCOCOEvaluator(COCOEvaluator):
    def evaluate(self):
        results = super().evaluate()
        # Assume self._coco_eval stores COCOeval objects
        for coco_eval in self._coco_eval.values():
            coco_eval.params.areaRng = [
                [0**2, 4**2],   # small
                [4**2, 12**2], # medium
                [12**2, 1e5**2] # large
            ]
            coco_eval.params.areaRngLbl = ["small", "medium", "large"]
        return results

class GradientAccumulationOptimizer(Optimizer):
    def __init__(self, optimizer, accumulation_steps):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0

    def zero_grad(self):
        if self.step_counter % self.accumulation_steps == 0:
            self.optimizer.zero_grad()
        self.step_counter += 1

    def step(self, closure=None):
        self.optimizer.step(closure)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        #  self.optimizer.load_state_dict(state_dict)
        pass

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    args.config_file = '/home/kzlab/cellotype/CelloType/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.META_ARCHITECTURE = "MyMaskDINO"

    # cfg.MODEL.WEIGHTS = '/home/kzlab/cellotype/CelloType/models/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth'
    cfg.MODEL.WEIGHTS = '/home/kzlab/cellotype/CelloType/models/tissuenet_model_0019999.pth'
    # cfg.MODEL.WEIGHTS = ''

    cfg.MODEL.IN_CHANS = 3
    cfg.DATASETS.TRAIN = ("cell_train_R1","cell_train_R2","cell_train_R3","cell_train_R4","cell_train_R5_NRM1","cell_train_R6_NRM2")
    cfg.DATASETS.VAL = ("cell_val_R1", "cell_val_R2","cell_val_R3","cell_val_R4","cell_val_R5_NRM1","cell_val_R6_NRM2")
    cfg.DATASETS.TEST = ('cell_test_R1', 'cell_test_R2','cell_test_R3','cell_test_R4','cell_test_R5_NRM1','cell_test_R6_NRM2')
    cfg.OUTPUT_DIR = '/home/kzlab/P_Panomito/output/cellotype/test_png_R1+R2+R3+R4+NRM1+NRM2_COCOInstanceNewBaselineDatasetMapper_bz=4_newconnectivityloss=1_maskloss=default_NOmaskdiceloss_token=1000_maskdinopretrain'
    cfg.SOLVER.AMP.ENABLED = False
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"  # valid option
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.1  # max norm value

    cfg.INPUT.MIN_SCALE = 0.2
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.IMAGE_SIZE = 512
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 2000

    cfg.MODEL.MaskDINO.CONNECTIVITY_WEIGHT = 0.0 # 0.1
    cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.OUR_DICE_WEIGHT = 0.0 #5.0
    cfg.MODEL.MaskDINO.OUR_MASK_WEIGHT = 0.0 #5.0

    cfg.TEST.EVAL_PERIOD = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 30000
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 1000

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="cellotype")
    return cfg

def main(args):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    data_dir_EM = '/home/kzlab/P_Mitomics/MitoPanopticSeg/EM/cellmap_fluo_rle'
    data_dir_R1 = '/home/kzlab/P_Mitomics/MitoPanopticSeg/Fluo_R1'
    data_dir_R2 = '/home/kzlab/P_Mitomics/MitoPanopticSeg/Fluo_R2'
    data_dir_R3 = '/home/kzlab/P_Mitomics/MitoPanopticSeg/Fluo_R3'
    data_dir_R4 = '/home/kzlab/P_Mitomics/MitoPanopticSeg/Fluo_R4'
    data_dir_R5_NRM1 = '/home/kzlab/P_Mitomics/MitoPanopticSeg/Fluo_R5_NRM1'
    data_dir_R6_NRM2 = '/home/kzlab/P_Mitomics/MitoPanopticSeg/Fluo_R6_NRM2'

    for d in ["train","val", "test"]:
        DatasetCatalog.register("cell_" + d + "_EM",lambda d=d: np.load(os.path.join(data_dir_EM, '{}.npy'.format(d)), allow_pickle=True))
        # DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        MetadataCatalog.get("cell_" + d + "_EM").set(thing_classes=["Globule", "Tubule", "Loop", "Branch"])
        DatasetCatalog.register("cell_" + d + "_R1", lambda d=d: np.load(os.path.join(data_dir_R1, '{}.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d + "_R1").set(thing_classes=["Globule", "Tubule", "Loop", "Branch"])
        DatasetCatalog.register("cell_" + d + "_R2", lambda d=d: np.load(os.path.join(data_dir_R2, '{}.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d + "_R2").set(thing_classes=["Globule", "Tubule", "Loop", "Branch"])
        DatasetCatalog.register("cell_" + d + "_R3", lambda d=d: np.load(os.path.join(data_dir_R3, '{}.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d + "_R3").set(thing_classes=["Globule", "Tubule", "Loop", "Branch"])
        DatasetCatalog.register("cell_" + d + "_R4", lambda d=d: np.load(os.path.join(data_dir_R4, '{}.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d + "_R4").set(thing_classes=["Globule", "Tubule", "Loop", "Branch"])
        DatasetCatalog.register("cell_" + d + "_R5_NRM1", lambda d=d: np.load(os.path.join(data_dir_R5_NRM1, '{}.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d + "_R5_NRM1").set(thing_classes=["Globule", "Tubule", "Loop", "Branch"])
        DatasetCatalog.register("cell_" + d + "_R6_NRM2", lambda d=d: np.load(os.path.join(data_dir_R6_NRM2, '{}.npy'.format(d)), allow_pickle=True))
        MetadataCatalog.get("cell_" + d + "_R6_NRM2").set(thing_classes=["Globule", "Tubule", "Loop", "Branch"])

    args.resume = False

    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(MyTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    main(args)
