import os
import sys

# LD_LIBRARY_PATH is read by the dynamic linker at process start. Clearing it
# later via os.environ does not fix mixed CUDA libraries on Linux; re-exec if set.
if os.environ.get("LD_LIBRARY_PATH"):
    _clean_env = os.environ.copy()
    _clean_env.pop("LD_LIBRARY_PATH", None)
    os.execve(sys.executable, [sys.executable] + sys.argv, _clean_env)

# Pin to a single GPU before any CUDA-aware import (torch via detectron2) runs.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
# MaskDINO position encoding triggers pytorch's __floordiv__ UserWarning on
# every forward pass; silence to keep training logs readable.
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

import detectron2.utils.comm as comm
from detectron2.config import get_cfg, CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_setup, default_argument_parser
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from panomito.mytrainer import MyTrainer
from panomito.myconfig import set_panomito_cfg

# Resolve project paths relative to this script so the code stays portable.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def setup(args):
    """Build the detectron2 / MaskDINO config used for PanoMito training."""
    cfg = get_cfg()

    # Register config schemas before merging the YAML file.
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)

    args.config_file = os.path.join(
        _PROJECT_ROOT, 'configs', 'maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'
    )
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # ----- Model architecture -------------------------------------------------
    cfg.MODEL.META_ARCHITECTURE = "PanoMito"
    cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "HDMaskDINODecoder"
    # cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINODecoderDynamicHead"

    cfg.MODEL.IN_CHANS = 3
    cfg.MODEL.WEIGHTS = os.path.join(
        _PROJECT_ROOT, 'pretrain_model', 'tissuenet_model_0019999.pth'
    )

    # ----- Datasets -----------------------------------------------------------
    cfg.DATASETS.TRAIN = ("mito_train",)
    cfg.DATASETS.TEST = ("mito_test",)

    # ----- Input pipeline (multi-scale 512 px crops) --------------------------
    cfg.INPUT.MIN_SCALE = 0.2
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.IMAGE_SIZE = 512
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 2000
    cfg.INPUT.BLUR = False

    # ----- Loss weights -------------------------------------------------------
    cfg.MODEL.MaskDINO.CONNECTIVITY_WEIGHT = 0.1  # 0.1

    cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.DICE_WEIGHT = 8.0
    cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.0   # default: 5.0
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = 5.0  # default: 5.0

    # Hungarian matcher costs mirror loss weights for stable assignment.
    cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_DICE_WEIGHT = 8.0
    cfg.MODEL.MaskDINO.COST_BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT = 5.

    cfg.MODEL.MaskDINO.DN_NOISE_SCALE = 0.2
    cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = 'mask2box'

    # ----- Solver / schedule --------------------------------------------------
    cfg.SOLVER.AMP.ENABLED = False
    cfg.SOLVER.ACCUMULATION_STEPS = 16
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.1

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00001
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.STEPS = (5000, 17000)

    cfg.TEST.EVAL_PERIOD = 5000
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000

    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="PanoMitoNet")
    return cfg


def main(n_query=100):
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    args.resume = True

    # ----- Dataset registration ----------------------------------------------
    # Each split is a single .npy holding RLE-encoded instance annotations.
    train_data_dir = os.path.join(_PROJECT_ROOT, 'data', 'train')
    test_data_dir = os.path.join(_PROJECT_ROOT, 'data', 'test')
    # data_dir_Primary = os.path.join(_PROJECT_ROOT, 'data', 'train', 'Primary', 'crop_256')

    train_dataset_name = "mito_train"
    test_dataset_name = "mito_test"

    # Re-register cleanly so main() can be invoked multiple times safely.
    if train_dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(train_dataset_name)
    if test_dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(test_dataset_name)

    DatasetCatalog.register(
        train_dataset_name,
        lambda: np.load(
            os.path.join(train_data_dir, '_train_subdataset_gt_nocate_rle.npy'), allow_pickle=True
        ),
    )
    MetadataCatalog.get(train_dataset_name).set(thing_classes=["Mito"])

    DatasetCatalog.register(
        test_dataset_name,
        lambda: np.load(
            os.path.join(test_data_dir, '_test_subdataset_gt_nocate_rle.npy'), allow_pickle=True
        ),
    )
    MetadataCatalog.get(test_dataset_name).set(thing_classes=["Mito"])

    # ----- Build config and resolve per-run output dir -----------------------
    cfg = setup(args)
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = n_query

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = os.path.join(_PROJECT_ROOT, 'output', script_name)
    cfg.OUTPUT_DIR = output_dir

    # ----- PanoMito-specific extra config ------------------------------------
    cfg.PANOMITO = CfgNode()
    cfg.PANOMITO.SemLoss = True
    cfg.PANOMITO.AllSemLoss = True
    cfg.PANOMITO.SemLossWeight = 1
    cfg.PANOMITO.SemLossSigmoid = False
    set_panomito_cfg(cfg)

    cfg.freeze()
    print("Command cfg:", cfg)

    # ----- Train -------------------------------------------------------------
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    main()
