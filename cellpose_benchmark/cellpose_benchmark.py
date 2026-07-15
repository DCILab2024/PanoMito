import os
import sys

if os.environ.get("LD_LIBRARY_PATH"):
    _clean_env = os.environ.copy()
    _clean_env.pop("LD_LIBRARY_PATH", None)
    os.execve(sys.executable, [sys.executable] + sys.argv, _clean_env)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from cellpose import models
from cellpose.io import imread_2D
from tqdm import tqdm

from panomito.benchmark_utils import align_coco_ids, npy_to_coco
from panomito.evaluator_utils import evaluate_coco_results


def run_cellpose_benchmark(
    input_dir: str,
    results_dir: str,
    gt_json: str,
    model_path: str,
    cellprob_thresholds=None,
    flow_thresholds=None,
    diameter=None,
    ):
    """Run Cellpose inference on test images and evaluate against ground truth.

    Args:
        input_dir: directory containing input PNG images.
        results_dir: output directory for NPY masks and COCO JSON files.
        gt_json: ground-truth COCO JSON for evaluation.
        model_path: path to the Cellpose model under ``./MODEL/``.
        cellprob_thresholds: list of cell probability thresholds.
        flow_thresholds: list of flow thresholds.
        diameter: expected object diameter passed to Cellpose.
    """
    if cellprob_thresholds is None:
        cellprob_thresholds = [-1.0]
    if flow_thresholds is None:
        flow_thresholds = [0.4]
    if diameter is None:
        diameter = 25

    normalize_default = {
        "lowhigh": None,
        "percentile": [1.0, 99.0],
        "normalize": True,
        "norm3D": True,
        "sharpen_radius": 0,
        "smooth_radius": 0,
        "tile_norm_blocksize": 0,
        "tile_norm_smooth3D": 1,
        "invert": False,
    }

    os.makedirs(results_dir, exist_ok=True)
    npy_dir = os.path.join(results_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    filelist = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png"))
    for flow_threshold in tqdm(flow_thresholds, desc="Processing flow thresholds"):
        for cellprob_threshold in tqdm(cellprob_thresholds, desc="Processing cellprob thresholds", leave=False):
            for fname in tqdm(filelist, desc="Processing images", leave=False):
                image_path = os.path.join(input_dir, fname)
                data = imread_2D(image_path)
                data = data[np.newaxis, ...]

                data_min = data.min()
                data_max = data.max()
                data = data.astype(np.float32)
                data -= data_min
                if data_max > data_min + 1e-3:
                    data /= (data_max - data_min)
                data *= 255
                data = data.copy().squeeze()

                masks, flows = model.eval(
                    data,
                    diameter=diameter,
                    normalize=normalize_default,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                )[:2]

                cellpix = masks[np.newaxis, :, :]
                flows = [flows[n][np.newaxis, ...] for n in range(len(flows))]
                dat = {
                    "masks": cellpix.squeeze().astype(np.int32),
                    "filename": image_path,
                    "flows": flows,
                    "outlines": None,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "normalize_params": normalize_default,
                }
                base = os.path.splitext(fname)[0]
                np.save(os.path.join(npy_dir, f"{base}_seg.npy"), dat)

    cellpose_pred_json = os.path.join(results_dir, "cellpose_pred_coco.json")
    cellpose_adjust_id_json = os.path.join(results_dir, "cellpose_pred_coco_adjust_id.json")

    npy_to_coco(npy_dir, cellpose_pred_json)
    align_coco_ids(gt_json, cellpose_pred_json, cellpose_adjust_id_json)
    evaluate_coco_results(gt_json, cellpose_adjust_id_json, use_F1score=True)


if __name__ == "__main__":
    _input_dir = os.path.join(_PROJECT_ROOT, "data", "test")
    _results_dir = os.path.join(_PROJECT_ROOT, "results", "cellpose_pre")
    _gt_json = os.path.join(_PROJECT_ROOT, "data", "test", "_test_subdataset_gt.json")
    _model_path = os.path.join(_PROJECT_ROOT, "MODEL", "cellpose_model")

    run_cellpose_benchmark(
        input_dir=_input_dir,
        results_dir=_results_dir,
        gt_json=_gt_json,
        model_path=_model_path,
    )
