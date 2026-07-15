import glob
import os
import sys
if os.environ.get("LD_LIBRARY_PATH"):
    _clean_env = os.environ.copy()
    _clean_env.pop("LD_LIBRARY_PATH", None)
    os.execve(sys.executable, [sys.executable] + sys.argv, _clean_env)
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add project root to sys.path so `from panomito...` works from any working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
from predict import run_panomito_predict
from panomito.benchmark_utils import align_coco_ids
from panomito.evaluator_utils import evaluate_coco_results
from panomito.utils import load_ann_from_img_json, vis_instance_outlines


def _setup_log_tee(results_dir: str) -> str:
    """Mirror stdout/stderr to terminal and a timestamped log file under ``results_dir``."""
    log_path = os.path.join(results_dir, f"panomito_benchmark_{datetime.now():%Y%m%d_%H%M%S}.txt")
    log_fp = open(log_path, "w", encoding="utf-8")

    def _tee(stream):
        orig = stream

        class T:
            def write(self, s):
                orig.write(s)
                log_fp.write(s)
                log_fp.flush()

            def flush(self):
                orig.flush()
                log_fp.flush()

            def __getattr__(self, name):
                return getattr(orig, name)

        return T()

    sys.stdout, sys.stderr = _tee(sys.__stdout__), _tee(sys.__stderr__)
    print("Log file:", log_path, flush=True)
    return log_path


def run_panomito_benchmark(
    input_dir: str,
    output_path: str,
    gt_json: str,
    merge_thresh: float = 0.4,
    output_min_score: float = 0.0,
    postprocess: str = "use_NMS",
    crop_size: int = 512,
    merge_iou: float = 0.8,
    containment_threshold: float = 0.8,
    confidence_score: float = 0.4,
):
    """Run PanoMito benchmark: predict (same pipeline as ``predict.py``), evaluate, visualize.

    Args:
        input_dir: directory containing the .png images to predict on.
        output_path: destination path of the COCO-format JSON.
        gt_json: ground-truth COCO JSON for evaluation.
        merge_thresh: merge threshold passed to tile-instance merging.
        output_min_score: confidence threshold for prediction and final filtering.
        postprocess: ``"use_NMS"`` selects the NMS branch; any other value selects the plain branch.
        crop_size: tile size; 50% overlap is preserved.
        merge_iou: IoU threshold used when merging tile predictions.
        containment_threshold: intersection / min(area) threshold for merging instances.
        confidence_score: score threshold used in evaluation and visualization.
    """
    run_panomito_predict(
        input_dir=input_dir,
        output_path=output_path,
        merge_thresh=merge_thresh,
        output_min_score=output_min_score,
        postprocess=postprocess,
        crop_size=crop_size,
        merge_iou=merge_iou,
        containment_threshold=containment_threshold,
    )

    results_dir = os.path.dirname(output_path)
    panomito_pred_json = output_path
    panomito_adjust_id_json = os.path.join(results_dir,output_path.replace(".json", "_adjust_id.json"))
    align_coco_ids(gt_json, panomito_pred_json, panomito_adjust_id_json)
    evaluate_coco_results(
        gt_json, panomito_adjust_id_json, use_F1score=True, confidence_score=confidence_score
    )

    with open(panomito_pred_json, 'r') as f:
        coco_data = json.load(f)

    vis_dir = os.path.join(results_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    for image in coco_data["images"][:6]:
        image_name = image["file_name"]
        if os.path.isabs(image_name):
            image_path = image_name
        else:
            image_path = os.path.join(input_dir, os.path.basename(image_name))
        _, annotations = load_ann_from_img_json(image_path, panomito_pred_json)
        annotations['annotations'] = [
            ann for ann in annotations['annotations'] if ann.get('score', 1.0) > confidence_score
        ]
        vis_instance_outlines(
            annotations,
            fillcolor=[0, 255, 0],
            linewidth=2,
            verbose=False,
            save_path=os.path.join(
                vis_dir,
                f"{os.path.basename(image_name)}".replace('.png', '_digital_structure_image.png'),
            ),
        )


if __name__ == "__main__":
    _merge_thresh = 0.4
    _output_min_score = 0.0
    _postprocess = "use_NMS"
    _crop_size = 512
    _merge_iou = 0.8
    _confidence_score = 0.4
    _input_dir = f"{_PROJECT_ROOT}/data/test"
    _results_dir = os.path.join(_PROJECT_ROOT, "results", "panomito_benchmark")
    os.makedirs(_results_dir, exist_ok=True)
    _output_path = (
        f"{_results_dir}/_sub_predictor_{_postprocess}_dataset=all_merge_thresh={_merge_thresh}"
        f"_c={_output_min_score}_merge_q300.json"
    )
    _gt_json = f"{_PROJECT_ROOT}/data/test/_test_subdataset_gt.json"

    _setup_log_tee(_results_dir)

    run_panomito_benchmark(
        input_dir=_input_dir,
        output_path=_output_path,
        gt_json=_gt_json,
        merge_thresh=_merge_thresh,
        output_min_score=_output_min_score,
        postprocess=_postprocess,
        crop_size=_crop_size,
        merge_iou=_merge_iou,
        containment_threshold=0.8,
        confidence_score=_confidence_score,
    )
