import os
import sys
if os.environ.get("LD_LIBRARY_PATH"):
    _clean_env = os.environ.copy()
    _clean_env.pop("LD_LIBRARY_PATH", None)
    os.execve(sys.executable, [sys.executable] + sys.argv, _clean_env)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from predict import run_panomito_predict
from panomito.cluster import AutoencoderLabelTrainer
from panomito.cluster_utils import export_resized_masks_from_coco, export_cluster_morphology_metrics


def run_predict_cluster_analysis(
    input_dir: str,
    output_json_path: str,
    model_dir: str,
    mito_instance_dir: str,
    cluster_output_dir: str,
    merge_thresh: float = 0.4,
    output_min_score: float = 0.0,
    postprocess: str = "use_NMS",
    crop_size: int = 512,
    merge_iou: float = 0.8,
    containment_threshold: float = 0.8,
    merge_map=None,
):
    """Run PanoMito segmentation (same as ``predict.py``) followed by clustering.

    Args:
        input_dir: directory containing the .png images to predict on.
        output_json_path: destination path of the COCO-format JSON.
        model_dir: directory containing PanoMitoCluster.pth.
        mito_instance_dir: output folder for resized instance masks.
        cluster_output_dir: output folder for clustering results.
        merge_thresh: merge threshold passed to tile-instance merging.
        output_min_score: confidence threshold for prediction and final filtering.
        postprocess: ``"use_NMS"`` selects the NMS branch; any other value selects the plain branch.
        crop_size: tile size; 50% overlap is preserved.
        merge_iou: IoU threshold used when merging tile predictions.
        containment_threshold: intersection / min(area) threshold for merging instances.
        merge_map: optional cluster label merge mapping passed to ``ae_trainer.cluster``.
    """

    run_panomito_predict(
        input_dir=input_dir,
        output_path=output_json_path,
        merge_thresh=merge_thresh,
        output_min_score=output_min_score,
        postprocess=postprocess,
        crop_size=crop_size,
        merge_iou=merge_iou,
        containment_threshold=containment_threshold,
    )

    export_resized_masks_from_coco(
        coco_json_path=output_json_path,
        image_root=f"{input_dir}/",
        out_root=mito_instance_dir,
        filter_mito=False,
        postfix="all_data",
        verbose=True,
    )

    ae_trainer = AutoencoderLabelTrainer(root_dir=f"{model_dir}/", output_dir=f"{_PROJECT_ROOT}/")
    ae_trainer.load_model(model_path=f"{model_dir}/PanoMitoCluster.pth")
    ae_trainer.cluster(
        image_folder=mito_instance_dir,
        output_csv="mito_clusters.csv",
        output_dir=cluster_output_dir,
        merge_map=None,
    )

    export_cluster_morphology_metrics(
    cluster_dir=cluster_output_dir,
    output_dir=os.path.dirname(output_json_path),
)


if __name__ == "__main__":
    _merge_thresh = 0.4
    _output_min_score = 0.0
    _postprocess = "use_NMS"
    _crop_size = 512
    _merge_iou = 0.8
    _input_dir = f"{_PROJECT_ROOT}/data/test"
    _results_dir = os.path.join(_PROJECT_ROOT, "results", "predict_cluster_analysis")
    os.makedirs(_results_dir, exist_ok=True)
    _output_json_path = (
        f"{_results_dir}/_sub_predictor_{_postprocess}_dataset=all_merge_thresh={_merge_thresh}"
        f"_c={_output_min_score}_merge_q300.json"
    )

    run_predict_cluster_analysis(
        input_dir=_input_dir,
        output_json_path=_output_json_path,
        model_dir=f"{_PROJECT_ROOT}/MODEL",
        mito_instance_dir=f"{_results_dir}/MitoInstance",
        cluster_output_dir=f"{_results_dir}/Cluster",
        merge_thresh=_merge_thresh,
        output_min_score=_output_min_score,
        postprocess=_postprocess,
        crop_size=_crop_size,
        merge_iou=_merge_iou,
        containment_threshold=0.8,
    )
