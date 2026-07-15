import glob
import os
import sys
if os.environ.get("LD_LIBRARY_PATH"):
    _clean_env = os.environ.copy()
    _clean_env.pop("LD_LIBRARY_PATH", None)
    os.execve(sys.executable, [sys.executable] + sys.argv, _clean_env)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from skimage import io
import warnings
warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")
from panomito.predict import PanomitoPredictor
from detectron2.structures import Instances
import numpy as np
from tqdm import tqdm
import tifffile as tf
from panomito.utils import split_image, instance_to_coco_EM, instance_to_coco_only_maxconnection_NMS_4cvat_EM, merge_instances_without_cls_bbox_filter
import json, gc
import torch


def run_panomito_predict(
    input_dir: str,
    output_path: str,
    merge_thresh: float = 0.0,
    output_min_score: float = 0.0,
    postprocess: str = "use_NMS",
    crop_size: int = 256,
    merge_iou: float = 0.8,
    containment_threshold: float = 0.8,
):
    """Run PanoMito tiled prediction over PNG images in ``input_dir`` and dump COCO JSON.

    Logic is preserved verbatim from the original script; only constants
    have been promoted to parameters and module-level code wrapped in a function.

    Args:
        input_dir: directory containing the .png images to predict on.
        output_path: destination path of the COCO-format JSON.
        thresh: confidence threshold passed to the predictor (and used to filter merges).
        postprocess: ``"use_NMS"`` selects the NMS branch; any other value selects the plain branch.
        crop_size: tile size; the original 50% overlap is preserved.
        merge_iou: IoU threshold used when merging tile predictions.
    """

    # Load segmentation model.
    model = PanomitoPredictor(
        model_path=f'{_PROJECT_ROOT}/MODEL/PanoMitoSeg.pth',
        confidence_thresh=output_min_score,
        max_det=300,
        device='cuda',
        config_path=f'{_PROJECT_ROOT}/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml')

    file_list = glob.glob(f"{input_dir}/*.png")
    file_list.sort()
    coco_data = []

    for image_file in tqdm(file_list, desc = 'for_all_images'):
        img = io.imread(image_file)  # [H, W, 3]
        input_height, image_width = img.shape[:2]
        outputs = []
        # Split large image into overlapping tiles (50% overlap).
        subimages = split_image(img, tile_size=crop_size, overlap=int(crop_size/2))
        for (y1, y2, x1, x2), subimage in tqdm(subimages, desc = 'for_sub_images', leave=False):
            output = model.predictor(subimage)
            # Store tile offset for merging predictions back to full image.
            output['instances'].set("position", torch.tensor([[y1, y2, x1, x2]] * output['instances'].pred_masks.shape[0], device="cuda"))
            output["instances"] = output["instances"][output["instances"].scores > output_min_score]
            if output["instances"].has("pred_masks"):
                output["instances"].pred_masks = output["instances"].pred_masks > 0.5 
            gc.collect()
            torch.cuda.empty_cache()
            if outputs == []:
                outputs = output 
                # print(len(output["instances"]))                  
            else:        
                outputs['instances'] = Instances.cat([outputs['instances'], output["instances"]])
                # print(len(outputs["instances"]))
        if len(outputs["instances"]) == 0:
            continue

        # Merge overlapping instances from all tiles into full-image coordinates.
        merge_masks, merge_classes, merge_bbox, merge_score = merge_instances_without_cls_bbox_filter(outputs['instances'], whole_image_size=(input_height, image_width), iou_threshold=merge_iou, containment_threshold=containment_threshold, merge_thr=merge_thresh)
        merge_outputs = {'instances': Instances((input_height, image_width))}
        merge_outputs['instances'].pred_masks = merge_masks
        merge_outputs['instances'].pred_classes = merge_classes
        merge_outputs['instances'].pred_boxes = merge_bbox
        merge_outputs['instances'].scores = merge_score
        merge_outputs["instances"] = merge_outputs["instances"][merge_outputs["instances"].scores >= output_min_score]  
        instances = merge_outputs["instances"].to('cpu')
        instances_list = []
        instances_list.append(instances)
        # Convert instances to COCO JSON (optional NMS post-processing).
        if postprocess == "use_NMS":
            coco_data = instance_to_coco_only_maxconnection_NMS_4cvat_EM(instances_list, [image_file], coco_data)
        else:
            coco_data = instance_to_coco_EM(instances_list, [image_file], coco_data)
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)        
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"COCO format json save to {output_path}")


if __name__ == "__main__":
    # Default settings when running this script directly.
    _merge_thresh = 0.4
    _output_min_score = 0.4
    _postprocess = "use_NMS"
    _crop_size = 512
    _merge_iou = 0.8
    _input_dir = f"{_PROJECT_ROOT}/data/test"
    _output_path = f"{_PROJECT_ROOT}/results/predict_results/_sub_predictor_{_postprocess}_dataset=all_merge_thresh={_merge_thresh}_c={_output_min_score}_merge_q300.json"

    run_panomito_predict(
        input_dir=_input_dir,
        output_path=_output_path,
        merge_thresh=_merge_thresh,
        output_min_score = _output_min_score,
        postprocess=_postprocess,
        crop_size=_crop_size,
        merge_iou=_merge_iou,
        containment_threshold=0.8,
    )
