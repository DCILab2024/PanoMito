from __future__ import annotations
import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import pycocotools.mask as maskUtils
import torch
from detectron2.structures import BoxMode
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
from scipy import ndimage
from PIL import Image
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, binary_dilation, gaussian_filter
import tifffile
import anndata
from importlib import import_module
from skimage.morphology import skeletonize
from scipy.ndimage import label


def instance_to_coco_EM(instance_list, image_files, coco_data):

	if coco_data ==[]:
		
		coco_data = {
		"categories": [{"id": 1, "name": "Mito", "supercategory": "mito"}],
		"images": [],  
		"annotations": []  
		}
	else:
		coco_data = coco_data

    
	image_id = len(coco_data["images"]) + 1
	annotation_id = len(coco_data["annotations"]) + 1

	for img_file, instances in zip(image_files, instance_list):
		
		coco_data["images"].append({
			"id": image_id,
			"file_name": os.path.basename(img_file),
			"width": instances.pred_masks.shape[2] if instances else 0,
			"height": instances.pred_masks.shape[1] if instances else 0,
		})

		
		for i in range(instances.pred_masks.shape[0]):
			instance = instances[i]
			
			pred_masks = instance.pred_masks.numpy().copy().squeeze().astype(np.uint8)
			pred_scores = instance.scores.numpy().copy().squeeze().astype(np.float32)
			pred_classes = instance.pred_classes.numpy().copy().squeeze().astype(np.float32)
			pred_boxes = instance.pred_boxes.tensor.numpy()
			pred_boxes = BoxMode.convert(pred_boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
			pred_boxes = pred_boxes.tolist()
			if isinstance(pred_masks, torch.Tensor):
				mask = pred_masks.cpu().numpy().astype(np.uint8)
			else:
				mask = pred_masks.astype(np.uint8)
			
			rle = maskUtils.encode(np.asfortranarray(mask))
			rle["counts"] = rle["counts"].decode("utf-8") 


			coco_data["annotations"].append({
				"id": annotation_id,
				"image_id": image_id,
				"category_id": int(pred_classes + 1),
				"segmentation": rle,
				"bbox": pred_boxes[0],
				"area": float(pred_boxes[0][2] * pred_boxes[0][3]),
				"iscrowd": 0,
				"score": float(pred_scores)
			})
			annotation_id += 1

		image_id += 1
	return coco_data
    
def instance_to_coco_only_maxconnection_NMS_4cvat_EM(instance_list, image_files, coco_data):
    """Convert Detectron2 Instances to COCO JSON.
        Post-process: keep the largest connected component for each mask, then remove
        overlapping masks using greedy IoU-based suppression (order-dependent, not score-sorted).
        Appends into coco_data (or initializes if coco_data == []).
        """

    if coco_data ==[]:
    
        coco_data = {
        "categories": [{"id": 1, "name": "Mito", "supercategory": "mito"}],
        "images": [],  
        "annotations": [] 
        }
    else:
        coco_data = coco_data
        
    all_annotations = []
    filtered_annotations = []

    image_id = len(coco_data["images"]) + 1
    annotation_id = len(coco_data["annotations"]) + 1

    for img_file, instances in zip(image_files, instance_list):

        coco_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(img_file),
            "width": instances.pred_masks.shape[2] if instances else 0,
            "height": instances.pred_masks.shape[1] if instances else 0,
        })


        for i in range(instances.pred_masks.shape[0]):
            instance = instances[i]
            pred_masks = instance.pred_masks.numpy().copy().squeeze().astype(np.uint8)

            pred_scores = instance.scores.numpy().copy().squeeze().astype(np.float32)
            pred_classes = instance.pred_classes.numpy().copy().squeeze().astype(np.float32)
            pred_boxes = instance.pred_boxes.tensor.numpy()
            pred_boxes = BoxMode.convert(pred_boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            pred_boxes = pred_boxes.tolist()
            if isinstance(pred_masks, torch.Tensor):
                mask = pred_masks.cpu().numpy().astype(np.uint8)
            else:
                mask = pred_masks.astype(np.uint8)
            

            num_labels, labels = cv2.connectedComponents(mask)
            if num_labels <= 2:  
                mask = mask
            else:

                max_label = np.argmax([np.sum(labels == i) for i in range(1, num_labels)]) + 1
                mask = labels == max_label
                mask = mask.astype(np.uint8)
            
            all_annotations.append({
                "image_id": image_id,
                "category_id": int(pred_classes + 1),
                "segmentation": mask,
                "bbox": pred_boxes[0],
                "area": float(pred_boxes[0][2] * pred_boxes[0][3]),
                "iscrowd": 0,
                "score": float(pred_scores)
            })
        # print(f"{img_file}------------------------------")    
        # print(f"before NMS：{len(all_annotations)}")
        filtered_annotations = apply_mask_nms_to_coco_annotations(all_annotations, iou_threshold=0.3)
        # print(f"after NMS：{len(filtered_annotations)}")
        # print(f"------------------------------") 
            
        for ann in filtered_annotations:
            mask = ann["segmentation"]
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8") 

            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "segmentation": rle,
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": 0,
                "score": ann["score"]
            })
            annotation_id += 1

        image_id += 1
    return coco_data 
    
def calculate_mask_iou(mask1, mask2):

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def apply_mask_nms_to_coco_annotations(coco_annotations, iou_threshold=0.5):

    keep_annotations = []
    

    while len(coco_annotations) > 0:
        current_ann = coco_annotations.pop(0)
        keep_annotations.append(current_ann)
        

        to_remove = []
        current_mask = current_ann['segmentation']
        
        for i, other_ann in enumerate(coco_annotations):
            other_mask = other_ann['segmentation']
            iou = calculate_mask_iou(current_mask, other_mask)
            

            if iou > iou_threshold:
                to_remove.append(i)
        

        for i in sorted(to_remove, reverse=True):
            coco_annotations.pop(i)
    
    return keep_annotations
  
  
def merge_instances_without_cls_bbox_filter(
    instances,
    whole_image_size=(4096, 4096),
    iou_threshold=0.5,
    containment_threshold=0.8,
    pixel_min=5,
    device='cuda',
    merge_thr=0.0,
):
    """Merge tiled instance predictions into full-image instances.

    Combines overlapping detections from adjacent image tiles without filtering
    by class or bounding-box constraints. Instances are grouped by tile row and
    column, merged in two passes (horizontal then vertical), and returned as
    full-resolution masks aligned to ``whole_image_size``.

    Predictions with confidence above ``merge_thr`` participate in overlap-based
    merging (mask IoU or intersection-over-minimum-area). Lower-confidence
    predictions are kept unchanged and appended to the output. Merged instance
    scores are averaged over contributing tiles.

    Args:
        instances: Detectron2-style instance container with ``position``,
            ``pred_masks``, ``pred_classes``, ``scores``, and ``pred_boxes``.
        whole_image_size: Full image height and width for embedding tile masks.
        iou_threshold: Minimum mask IoU to merge two instances.
        containment_threshold: Minimum intersection / min(area) to merge instances.
        pixel_min: Skip masks with at most this many foreground pixels.
        device: Torch device for intermediate mask operations ('cuda' or 'cpu').
        merge_thr: Confidence threshold; instances below this bypass merging.

    Returns:
        Tuple of (pred_mask, pred_classes, pred_bbox, score) with whole-image
        masks, class labels, bounding boxes, and confidence scores.
    """

    grouped_by_y1 = {}
    unique_y1 = set()
    kept_as_is = []

    for (y1, y2, x1, x2), mask, classes, score, bbox in zip(
        instances.position, instances.pred_masks, instances.pred_classes,
        instances.scores, instances.pred_boxes
    ):
        if mask is None or mask.max() == 0:
            continue

        ys, xs = torch.where(mask)
        bbox[0] += x1
        bbox[1] += y1
        bbox[2] += x1
        bbox[3] += y1
        if len(ys) > pixel_min:
            if float(score) > merge_thr:
                instance_data = {
                    'mask': mask.to(torch.uint8),
                    'class': classes,
                    'pixel_count': len(ys),
                    'position': [y1, x1, y2, x2],
                    'instance_position': [y1, x1, y2, x2],
                    'bbox': bbox,
                    'score': score,
                    'score_count': 1,
                }
                y1_key = y1.item()
                grouped_by_y1.setdefault(y1_key, []).append(instance_data)
                unique_y1 = list(grouped_by_y1.keys())
            else:
                kept_as_is.append({
                    'mask': mask.to(torch.uint8),
                    'class': classes,
                    'pixel_count': len(ys),
                    'position': [y1, x1, y2, x2],
                    'instance_position': [y1, x1, y2, x2],
                    'bbox': bbox,
                    'score': score,
                    'score_count': 1,
                })

    unique_y1 = list(unique_y1)
    unique_y1.sort()

    if grouped_by_y1 == {} and not kept_as_is:
        pred_mask = instances.pred_masks
        pred_classes = instances.pred_classes
        pred_bbox = instances.pred_boxes
        score = instances.scores
        return pred_mask, pred_classes, pred_bbox, score

    all_instances = []
    all_instances_y_merge = []

    if grouped_by_y1:
        for y1_key in tqdm(unique_y1, desc='row_merge', leave=False):
            grouped_by_x1 = {}
            for inst_y1 in grouped_by_y1.get(y1_key, []):
                grouped_by_x1.setdefault(str(inst_y1['position'][1]), []).append(inst_y1)
            all_x_keys = list(grouped_by_x1.keys())

            current_all_instances_y1_x1 = grouped_by_x1.get(all_x_keys[0], [])
            if current_all_instances_y1_x1 == []:
                print("Error")

            for i in range(1, len(all_x_keys)):
                other_all_instances_y1_x1 = grouped_by_x1.get(all_x_keys[i], [])
                if other_all_instances_y1_x1 == []:
                    continue
                merged_ids = []

                for k, current_instance in enumerate(current_all_instances_y1_x1):
                    current_mask = current_instance['mask']
                    current_class = current_instance['class']
                    current_pixels = current_instance['pixel_count']
                    current_position = current_instance['position']
                    current_instance_position = current_instance['instance_position']
                    current_score = current_instance['score']
                    current_score_count = current_instance['score_count']
                    current_bbox = current_instance['bbox']

                    for j, other_instance in enumerate(other_all_instances_y1_x1):
                        if j in merged_ids:
                            continue

                        other_inst_bbox = other_instance['bbox']
                        other_inst_mask = other_instance['mask']
                        other_inst_position = other_instance['position']
                        other_inst_instance_position = other_instance['instance_position']
                        other_inst_class = other_instance['class']
                        other_inst_pixels = other_instance['pixel_count']
                        other_inst_score = other_instance['score']
                        other_inst_score_count = other_instance['score_count']

                        if current_bbox[2] < other_inst_bbox[0] or current_bbox[0] > other_inst_bbox[2]:
                            continue
                        if current_bbox[3] < other_inst_bbox[1] or current_bbox[1] > other_inst_bbox[3]:
                            continue

                        current_whole_mask = torch.zeros(whole_image_size, device=device)
                        other_inst_whole_mask = torch.zeros(whole_image_size, device=device)
                        current_whole_mask[
                            current_instance_position[0]:current_instance_position[2],
                            current_instance_position[1]:current_instance_position[3]
                        ] = current_mask
                        other_inst_whole_mask[
                            other_inst_instance_position[0]:other_inst_instance_position[2],
                            other_inst_instance_position[1]:other_inst_instance_position[3]
                        ] = other_inst_mask

                        y_min = max(current_position[0], other_inst_position[0])
                        x_min = max(current_position[1], other_inst_position[1])
                        y_max = min(current_position[2], other_inst_position[2])
                        x_max = min(current_position[3], other_inst_position[3])
                        mask1_overlap = current_whole_mask[y_min:y_max, x_min:x_max]
                        mask2_overlap = other_inst_whole_mask[y_min:y_max, x_min:x_max]
                        if mask1_overlap is None or mask2_overlap is None:
                            continue

                        intersection = torch.logical_and(mask1_overlap, mask2_overlap).sum()
                        union = torch.logical_or(mask1_overlap, mask2_overlap).sum()
                        iou = intersection / (union + 1e-6)

                        cur_area = float(current_pixels) if not torch.is_tensor(current_pixels) else float(current_pixels.item())
                        oth_area = float(other_inst_pixels) if not torch.is_tensor(other_inst_pixels) else float(other_inst_pixels.item())
                        min_area = min(cur_area, oth_area)
                        iomin = intersection / (min_area + 1e-6)

                        should_merge = (iou > iou_threshold) or (iomin > containment_threshold)

                        if should_merge:
                            new_total_count = current_score_count + other_inst_score_count
                            current_score = (
                                current_score * current_score_count
                                + other_inst_score * other_inst_score_count
                            ) / new_total_count
                            current_score_count = new_total_count

                            current_instance_position = [
                                min(current_instance_position[0], other_inst_instance_position[0]),
                                min(current_instance_position[1], other_inst_instance_position[1]),
                                max(current_instance_position[2], other_inst_instance_position[2]),
                                max(current_instance_position[3], other_inst_instance_position[3]),
                            ]

                            current_merge_mask = torch.logical_or(
                                current_whole_mask, other_inst_whole_mask
                            ).to(torch.uint8)
                            current_mask = current_merge_mask[
                                current_instance_position[0]:current_instance_position[2],
                                current_instance_position[1]:current_instance_position[3]
                            ]
                            current_pixels = current_mask.sum()

                            current_bbox = torch.tensor([
                                min(current_bbox[0], other_inst_bbox[0]),
                                min(current_bbox[1], other_inst_bbox[1]),
                                max(current_bbox[2], other_inst_bbox[2]),
                                max(current_bbox[3], other_inst_bbox[3]),
                            ], device=device)

                            merged_ids.append(j)

                        new_inst = {
                            'mask': current_mask,
                            'class': current_class,
                            'pixel_count': current_pixels,
                            'position': current_position,
                            'instance_position': current_instance_position,
                            'bbox': current_bbox,
                            'score': current_score,
                            'score_count': current_score_count,
                        }
                        current_all_instances_y1_x1[k] = new_inst

                for n in sorted(merged_ids, reverse=True):
                    other_all_instances_y1_x1.pop(n)
                current_all_instances_y1_x1.extend(other_all_instances_y1_x1)

                current_position = [
                    min(current_position[0], other_inst_position[0]),
                    min(current_position[1], other_inst_position[1]),
                    max(current_position[2], other_inst_position[2]),
                    max(current_position[3], other_inst_position[3]),
                ]

                for current_instance in current_all_instances_y1_x1:
                    current_instance['position'] = current_position

            all_instances_y_merge.append(current_all_instances_y1_x1)

        current_all_instances_y1 = all_instances_y_merge[0]
        for i in tqdm(range(1, len(all_instances_y_merge)), desc='column_merge', leave=False):
            other_all_instances_y1 = all_instances_y_merge[i]
            merged_ids = []

            for k, current_instance in enumerate(current_all_instances_y1):
                current_mask = current_instance['mask']
                current_class = current_instance['class']
                current_pixels = current_instance['pixel_count']
                current_position = current_instance['position']
                current_instance_position = current_instance['instance_position']
                current_score = current_instance['score']
                current_score_count = current_instance['score_count']
                current_bbox = current_instance['bbox']

                for j, other_instance in enumerate(other_all_instances_y1):
                    if j in merged_ids:
                        continue

                    other_inst_bbox = other_instance['bbox']
                    other_inst_mask = other_instance['mask']
                    other_inst_position = other_instance['position']
                    other_inst_instance_position = other_instance['instance_position']
                    other_inst_class = other_instance['class']
                    other_inst_pixels = other_instance['pixel_count']
                    other_inst_score = other_instance['score']
                    other_inst_score_count = other_instance['score_count']

                    if current_bbox[2] < other_inst_bbox[0] or current_bbox[0] > other_inst_bbox[2]:
                        continue
                    if current_bbox[3] < other_inst_bbox[1] or current_bbox[1] > other_inst_bbox[3]:
                        continue

                    current_whole_mask = torch.zeros(whole_image_size, device=device)
                    other_inst_whole_mask = torch.zeros(whole_image_size, device=device)
                    current_whole_mask[
                        current_instance_position[0]:current_instance_position[2],
                        current_instance_position[1]:current_instance_position[3]
                    ] = current_mask
                    other_inst_whole_mask[
                        other_inst_instance_position[0]:other_inst_instance_position[2],
                        other_inst_instance_position[1]:other_inst_instance_position[3]
                    ] = other_inst_mask

                    y_min = max(current_position[0], other_inst_position[0])
                    x_min = max(current_position[1], other_inst_position[1])
                    y_max = min(current_position[2], other_inst_position[2])
                    x_max = min(current_position[3], other_inst_position[3])
                    mask1_overlap = current_whole_mask[y_min:y_max, x_min:x_max]
                    mask2_overlap = other_inst_whole_mask[y_min:y_max, x_min:x_max]
                    if mask1_overlap is None or mask2_overlap is None:
                        continue

                    intersection = torch.logical_and(mask1_overlap, mask2_overlap).sum()
                    union = torch.logical_or(mask1_overlap, mask2_overlap).sum()
                    iou = intersection / (union + 1e-6)

                    cur_area = float(current_pixels) if not torch.is_tensor(current_pixels) else float(current_pixels.item())
                    oth_area = float(other_inst_pixels) if not torch.is_tensor(other_inst_pixels) else float(other_inst_pixels.item())
                    min_area = min(cur_area, oth_area)
                    iomin = intersection / (min_area + 1e-6)

                    should_merge = (iou > iou_threshold) or (iomin > containment_threshold)

                    if should_merge:
                        new_total_count = current_score_count + other_inst_score_count
                        current_score = (
                            current_score * current_score_count
                            + other_inst_score * other_inst_score_count
                        ) / new_total_count
                        current_score_count = new_total_count

                        current_instance_position = [
                            min(current_instance_position[0], other_inst_instance_position[0]),
                            min(current_instance_position[1], other_inst_instance_position[1]),
                            max(current_instance_position[2], other_inst_instance_position[2]),
                            max(current_instance_position[3], other_inst_instance_position[3]),
                        ]

                        current_merge_mask = torch.logical_or(
                            current_whole_mask, other_inst_whole_mask
                        ).to(torch.uint8)
                        current_mask = current_merge_mask[
                            current_instance_position[0]:current_instance_position[2],
                            current_instance_position[1]:current_instance_position[3]
                        ]
                        current_pixels = current_mask.sum()

                        current_bbox = torch.tensor([
                            min(current_bbox[0], other_inst_bbox[0]),
                            min(current_bbox[1], other_inst_bbox[1]),
                            max(current_bbox[2], other_inst_bbox[2]),
                            max(current_bbox[3], other_inst_bbox[3]),
                        ], device=device)
                        merged_ids.append(j)

                    new_inst = {
                        'mask': current_mask,
                        'class': current_class,
                        'pixel_count': current_pixels,
                        'position': current_position,
                        'instance_position': current_instance_position,
                        'bbox': current_bbox,
                        'score': current_score,
                        'score_count': current_score_count,
                    }
                    current_all_instances_y1[k] = new_inst

            for n in sorted(merged_ids, reverse=True):
                other_all_instances_y1.pop(n)
            current_all_instances_y1.extend(other_all_instances_y1)

            current_position = [
                min(current_position[0], other_inst_position[0]),
                min(current_position[1], other_inst_position[1]),
                max(current_position[2], other_inst_position[2]),
                max(current_position[3], other_inst_position[3]),
            ]

            for current_instance in current_all_instances_y1:
                current_instance['position'] = current_position

        all_instances.extend(current_all_instances_y1)

    all_instances.extend(kept_as_is)

    if device == 'cuda':
        for i in all_instances:
            whole_mask = torch.zeros(whole_image_size, device=device, dtype=torch.uint8)
            mask = i['mask']
            instance_position = i['instance_position']
            whole_mask[
                instance_position[0]:instance_position[2],
                instance_position[1]:instance_position[3]
            ] = mask
            i['mask'] = whole_mask

        from detectron2.structures import Boxes
        pred_mask = torch.stack([i['mask'] for i in all_instances], dim=0)
        pred_classes = torch.stack([i['class'] for i in all_instances], dim=0)
        pred_bbox = Boxes(torch.stack([i['bbox'] for i in all_instances], dim=0))
        score = torch.stack([i['score'] for i in all_instances], dim=0)
    else:
        for i in all_instances:
            whole_mask = torch.zeros(whole_image_size, device=device, dtype=torch.uint8)
            mask = i['mask']
            instance_position = i['instance_position']
            whole_mask[
                instance_position[0]:instance_position[2],
                instance_position[1]:instance_position[3]
            ] = mask
            i['mask'] = whole_mask

        from detectron2.structures import Boxes
        pred_mask = torch.stack([i['mask'] for i in all_instances], dim=0).to('cpu')
        pred_classes = torch.stack([i['class'] for i in all_instances], dim=0).to('cpu')
        pred_bbox = Boxes(torch.stack([i['bbox'] for i in all_instances], dim=0).to('cpu'))
        score = torch.stack([i['score'] for i in all_instances], dim=0).to('cpu')

    return pred_mask, pred_classes, pred_bbox, score 
  
def split_image(image, tile_size=256, overlap=128):
   
    h, w = image.shape[:2]
    tiles = []
    
    y_steps = max((h - tile_size) // (tile_size - overlap), 0)
    x_steps = max((w - tile_size) // (tile_size - overlap), 0)
    if h >= tile_size:
        y_last = (h - tile_size) % (tile_size - overlap)
    else:
        y_last = 0
    if w >= tile_size:
        x_last = (w - tile_size) % (tile_size - overlap)
    else:
        x_last = 0
    
    for i in range(y_steps + 1):
        for j in range(x_steps + 1):
            y1 = i * (tile_size - overlap)
            y2 = min(y1 + tile_size, h)
            x1 = j * (tile_size - overlap)
            x2 = min(x1 + tile_size, w)
            tile = image[y1:y2, x1:x2]
            tiles.append(((y1, y2, x1, x2), tile))
        if x_last != 0:
            y1 = i * (tile_size - overlap)
            y2 = min(y1 + tile_size, h)
            x1 = w - tile_size
            x2 = min(x1 + tile_size, w)
            tile = image[y1:y2, x1:x2]
            tiles.append(((y1, y2, x1, x2), tile))
    if y_last != 0:
        for j in range(x_steps + 1):
            y1 = h - tile_size
            y2 = min(y1 + tile_size, h)
            x1 = j * (tile_size - overlap)
            x2 = min(x1 + tile_size, w)
            tile = image[y1:y2, x1:x2]
            tiles.append(((y1, y2, x1, x2), tile))
        if x_last != 0:
            y1 = h - tile_size
            y2 = min(y1 + tile_size, h)
            x1 = w - tile_size
            x2 = min(x1 + tile_size, w)
            tile = image[y1:y2, x1:x2]
            tiles.append(((y1, y2, x1, x2), tile))

    return tiles

def vis_instance_outlines(annotations, fillcolor=[22,229,22], linewidth=3, save_path=None, verbose=False):
    first_seg = annotations['annotations'][0]['segmentation']
    if isinstance(first_seg, dict):
        image_shape = tuple(first_seg['size'][::-1])  # COCO: [height, width] → (width, height) for np
    else:
        raise ValueError("Unsupported segmentation format")

    colored = np.zeros((*image_shape[::-1], 3), dtype=np.uint8)  # (H, W, 3)

    for i, ann in enumerate(annotations['annotations']):
        seg = ann['segmentation']
        if isinstance(seg, dict):
            if isinstance(seg["counts"], list):
                rle_obj = maskUtils.frPyObjects(seg, seg["size"][0], seg["size"][1])
            else:
                rle_obj = seg
            mask = maskUtils.decode(rle_obj)
            colored[mask == 1] = fillcolor

    for i, ann in enumerate(annotations['annotations']):
        seg = ann['segmentation']
        if isinstance(seg, dict):
            if isinstance(seg["counts"], list):
                rle_obj = maskUtils.frPyObjects(seg, seg["size"][0], seg["size"][1])
            else:
                rle_obj = seg
            mask = maskUtils.decode(rle_obj)
            contours = measure.find_contours(mask, level=0.5)
            for contour in contours:
                contour = contour[:, ::-1].astype(np.int32)  # (y,x) → (x,y)
                cv2.polylines(colored, [contour], isClosed=True, color=[255, 255, 255], thickness=1)
            
            dilated = mask.astype(bool)
            for kk in range(1, linewidth):
                dilated = binary_dilation(dilated)
                contours = measure.find_contours(dilated, level=0.5)
                for contour in contours:
                    contour = contour[:, ::-1].astype(np.int32)  # (y,x) → (x,y)
                    cv2.polylines(colored, [contour], isClosed=True, color=[255, 255, 255], thickness=1)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

    if verbose:
        plt.figure(figsize=(12, 8))
        plt.imshow(colored)
        plt.title("Colorized Instance Segmentation")
        plt.axis('off')
        plt.show(block=True)
        
def load_ann_from_img_json(image_path, json_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_name = os.path.basename(image_path)

    with open(json_path) as f:
        annotations = json.load(f)
        if 'annotations' not in annotations:
            raise ValueError("annotations is not in json")
    
    if 'images' in annotations and len(annotations['images'])>1:
        target_image_id = None
        for img in annotations['images']:
            if img.get('file_name') == image_name:
                target_image_id = img['id']
                break
        if target_image_id is not None:
            filtered_annos = [anno for anno in annotations.get('annotations', []) if anno.get('image_id') == target_image_id]
            annotations['annotations'] = filtered_annos

    return np.array(image), annotations



def cal_feret_diameter(mask):
    """cal maximum distance"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    main_contour = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(main_contour, returnPoints=True)
    hull = hull.squeeze(1) 

    max_distance = 0
    n = len(hull)

    # Check all possible vertex pairs
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(hull[i] - hull[j])
            if distance > max_distance:
                max_distance = distance

    return max_distance


def cal_branch_points(mask):
    """Count branch points (via skeletonization)."""
    # Skeletonize the mask
    skeleton = skeletonize(mask)

    # 3x3 kernel for branch-point detection
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Count neighbors at each skeleton pixel
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

    # Branch point: skeleton pixel with 3 or more neighbors
    branch_points = (skeleton > 0) & (neighbor_count >= 3)

    # Label connected components to count branch points
    labeled, num_objects = label(branch_points)

    return num_objects

def cal_mito_metrics(mask):
    """
    Compute geometric and morphological features for each instance in a COCO file,
    including mean grayscale.

    Args:
        coco_file: Path to a COCO-format JSON file.
        image_dir: Directory containing the original images.

    Returns:
        list: List of dicts with per-instance attributes.
    """
    mask = mask > 0

    h, w = mask.shape

    area = mask.sum()

    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        perimeter = 0
    else:
        # Use the largest contour by area
        main_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(main_contour, closed=True)

 
    rect = cv2.minAreaRect(main_contour) if contours else ((0, 0), (0, 0), 0)
    (_, (w, h), _) = rect
    max_side, min_side = max(w, h), min(w, h)
    aspect_ratio = max_side / min_side if min_side > 0 else 0

    # 6. Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # 7. Length (maximum Feret diameter)
    length = cal_feret_diameter(mask) if contours else 0

    # 8. Branch count (via skeletonization)
    branch_points = cal_branch_points(mask) if area > 10 else 0

    results = {
        'height': h,
        'width': w,
        'length': length,
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'branch_points': branch_points}

    return results

def ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = maskUtils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return maskUtils.decode(rle)  # 0/1 mask

def extract_mito_anndata(
        image_root: str,
        coco_file: str,
        chan_info: dict={'coco':'_mito.png','mito':'_mito.tif','func':'_tmrm.tif', 'type':'tmrm'},
        exp_parser: str='parser_exp_0715cccp',
        filter_mito: bool=True
    ):
    parser_func = getattr(import_module("panomito.parser_func"), exp_parser)

    coco_json_path = Path(os.path.join(image_root, coco_file))                 
    with coco_json_path.open("r") as f:
        coco = json.load(f)

    if filter_mito:
        coco['annotations'] = [ann for ann in coco['annotations'] if ann['category_id'] < 5]

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    image_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        image_annotations[ann["image_id"]].append(ann)

    all_metrics = []
    all_crops = []
    all_int_crops = []
    if  'func' in chan_info:
        all_func_crops = []

    for image_id, file_name in tqdm(id_to_filename.items(), desc="Extract mitos"):
        # file_name suffix must match the coco channel
        # if '_mito.png' not in file_name:
        #     continue
        # proc mito instance
        file_name = file_name.replace(chan_info['coco'],chan_info['mito'])   
        # print(f"{file_name}")
        condition = parser_func(file_name)
        img_path = os.path.join(image_root, file_name)        
        img = tifffile.imread(img_path)
        if 'func' in chan_info:            
            func_filename = file_name.replace(chan_info['mito'],chan_info['func'])
            img_func = tifffile.imread(os.path.join(image_root, func_filename))
            # save_ratio_img(img, img_func,
            #                func_filename.replace(chan_info['func'], '_ratio.png'),
            #                image_root,
            #                scale = 1/25.5)
        
        w, h = img.shape
        anns = image_annotations.get(image_id, [])
        if not anns:
            continue

        for ann in anns:
            mask01 = ann_to_mask(ann, h, w)
            if mask01.sum() == 0:
                continue

            ys, xs = np.where(mask01 > 0)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            # mask
            crop = mask01[y1:y2+1, x1:x2+1]
            crop = resize_mito_images(crop, int_ratio=255.)
            crop = (crop > 0) * 255
            all_crops.append(crop.astype(np.uint8))

            metrics = cal_mito_metrics(crop)

            metrics['category'] = ann['category_id']

            metrics['id'] = ann['id']
            metrics['filename'] = file_name
            metrics.update(condition)

            tmp = mask01*img
            crop_int = tmp[y1:y2+1, x1:x2+1]
            crop_int = resize_mito_images(crop_int)
            all_int_crops.append(crop_int.astype(np.uint8))
    
            if 'func' in chan_info:  
                tmp = mask01*img_func
                crop_func = tmp[y1:y2+1, x1:x2+1]
                crop_func = resize_mito_images(crop_func)
                all_func_crops.append(crop_func.astype(np.uint8))
                ratio = (crop_func.sum()+1e-6)/(crop_int.sum()+1e-6)
                metrics['ratio'] = ratio

            all_metrics.append(metrics)


    obs_df = pd.DataFrame(all_metrics)

    df_file = os.path.join(image_root, '_metrics.csv')
    obs_df.set_index("id", inplace=True)
    obs_df.to_csv(df_file, index=True, float_format='%.8f')
    
    adata = anndata.AnnData(obs=obs_df)
    adata.uns['original_crops'] = np.stack(all_crops, axis=0)
    adata.uns['int_crops'] = np.stack(all_int_crops, axis=0)
    if 'func' in chan_info: 
        adata.uns[chan_info['type']+'_crops'] = np.stack(all_func_crops, axis=0)
    adata_file = os.path.join(image_root, '_adata.h5ad')
    adata.write_h5ad(adata_file)

    return adata


def resize_mito_images(
    img,
    scale: float = 3.0,
    int_ratio: float = 1/25.5,
    target: int = 256,
    fill_value: int = 0,
):
    img = img * int_ratio
    resized = ndimage.zoom(img, scale, order=1)

    h, w = resized.shape[:2]
    if h > target or w > target:
        if h > w:
            resized = ndimage.zoom(img, target/img.shape[0], order=1)
        else:
            resized = ndimage.zoom(img, target/img.shape[1], order=1)
        # resized = resized.astype(np.uint8)
        h, w = resized.shape[:2]

    pad_top = (target - h) // 2 if h < target else 0
    pad_bottom = target - h - pad_top if h < target else 0
    pad_left = (target - w) // 2 if w < target else 0
    pad_right = target - w - pad_left if w < target else 0

    if pad_top or pad_bottom or pad_left or pad_right:
        resized = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(fill_value, fill_value)
        )
    
    return resized