import os, json
import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from glob import glob
import shutil
import tifffile as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.evaluation.coco_evaluation import COCOevalMaxDets, _evaluate_predictions_on_coco
import pandas as pd


class myCOCOeval(COCOeval):
    def __init__(self, cocoGt, cocoDt, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)
        
    def summarize(self):
        
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                
                s = self.eval['precision']
                
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((18,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[14] = _summarize(1, iouThr=.6, maxDets=self.params.maxDets[2])
            stats[15] = _summarize(1, iouThr=.7, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[16] = _summarize(1, iouThr=.8, maxDets=self.params.maxDets[2])
            stats[17] = _summarize(1, iouThr=.9, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

def evaluate_coco_results(gt_json_path, pred_json_path, use_F1score = False, confidence_score=0, iou_threshold=0.5, csv_path='',use_box_or_seg='seg'):
  
    coco_gt = COCO(gt_json_path)
    
    
    with open(pred_json_path,'r') as f:
        pred_data = json.load(f)
    
    if 'score' not in pred_data['annotations'][0]:
        confidence_score = 0
        output_json_path = pred_json_path.replace(".json", "_add_score.json")
        pred_data = assign_pseudo_confidence_to_cellpose(gt_json_path, pred_json_path, output_json_path)        
            
    for ann in pred_data['annotations']:
        ann["category_id"] = 1
    pred_annotations = pred_data['annotations']
    
    if use_F1score == True:
        score_threshold = confidence_score
        iou_threshold = iou_threshold
        F1score = calculate_f1_manual_for_per_image(gt_json_path, pred_data, iou_threshold=iou_threshold, score_threshold=score_threshold, use_box_or_seg=use_box_or_seg,output_csv=csv_path)
        print(f"score_threshold = {score_threshold}, iou_threshold = {iou_threshold} \n  {F1score}")
    
    coco_pred = coco_gt.loadRes(pred_annotations)
    
   
    for task in ["segm"]:
        coco_eval = myCOCOeval(coco_gt, coco_pred, task)  
        coco_eval.params.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]) 
        coco_eval.params.maxDets = [1, 10, 500]
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def calculate_f1_manual_for_per_image(gt_json_path, pred_json_path, iou_threshold=0.5, score_threshold=0.0, 
                        use_box_or_seg='seg', output_csv=None):
    """
    Compute per-image Precision/Recall/F1 by matching predictions to GT with an IoU threshold.

    - Read GT JSON and group annotations by image_id.
    - Use pred_json_path as a prediction dict (expects pred_json_path['annotations']).
    - Filter predictions by score_threshold (if "score" exists).
    - For each image:
        * compute IoU matrix (bbox or segm via use_box_or_seg)
        * greedily match pairs with IoU >= iou_threshold (highest IoU first)
        * derive TP/FP/FN, then precision/recall/F1 (and accuracy)
    - Optionally save per-image metrics to CSV.

    Returns:
        A dict with overall precision/recall/F1/TP/FP/FN and a list of per-image metrics.
    """   
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    pred_data = pred_json_path
    
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    image_id_to_name = {img['id']: img['file_name'] for img in gt_data['images']}
    
    for ann in gt_data['annotations']:
        image_id = ann['image_id']
        gt_by_image[image_id].append(ann)
    
    for ann in pred_data['annotations']:
        if 'score' in ann and ann['score'] < score_threshold:
            continue
        image_id = ann['image_id']
        pred_by_image[image_id].append(ann)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    image_metrics = []
    
    sorted_image_ids = sorted(gt_by_image.keys())
    for image_id in tqdm(sorted_image_ids, desc='计算F1score'):
        gt_anns = gt_by_image.get(image_id, [])
        pred_anns = pred_by_image.get(image_id, [])
        
        if not gt_anns and not pred_anns:
            continue
        
        n_gt = len(gt_anns)
        n_pred = len(pred_anns)
        
        tp = 0
        fp = 0
        fn = 0
        
        if n_pred == 0:
            fn = n_gt
            total_fn += fn
        elif n_gt == 0:
            fp = n_pred
            total_fp += fp
        else:
            iou_matrix = np.zeros((n_gt, n_pred))
            for i, gt_ann in enumerate(gt_anns):
                for j, pred_ann in enumerate(pred_anns):
                    iou = calculate_iou(gt_ann, pred_ann, box_or_seg=use_box_or_seg)
                    iou_matrix[i, j] = iou
            
            matched_gt = set()
            matched_pred = set()
            
            matches = []
            for i in range(n_gt):
                for j in range(n_pred):
                    if iou_matrix[i, j] >= iou_threshold:
                        matches.append((iou_matrix[i, j], i, j))
            
            matches.sort(reverse=True, key=lambda x: x[0])
            
            for _, i, j in matches:
                if i not in matched_gt and j not in matched_pred:
                    matched_gt.add(i)
                    matched_pred.add(j)
            
            tp = len(matched_gt)
            fp = n_pred - len(matched_pred)
            fn = n_gt - len(matched_gt)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision_img = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_img = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy_img = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        f1_img = 2 * (precision_img * recall_img) / (precision_img + recall_img) if (precision_img + recall_img) > 0 else 0
        
        image_metrics.append({
            'file_name': image_id_to_name.get(image_id, f'image_{image_id}'),
            'image_id': image_id,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_count': n_gt,
            'pred_count': n_pred,
            'precision': precision_img,
            'recall': recall_img,
            'f1_score': f1_img,
            'accuracy': accuracy_img
        })
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if output_csv and image_metrics:
        df = pd.DataFrame(image_metrics)
        df.to_csv(output_csv, index=False)
        print(f"Metrics have been saved to: {output_csv}")

    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'iou_threshold': iou_threshold,
        'accuracy': accuracy,
        'image_metrics': image_metrics
    }


def calculate_f1_manual(gt_json_path, pred_json_path, iou_threshold=0.5, score_threshold=0.0, use_box_or_seg = 'seg'):

    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    pred_data = pred_json_path
    
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for ann in gt_data['annotations']:
        image_id = ann['image_id']
        gt_by_image[image_id].append(ann)
    
    for ann in pred_data['annotations']:
        if 'score' in ann and ann['score'] < score_threshold:
            continue
        image_id = ann['image_id']
        pred_by_image[image_id].append(ann)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for image_id in tqdm(gt_by_image.keys(), desc='calculate F1score'):
        gt_anns = gt_by_image.get(image_id, [])
        pred_anns = pred_by_image.get(image_id, [])
        
        if not gt_anns and not pred_anns:
            continue
        
        n_gt = len(gt_anns)
        n_pred = len(pred_anns)
        
        if n_pred == 0:
            total_fn += n_gt
            continue
        
        if n_gt == 0:
            total_fp += n_pred
            continue
        
        iou_matrix = np.zeros((n_gt, n_pred))
        for i, gt_ann in enumerate(gt_anns):
            for j, pred_ann in enumerate(pred_anns):
                iou = calculate_iou(gt_ann, pred_ann, box_or_seg = use_box_or_seg)
                iou_matrix[i, j] = iou
        
        matched_gt = set()
        matched_pred = set()
        
        matches = []
        for i in range(n_gt):
            for j in range(n_pred):
                if iou_matrix[i, j] >= iou_threshold:
                    matches.append((iou_matrix[i, j], i, j))
        
        matches.sort(reverse=True, key=lambda x: x[0])
        
        for _, i, j in matches:
            if i not in matched_gt and j not in matched_pred:
                matched_gt.add(i)
                matched_pred.add(j)
        
        tp = len(matched_gt)
        fp = n_pred - len(matched_pred)
        fn = n_gt - len(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'iou_threshold': iou_threshold,
        'accuracy':accuracy
    }

def calculate_iou(ann1, ann2, box_or_seg):
    
    if box_or_seg == 'seg' and 'segmentation' in ann1 and 'segmentation' in ann2:
  
        mask1 = maskUtils.decode(ann1['segmentation'])
        mask2 = maskUtils.decode(ann2['segmentation'])
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union > 0 else 0.0
    
    elif box_or_seg == 'box' and 'bbox' in ann1 and 'bbox' in ann2:
       
        x1, y1, w1, h1 = ann1['bbox']
        x2, y2, w2, h2 = ann2['bbox']
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
    else:
        raise Exception('there are no box or seg in ann')
    
    return iou



def assign_pseudo_confidence_to_cellpose(gt_json_path, pred_json_path, output_json_path=None,
                                         min_confidence=0.0, max_confidence=0.99):

    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    
    for i, ann in enumerate(pred_data['annotations']):
        pred_by_image[ann['image_id']].append(ann)
    
    all_predictions = []
    
    for image_id in tqdm(pred_by_image.keys(), desc='assign score'):
        gt_anns = gt_by_image.get(image_id, [])
        pred_anns = pred_by_image.get(image_id, [])
        
        if not pred_anns:
            continue
        
        if not gt_anns:
            for pred in pred_anns:
                pred['score'] = min_confidence
                all_predictions.append(pred)
            continue
        
        iou_matrix = compute_iou_matrix(pred_anns, gt_anns)
        
        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
        
        pred_to_best_iou = {}
        for p_idx, g_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[p_idx, g_idx]
            pred_to_best_iou[p_idx] = iou
        
        for p_idx, pred in enumerate(pred_anns):
            best_iou = pred_to_best_iou.get(p_idx, 0.0)
            
            confidence = best_iou
        
        
            confidence = max(min_confidence, min(confidence, max_confidence))
            
      
            pred['score'] = float(confidence)
            all_predictions.append(pred)
    

    output_data = {
        'info': {
            'description': 'CellposePredicted results (with pseudo confidence scores added)',
            'min_confidence': min_confidence,
            'max_confidence': max_confidence
        },
        'images': gt_data.get('images', []),
        'categories': gt_data.get('categories', []),
        'annotations': all_predictions
    }
    

    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✅ Saved to: {output_json_path}")
        print(f"   Number of predictions: {len(all_predictions)}")

        

    return output_data

def compute_iou_matrix(pred_anns, gt_anns):

    n_pred = len(pred_anns)
    n_gt = len(gt_anns)
    
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt))
    
    iou_matrix = np.zeros((n_pred, n_gt))
    
    for i, pred_ann in enumerate(pred_anns):
        for j, gt_ann in enumerate(gt_anns):
            iou = compute_mask_iou(pred_ann, gt_ann)
            iou_matrix[i, j] = iou
    
    return iou_matrix

def compute_bbox_iou(ann1, ann2):

    if 'bbox' not in ann1 or 'bbox' not in ann2:
        return 0.0
    
    x1, y1, w1, h1 = ann1['bbox']
    x2, y2, w2, h2 = ann2['bbox']
    
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area > 0:
        return inter_area / union_area
    else:
        return 0.0

def compute_mask_iou(ann1, ann2):

    if 'segmentation' in ann1 and 'segmentation' in ann2:
        mask1  = maskUtils.decode(ann1['segmentation'])
        mask2  = maskUtils.decode(ann2['segmentation'])
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union > 0 else 0.0
    return iou


def merge_coco_cultured_primary(
    json1_path,
    json2_path,
    output_path = "merged_coco_cultured_primary.json"
):

    import json
    import copy
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(json1_path, 'r') as f:
        coco1 = json.load(f)
    
    with open(json2_path, 'r') as f:
        coco2 = json.load(f)
    
    print(f"COCO1: {len(coco1.get('images', []))} images")
    print(f"COCO2: {len(coco2.get('images', []))} images")
    

    merged = {
        'info': coco1.get('info', {}),
        'licenses': coco1.get('licenses', []),
        'categories': coco1.get('categories', []),
        'images': [],
        'annotations': []
    }
    

    image_id_map = {} 
    current_image_id = 1
    

    for img in coco1.get('images', []):
        new_img = copy.deepcopy(img)
        old_id = new_img['id']
        new_img['id'] = current_image_id
        image_id_map[(1, old_id)] = current_image_id
        merged['images'].append(new_img)
        current_image_id += 1
    

    for img in coco2.get('images', []):
        new_img = copy.deepcopy(img)
        old_id = new_img['id']
        new_img['id'] = current_image_id
        image_id_map[(2, old_id)] = current_image_id
        merged['images'].append(new_img)
        current_image_id += 1
    

    current_ann_id = 1
    

    for ann in coco1.get('annotations', []):
        new_ann = copy.deepcopy(ann)
        new_ann['id'] = current_ann_id
        new_ann['image_id'] = image_id_map[(1, ann['image_id'])]
        merged['annotations'].append(new_ann)
        current_ann_id += 1
    

    for ann in coco2.get('annotations', []):
        new_ann = copy.deepcopy(ann)
        new_ann['id'] = current_ann_id
        new_ann['image_id'] = image_id_map[(2, ann['image_id'])]
        merged['annotations'].append(new_ann)
        current_ann_id += 1
    

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"✅ Merge completed!")
    print(f"   Images: {len(merged['images'])}")
    print(f"   Annotations: {len(merged['annotations'])}")
    print(f"   Categories: {len(merged['categories'])}")
    print(f"   Saved to: {output_path}")

    
    return merged



def evaluate_coco_results_per_image(gt_json_path, pred_json_path, use_F1score=False, confidence_score=0, 
                          iou_threshold=0.5, F1_per_image_csv_path=None, use_box_or_seg='seg', AP_per_image_csv_path=None):
    """
    Evaluate COCO-format predictions per image.
    - Load GT with COCO API.
    - Load prediction JSON; if missing "score", assign pseudo confidence scores.
    - Optionally compute per-image F1 (TP/FP/FN) using IoU and score thresholds.
    - Run COCOeval for segm and export per-image AP metrics to CSV (optional).
    """
    coco_gt = COCO(gt_json_path)
    
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    
    if 'score' not in pred_data['annotations'][0]:
        confidence_score = 0
        output_json_path = pred_json_path.replace(".json", "_add_score.json")
        pred_data = assign_pseudo_confidence_to_cellpose(gt_json_path, pred_json_path, output_json_path)
    
    for ann in pred_data['annotations']:
        ann["category_id"] = 1
    pred_annotations = pred_data['annotations']
    
    if use_F1score:
        score_threshold = confidence_score
        iou_threshold = iou_threshold
        F1score = calculate_f1_manual_for_per_image(gt_json_path, pred_data, 
                                                   iou_threshold=iou_threshold, 
                                                   score_threshold=score_threshold, 
                                                   use_box_or_seg=use_box_or_seg,
                                                   output_csv=F1_per_image_csv_path)
        print(f"score_threshold = {score_threshold}, iou_threshold = {iou_threshold} \n  {F1score}")
    
    coco_pred = coco_gt.loadRes(pred_annotations)
    

    all_image_data = []
    

    for task in ["segm"]:
        coco_eval = myCOCOeval(coco_gt, coco_pred, task)
        coco_eval.params.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        coco_eval.params.maxDets = [1, 10, 500]
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        

        image_ap_metrics = extract_ap_from_global_precision(coco_eval, coco_gt, task)
        all_image_data.append((task, image_ap_metrics))
        
        coco_eval.summarize()
    

    if AP_per_image_csv_path and all_image_data:
        save_combined_metrics_to_csv(all_image_data, AP_per_image_csv_path, coco_gt)
        print(f"Per-image AP metrics have been saved to: {AP_per_image_csv_path}")

    
    return all_image_data


def extract_ap_from_global_precision(coco_eval, coco_gt, task):

    image_metrics = []
    
    iou_thresholds = coco_eval.params.iouThrs
    
    for img_id in coco_eval.params.imgIds:
        try:
            img_info = coco_gt.loadImgs(int(img_id))[0]
            metrics = {
                'file_name': img_info.get('file_name', f'image_{img_id}'),
                'image_id': int(img_id)
            }
            
            eval_data_list = [eval_data for eval_data in coco_eval.evalImgs 
                            if eval_data is not None and eval_data['image_id'] == img_id]
            
            if not eval_data_list:
                for i, iou_thr in enumerate(iou_thresholds):
                    metrics[f'{task}_AP@{iou_thr:.2f}'] = 0.0
                image_metrics.append(metrics)
                continue
            
            eval_data = eval_data_list[0]
            
            dt_matches = eval_data.get('dtMatches', np.array([]))
            dt_scores = eval_data.get('dtScores', [])
            gt_ids = eval_data.get('gtIds', [])
            
            if dt_matches.size == 0 or len(dt_scores) == 0:
                for i, iou_thr in enumerate(iou_thresholds):
                    metrics[f'{task}_AP@{iou_thr:.2f}'] = 0.0
                image_metrics.append(metrics)
                continue

            for i, iou_thr in enumerate(iou_thresholds):
                if i >= dt_matches.shape[0]:
                    metrics[f'{task}_AP@{iou_thr:.2f}'] = 0.0
                    continue
                

                matches = dt_matches[i]  
                
                if len(matches) == 0:
                    metrics[f'{task}_AP@{iou_thr:.2f}'] = 0.0
                    continue

                sorted_indices = np.argsort(dt_scores)[::-1]  
                sorted_matches = matches[sorted_indices]
                
                cum_tp = np.cumsum(sorted_matches > 0)
                cum_fp = np.cumsum(sorted_matches == 0)
                
                precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
                recalls = cum_tp / (len(gt_ids) + 1e-8)
                
                ap = compute_ap_11_point(precisions, recalls)
                metrics[f'{task}_AP@{iou_thr:.2f}'] = float(ap)
            
            image_metrics.append(metrics)
            
        except Exception as e:
            print(f"❌ Error while processing image {img_id}: {e}")

            continue
    
    return image_metrics

def compute_ap_11_point(precisions, recalls):

    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0
    
    recall_levels = np.linspace(0, 1, 11)
    
    ap = 0.0
    for recall_level in recall_levels:
        mask = recalls >= recall_level
        if np.any(mask):
            max_precision = np.max(precisions[mask])
        else:
            max_precision = 0.0
        ap += max_precision
    
    return ap / 11.0

def save_combined_metrics_to_csv(all_image_data, output_path, coco_gt):
    if not all_image_data:
        return
    
    img_ids = all_image_data[0][1][0].keys() if all_image_data[0][1] else []
    
    all_images = []
    for task, image_metrics in all_image_data:
        for metrics in image_metrics:
            img_id = metrics['image_id']
            
            existing = next((img for img in all_images if img['image_id'] == img_id), None)
            if not existing:
                img_info = coco_gt.loadImgs(int(img_id))[0]
                existing = {
                    'file_name': img_info.get('file_name', f'image_{img_id}'),
                    'image_id': img_id
                }
                all_images.append(existing)
            
            for key, value in metrics.items():
                if key not in ['file_name', 'image_id', 'task']:
                    existing[f'{task}_{key}'] = value
    

    df = pd.DataFrame(all_images)
    

    base_cols = ['file_name', 'image_id']
    other_cols = [col for col in df.columns if col not in base_cols]
    df = df[base_cols + sorted(other_cols)]
    
    df.to_csv(output_path, index=False, float_format='%.4f')