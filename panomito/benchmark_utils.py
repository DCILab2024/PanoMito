from glob import glob
import os, json
import numpy as np
import shutil
import tifffile
from pycocotools import mask as maskUtils
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def align_coco_ids(gt_json_path, pred_json_path, output_json_path):

    with open(gt_json_path , 'r') as f:
        gt_data = json.load(f)
    

    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)   

    filename_to_gt_id = {img['file_name']: img['id'] for img in gt_data['images']}
    
    pred_filename_to_id = {}
    if 'images' in pred_data:
        pred_filename_to_id = {img['file_name']: img['id'] for img in pred_data['images']}
    
    id_mapping = {}
    for filename, gt_id in filename_to_gt_id.items():
        if filename in pred_filename_to_id:
            pred_id = pred_filename_to_id[filename]
            id_mapping[pred_id] = gt_id

    adjusted_annotations = []
    pred_annotations = pred_data['annotations']
   
    for ann in pred_annotations:
        if ann['image_id'] in id_mapping:
            adjusted_ann = ann.copy()
            adjusted_ann['image_id'] = id_mapping[ann['image_id']]
            adjusted_annotations.append(adjusted_ann)
        else:
            print(f"Warning: image_id {ann['image_id']} in the predictions has no corresponding ground-truth annotation and has been skipped")

    
    output_data = {
        "info": pred_data.get('info', {}),
        "licenses": pred_data.get('licenses', []),
        "categories": gt_data['categories'],
        "images": gt_data['images'],  
        "annotations": adjusted_annotations,
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    

    print(f"The adjusted prediction results have been saved to: {output_json_path}")
    print(f"Number of original predicted annotations: {len(pred_annotations)}")
    print(f"Number of valid annotations after adjustment: {len(adjusted_annotations)}")
    print(f"Number of successfully matched images: {len(id_mapping)}/{len(filename_to_gt_id)}")


def npy_to_coco(npy_dir, output_json):

    coco_data = {
        "info": None, 
        "licenses": [], 
        "categories": [{"id": 1, "name": "Globule", "supercategory": ""}, 
                       {"id": 2, "name": "Tubule", "supercategory": ""}, 
                       {"id": 3, "name": "Loop", "supercategory": ""}, 
                       {"id": 4, "name": "Branch", "supercategory": ""}],
        "images": [],
        "annotations": [],       
        }
    
    
    image_id = 1
    annotation_id = 1
    

    tif_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(('.npy'))])
    
    for filename in tqdm(tif_files, desc="Processing masks"):
        file_path = os.path.join(npy_dir, filename)
        
       
        dat = np.load(file_path, allow_pickle=True).item()
        mask = dat["masks"]
        height, width = mask.shape
        
       
        image_entry = {
            "id": image_id,
            "file_name": filename.replace('_seg.npy', '.png'),
            "height": height,
            "width": width,
            "license": 1,
        }
        coco_data["images"].append(image_entry)
        
        
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]
        
       
        for instance_id in instance_ids:
        
            instance_mask = (mask == instance_id).astype(np.uint8)
            
    
            rle = maskUtils.encode(np.asfortranarray(instance_mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            area = maskUtils.area(rle).item()
            bbox = maskUtils.toBbox(rle).tolist()
            
            
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation_entry)
            annotation_id += 1
        
        image_id += 1
    
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    

    print(f"COCO-format JSON file has been generated: {output_json}") 
    print(f"Contains {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
