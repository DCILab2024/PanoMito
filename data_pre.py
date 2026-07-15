import numpy as np
import os, json
from PIL import Image
import tifffile
from pycocotools import mask as maskUtils
import cv2
import scipy as sp
from skimage import measure
import json
import random
import os
import sys
sys.path.append("/home/kzlab/panomito")

def convert_rle_to_compressed(annotation_path, output_path=None):
    if isinstance(annotation_path, dict):
        data = annotation_path
    else:
        with open(annotation_path, 'r') as f:
            data = json.load(f)

    for ann in data['annotations']:
        # Skip if counts is already a string
        if isinstance(ann['segmentation']['counts'], str):
            continue
        # Get integer array counts and image size
        counts = ann['segmentation']['counts']
        size = ann['segmentation']['size']
        # Create RLE object from integer array format
        rle_obj = maskUtils.frPyObjects([{'counts': counts, 'size': size}], size[0], size[1])
        # Convert to compressed string
        compressed = maskUtils.encode(np.asfortranarray(maskUtils.decode(rle_obj)))[0]
        # Update annotation
        ann['segmentation'] = {
            'counts': compressed['counts'].decode('utf-8'),
            'size': compressed['size'],
        }
        ann['iscrowd'] = 0

    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    return data

def split_coco_dataset_traintest(coco_path, output_dir, train_ratio=0.7):
    """Split a COCO-format dataset into train and test sets.

    Args:
        coco_path: Path to the COCO JSON file.
        output_dir: Output directory.
        train_ratio: Fraction of images assigned to the training set.
        val_ratio: Validation set ratio (documented for reference).
    """
    # Validate ratio
    if not (0 < train_ratio < 1):
        raise ValueError("Invalid ratio parameter")

    # Load original data
    with open(coco_path) as f:
        data = json.load(f)

    # Remove categories and annotations with category id > 4
    data['annotations'] = [ann for ann in data['annotations'] if ann['category_id'] <= 4]
    data['categories'] = [cat for cat in data['categories'] if cat['id'] <= 4]

    # Group annotations by image
    img_anns = {}
    for ann in data['annotations']:
        img_anns.setdefault(ann['image_id'], []).append(ann)

    # Shuffle and split image IDs
    img_ids = list(img_anns.keys())
    random.shuffle(img_ids)

    # Compute split index
    train_end = int(len(img_ids) * train_ratio)

    # Collect image IDs for each split
    train_ids = img_ids[:train_end]
    test_ids = img_ids[train_end:]
    print("train:{},  test:{}".format(len(train_ids), len(test_ids)))

    # Build subset data for each split
    def _build_subset(ids):
        images = [img for img in data['images'] if img['id'] in ids]
        anns = [ann for img_id in ids for ann in img_anns[img_id]]
        coco = {
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'categories': data['categories'],
            'images': images,
            'annotations': anns
        }
        coco =  convert_rle_to_compressed(coco)

        return coco

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for name, ids in [('_train', train_ids), ('_test', test_ids)]:
        with open(f"{output_dir}/{name}.json", 'w') as f:
            json.dump(_build_subset(ids), f, indent=2)

def coco2npy_rle(cocodir, json_files = ['_train.json', '_test.json'], rm_cate=False):
    """Convert COCO-format annotation files to cellotype format.

    Args:
        coco_json_path (str): Path to the COCO JSON file.
        image_dir (str): Directory containing image files.
        output_npy_path (str): Output .npy file path.
    """
    for json_file in json_files:
        coco_json_path = os.path.join(cocodir, json_file)
        if rm_cate:
            output_npy_path = os.path.join(cocodir, json_file.replace('.json', '_nocate_rle.npy'))
        else:
            output_npy_path = os.path.join(cocodir, json_file.replace('.json', '_rle.npy'))
        # Read COCO JSON file
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        images = coco_data['images']
        cellotype_data = []
        for img_info in images:
            img_id = img_info['id']
            filename = img_info['file_name']
            height = img_info['height']
            width = img_info['width']
            pngfile = filename.replace('.tiff', '.png')
            # Build image path
            img_path = os.path.join(cocodir, pngfile)
            # Collect all annotations for this image_id
            ann_dict = []
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    if rm_cate:
                        ann['category_id'] = 0
                    else:
                        ann['category_id'] = ann['category_id'] - 1
                    ann['bbox_mode'] = 1
                    ann['iscrowd'] = 0
                    ann['attributes'] = {'occluded':False}
                    ann_dict.append(ann)
            
            # Build cellotype-format dict
            cellotype_item = {
                'file_name': img_path,
                'height': height,
                'width': width,
                'image_id': img_id,
                'annotations': ann_dict
            }
            cellotype_data.append(cellotype_item)
        
        # Save as .npy file
        np.save(output_npy_path, cellotype_data)
        print(f"Successfully converted and saved to {output_npy_path}")


if __name__ == "__main__":
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------------------------
    # Prepare training data for train.py
    #
    # train.py expects RLE-encoded .npy files under ./data/train/ and
    # ./data/test/. Raw PanoMitoAtlas releases are COCO JSON; this script
    # converts them to the format Detectron2 / PanoMito training consumes.
    #
    # Two workflows (see README §1 PanoMitoAtlas and § Train):
    #
    # A) Paper pre-split (default below)
    #    1. Copy train_test_subset/ from PanoMitoAtlas.zip into ./data/
    #       so that images and JSON live under ./data/train/ and ./data/test/.
    #    2. Run: python data_pre.py
    #    3. Outputs:
    #         ./data/train/_train_subdataset_gt_nocate_rle.npy
    #         ./data/test/_test_subdataset_gt_nocate_rle.npy
    #
    # B) Custom train/test split
    #    1. Start from a merged COCO JSON (e.g. all_2d_dataset/ in the zip).
    #    2. Uncomment and edit split_coco_dataset_traintest() below to write
    #       _train.json and _test.json, then move/copy images + JSON into
    #       ./data/train/ and ./data/test/ (e.g. _train.json, _test.json).
    #    3. Call coco2npy_rle() the same way as in A): update json_files in
    #       the calls below to match your JSON filenames, then run this script
    #       to generate *_nocate_rle.npy under ./data/train/ and ./data/test/.
    # -------------------------------------------------------------------------
    # split_coco_dataset_traintest(
    #     coco_path=os.path.join(_PROJECT_ROOT, "path/to/merged_annotations.json"),
    #     output_dir=os.path.join(_PROJECT_ROOT, "data"),
    #     train_ratio=0.7,
    # )

    # --- JSON -> NPY (default: paper pre-split filenames) --------------------
    # rm_cate=True sets all category_id to 0 (single-class "Mito" training).
    cocodir = os.path.join(_PROJECT_ROOT, "data", "train")
    coco2npy_rle(cocodir, json_files=["_train_subdataset_gt.json"], rm_cate=True)

    cocodir = os.path.join(_PROJECT_ROOT, "data", "test")
    coco2npy_rle(cocodir, json_files=["_test_subdataset_gt.json"], rm_cate=True)
