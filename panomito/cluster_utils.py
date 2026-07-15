from __future__ import annotations
import os, json
import tifffile
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import anndata
import pandas as pd
from importlib import import_module
from scipy import ndimage
from tqdm import tqdm
from pycocotools import mask as maskUtils
from skimage.morphology import skeletonize
from scipy.ndimage import label, binary_fill_holes
from typing import Dict, Any
from PIL import Image


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

def export_resized_masks_from_coco(
    coco_json_path,
    image_root,
    out_root,
    filter_mito: bool = True,
    postfix: str="none",
    verbose: bool = True,
) -> Dict[str, Any]:
 
    coco_json_path = Path(coco_json_path)
    image_root = Path(image_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with coco_json_path.open("r") as f:
        coco = json.load(f)

    if filter_mito:
        coco['annotations'] = [ann for ann in coco['annotations'] if ann['category_id'] < 5]

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    image_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        image_annotations[ann["image_id"]].append(ann)

    counter = 1
    for image_id, file_name in tqdm(id_to_filename.items(), desc="Exporting binary masks"):
        img_path = image_root / file_name
        if not img_path.exists():
            if verbose:
                print(f"⚠️ Image not found: {img_path}")

            continue

        with Image.open(img_path) as img:
            w, h = img.size

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

            crop_mask = mask01[y1:y2+1, x1:x2+1] * 255
            crop_mask = crop_mask.astype(np.uint8)

            out = resize_mito_images(crop_mask)

            filename = f"{ann['id']:06d}_{postfix}.png"
            out_path = out_root / filename
            Image.fromarray(out).save(out_path)

            counter += 1

    if verbose:
        print(f"✅ Exported {counter - 1} binary mask files to: {out_root}")


    return {
        "exported_count": counter - 1,
        "output_dir": out_root,
    }

def resize_mito_images(
    img,
    scale: float = 3.0,
    target: int = 255,
    fill_value: int = 0,
):
    def best_interp(src_h, src_w, dst_h, dst_w):
        src_long = max(src_h, src_w)
        dst_long = max(dst_h, dst_w)
        return cv2.INTER_AREA if dst_long < src_long else cv2.INTER_CUBIC

    def resize_fixed_scale(img_rgb: np.ndarray, scale: float) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        inter = best_interp(h, w, new_h, new_w)
        return cv2.resize(img_rgb, (new_w, new_h), interpolation=inter)

    def center_crop_or_pad(img_rgb: np.ndarray, target: int, fill: int = 0):
        h, w = img_rgb.shape[:2]
        if h > target:
            top = (h - target) // 2
            img_rgb = img_rgb[top:top + target]
            h = target
        if w > target:
            left = (w - target) // 2
            img_rgb = img_rgb[:, left:left + target]
            w = target

        pad_top = (target - h) // 2 if h < target else 0
        pad_bottom = target - h - pad_top if h < target else 0
        pad_left = (target - w) // 2 if w < target else 0
        pad_right = target - w - pad_left if w < target else 0

        if pad_top or pad_bottom or pad_left or pad_right:
            img_rgb = cv2.copyMakeBorder(
                img_rgb, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=(fill, fill)
            )

        return img_rgb, pad_top, pad_left

    resized = ndimage.zoom(img, scale, order=1) > 122.
    resized = resized.astype(np.uint8) * 255

    out, pad_top, pad_left = center_crop_or_pad(resized, target, fill_value)
    
    return out


def update_coco_with_clusters(csv_file, input_json, output_json, cluster_col='cluster'):

    df = pd.read_csv(csv_file)
    with open(input_json, 'r') as f:
        coco_data = json.load(f)

    assert len(df) == len(coco_data['annotations']), "ann number of CSV and JSONis not match!!"
    
    for i, ann in enumerate(coco_data['annotations']):
        ann['category_id'] = int(df.iloc[i][cluster_col]) + 1
    
    coco_data['categories'] = [{"id": 1, "name": "Globule", "supercategory": "mito"},
						{"id": 2, "name": "Short_Tubule", "supercategory": "mito"},
						{"id": 3, "name": "Mid_Tubule", "supercategory": "mito"},
						{"id": 4, "name": "Long_tubule", "supercategory": "mito"},
                        {"id": 5, "name": "Branch", "supercategory": "mito"},
                        {"id": 6, "name": "Loop", "supercategory": "mito"}],
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Update completed: {output_json}")


def cal_skeleton_stats(mask_bool):
    """Count skeleton topology features from a binary mask.

    Args:
        mask_bool: 2D boolean array where True marks foreground pixels.

    Returns:
        Tuple of (n_endpoints, n_branch_points, skeleton_pixel_count).
        Endpoints have exactly one skeleton neighbor; branch points are counted
        as connected components with three or more skeleton neighbors.
    """
    skeleton = skeletonize(mask_bool)
    skel_len = int(skeleton.sum())
    if skel_len == 0:
        return 0, 0, 0

    # Count skeleton neighbors in the 8-connected neighborhood.
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

    endpoints = (skeleton > 0) & (neighbor_count == 1)
    branch_mask = (skeleton > 0) & (neighbor_count >= 3)

    n_endpoints = int(endpoints.sum())
    # Label branch components so adjacent pixels are not double-counted.
    _, n_branch = label(branch_mask)
    return n_endpoints, int(n_branch), skel_len


def cal_feret_diameter(mask_bool):
    """Compute the maximum Feret diameter of a binary mask.

    The Feret diameter is the largest pairwise distance between points on the
    convex hull of the mask contour.

    Args:
        mask_bool: 2D boolean array where True marks foreground pixels.

    Returns:
        Maximum Feret diameter in pixels, or 0.0 if the mask is empty.
    """
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    main_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(main_contour, returnPoints=True)
    hull = hull.reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        return 0.0
    diff = hull[:, None, :] - hull[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return float(dist.max())


def cal_num_holes(mask_bool):
    """Count enclosed holes in a binary mask.

    Holes are foreground-free regions that become filled when the mask is
    morphologically filled.

    Args:
        mask_bool: 2D boolean array where True marks foreground pixels.

    Returns:
        Number of hole connected components.
    """
    filled = binary_fill_holes(mask_bool)
    holes = filled & (~mask_bool)
    _, num_holes = label(holes)
    return int(num_holes)


def cal_mask_metrics(mask_bool):
    """Compute morphology metrics for a single binary mask.

    Args:
        mask_bool: 2D boolean array where True marks foreground pixels.

    Returns:
        Dict of scalar metrics (area, perimeter, length, aspect_ratio,
        circularity, solidity, skeleton stats, and hole count), or None if the
        mask has no foreground contour.
    """
    area = int(mask_bool.sum())

    contours, _ = cv2.findContours(mask_bool.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(main_contour, closed=True))

    (_, (rw, rh), _) = cv2.minAreaRect(main_contour)
    max_side, min_side = max(rw, rh), min(rw, rh)
    aspect_ratio = max_side / min_side if min_side > 0 else 0.0

    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    length = cal_feret_diameter(mask_bool)

    hull = cv2.convexHull(main_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    n_endpoints, n_branch_points, skel_length = cal_skeleton_stats(mask_bool)
    num_holes = cal_num_holes(mask_bool)

    return {
        "area": area,
        "perimeter": round(perimeter, 3),
        "length": round(length, 3),
        "aspect_ratio": round(aspect_ratio, 4),
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "skel_length": skel_length,
        "n_endpoints": n_endpoints,
        "n_branch_points": n_branch_points,
        "num_holes": num_holes,
    }


def classify_clusters(cluster_df):
    """Assign macro morphology labels to clusters using sequential rules.

    Each step only applies to clusters not yet labeled; earlier steps take
    priority over later ones.

    Args:
        cluster_df: DataFrame indexed by cluster id with at least these mean
            columns: ``aspect_ratio_mean``, ``circularity_mean``,
            ``n_branch_points_mean``.

    Returns:
        pandas.Series indexed by cluster with macro labels:
        ``error``, ``branch``, ``loop``, ``globule``, ``long``, ``middle``,
        or ``short``.

    Rule order:
        1. ``aspect_ratio_mean > 10`` -> ``error``
        2. ``n_branch_points_mean > 1`` and ``circularity_mean < 0.35`` -> ``branch``
        3. ``n_branch_points_mean > 1`` and ``circularity_mean >= 0.35`` -> ``loop``
        4. ``aspect_ratio_mean < 1.3`` -> ``globule``
        5. Remaining clusters: normalize ``aspect_ratio_mean - 1`` to [0, 1] and
           split by 3:4:3 thresholds into ``short`` / ``middle`` / ``long``
           (normalized values in [0, 0.3), [0.3, 0.7), [0.7, 1]).
    """
    labels = pd.Series(index=cluster_df.index, dtype=object)

    def remaining():
        return labels[labels.isna()].index

    def assign(cond):
        """Return cluster ids among unlabeled rows that satisfy ``cond``."""
        idx = remaining()
        sub = cond.reindex(idx).fillna(False)
        return sub[sub].index

    labels[assign(cluster_df["aspect_ratio_mean"] > 10)] = "error"

    cond_branch = (cluster_df["n_branch_points_mean"] > 1) & (cluster_df["circularity_mean"] < 0.35)
    labels[assign(cond_branch)] = "branch"

    cond_loop = (cluster_df["n_branch_points_mean"] > 1) & (cluster_df["circularity_mean"] > 0.35)
    labels[assign(cond_loop)] = "loop"

    labels[assign(cluster_df["aspect_ratio_mean"] < 1.3)] = "globule"

    rest = remaining()
    if len(rest) > 0:
        ar = cluster_df.loc[rest, "aspect_ratio_mean"] - 1.0
        ar = ar.clip(lower=0)
        max_ar = ar.max()
        norm = ar / max_ar if max_ar > 0 else ar * 0.0
        for cl in rest:
            v = norm[cl]
            if v >= 0.7:
                labels[cl] = "long"
            elif v >= 0.3:
                labels[cl] = "middle"
            else:
                labels[cl] = "short"

    return labels


METRIC_KEYS = [
    "area", "perimeter", "length", "aspect_ratio", "circularity",
    "solidity", "skel_length", "n_endpoints", "n_branch_points", "num_holes",
]


def export_cluster_morphology_metrics(
    cluster_dir: str,
    output_dir: str,
    n_clusters: int = 12,
    cluster_prefix: str = "KmeansLabelRefine_",
    file_prefix: str | None = None,
) -> dict:
    """Compute morphology metrics per mask and summarize clusters into macro classes.

    Reads binary PNG masks from ``{cluster_dir}/{cluster_prefix}{i}/``, computes
    per-instance metrics, aggregates by cluster, assigns macro labels via
    ``classify_clusters``, and writes two CSV files under ``output_dir``.

    Args:
        cluster_dir: root directory containing ``KmeansLabelRefine_*`` subfolders.
        output_dir: directory for ``*_instance_metrics.csv`` and ``*_cluster_summary.csv``.
        n_clusters: number of cluster subfolders to scan (0 .. n_clusters-1).
        cluster_prefix: prefix of each cluster subdirectory name.
        file_prefix: optional stem prefix for output CSV filenames; defaults to
            the basename of ``output_dir``.

    Returns:
        Dict with ``instance_df``, ``summary_df``, ``instance_csv``, and ``summary_csv``.
        Empty dict if no masks were processed.
    """
    os.makedirs(output_dir, exist_ok=True)
    if file_prefix is None:
        file_prefix = os.path.basename(os.path.normpath(output_dir))

    rows = []
    for c in range(n_clusters):
        cdir = os.path.join(cluster_dir, f"{cluster_prefix}{c}")
        if not os.path.isdir(cdir):
            print(f"[warn] missing cluster dir: {cdir}")
            continue
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(".png"))
        print(f"[cluster {c}] {len(files)} masks")
        for fn in files:
            fpath = os.path.join(cdir, fn)
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  [skip] cannot read {fpath}")
                continue
            if img.ndim == 3:
                img = img[..., 0]
            mask_bool = img > 0
            if mask_bool.sum() == 0:
                continue
            m = cal_mask_metrics(mask_bool)
            if m is None:
                continue
            m_out = {"filename": fn, "cluster": c}
            m_out.update(m)
            rows.append(m_out)

    if not rows:
        print("No masks processed. Check cluster_dir.")
        return {}

    df = pd.DataFrame(rows)
    instance_csv = os.path.join(output_dir, f"{file_prefix}_instance_metrics.csv")
    df.to_csv(instance_csv, index=False)
    print(f"\n[saved] per-instance metrics -> {instance_csv}  ({len(df)} rows)")

    agg = df.groupby("cluster")[METRIC_KEYS].mean()
    agg.columns = [f"{m}_mean" for m in agg.columns]
    agg.insert(0, "count", df.groupby("cluster").size())
    agg["macro"] = classify_clusters(agg)

    summary_csv = os.path.join(output_dir, f"{file_prefix}_cluster_summary.csv")
    agg.to_csv(summary_csv)
    print(f"[saved] per-cluster summary  -> {summary_csv}")

    df["macro"] = df["cluster"].map(agg["macro"])
    df.to_csv(instance_csv, index=False)
    print(f"[updated] per-instance metrics (added 'macro') -> {instance_csv}")

    show_cols = ["aspect_ratio_mean", "circularity_mean", "n_branch_points_mean"]
    show = agg[["count"] + show_cols + ["macro"]].copy()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== per-cluster mean metrics + macro ===")
    print(show.round(3).to_string())

    return {
        "instance_df": df,
        "summary_df": agg,
        "instance_csv": instance_csv,
        "summary_csv": summary_csv,
    }

