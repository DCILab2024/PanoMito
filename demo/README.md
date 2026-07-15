# Demo

Example end-to-end pipelines for PanoMito. Complete the [installation guide](../README.md) at the repository root before use.

Run all commands from the **repository root**:

```bash
python demo/<script_name>.py
```

| Script | Purpose |
| --- | --- |
| `panomito_benchmark.py` | Predict + ground-truth evaluation (AP, AR, F1) |
| `predict_cluster_analysis.py` | Predict + morphological clustering |

Both scripts call the same prediction logic as [`predict.py`](../predict.py) (input: `./data/test/`, tile size 512, `merge_thresh=0.4`), then run additional analysis steps.

---

## Step-by-step demo

### 1. Benchmark — `panomito_benchmark.py`

Segmentation benchmark with ground-truth evaluation.

```bash
python demo/panomito_benchmark.py
```

**Input**

- **Images:** `./data/test/` (PNG)
- **Ground truth:** `./data/test/_test_subdataset_gt.json`
- **Model:** `./MODEL/PanoMitoSeg.pth`

**Output** (under `./results/panomito_benchmark/`)

- `_sub_predictor_use_NMS_dataset=all_merge_thresh=0.4_c=0.0_merge_q300.json` — COCO-format predictions
- `_sub_predictor_use_NMS_dataset=all_merge_thresh=0.4_c=0.0_adjust_id.json` — predictions with IDs aligned to ground truth
- Evaluation metrics printed to the terminal
- Sample visualizations in `visualization/`
- Timestamped log: `panomito_benchmark_YYYYMMDD_HHMMSS.txt`

---

### 2. Clustering — `predict_cluster_analysis.py`

Segmentation followed by morphological clustering and morphology class assignment.

**Pipeline steps:** segmentation → instance mask export → K-means clustering (K=12) → morphology classification summary.

```bash
python demo/predict_cluster_analysis.py
```

**Input**

- **Images:** `./data/test/` (PNG)
- **Models:** `./MODEL/PanoMitoSeg.pth`, `./MODEL/PanoMitoCluster.pth`

**Output** (under `./results/predict_cluster_analysis/`)

**Root directory**

- `_sub_predictor_use_NMS_dataset=all_merge_thresh=0.4_c=0.0_merge_q300.json` — COCO-format segmentation results
- `predict_cluster_analysis_instance_metrics.csv` — per-instance morphology metrics (with morphology class labels)
- `predict_cluster_analysis_cluster_summary.csv` — per-cluster mean metrics and morphology class assignment

**`MitoInstance/`**

- `{annotation_id:06d}_all_data.png` — per-instance 255×255 binary mask crops
- `all_z_np.npy` — autoencoder latent features used for clustering

**`Cluster/`**

- `mito_clusters.csv` — filename-to-cluster mapping (K-means labels 0–11)
- `clustering_umap.png` — 2D UMAP visualization of cluster embeddings
- `KmeansLabelRefine_0/` … `KmeansLabelRefine_11/` — instance masks grouped by cluster

---

### 3. Visualization

Upload images to [CVAT](https://www.cvat.ai/) and import the prediction JSON for interactive visualization.

---

Both scripts expose a `run_*` function. Edit the default arguments in `if __name__ == "__main__":`, or import the function to customize paths, thresholds, and post-processing options.
