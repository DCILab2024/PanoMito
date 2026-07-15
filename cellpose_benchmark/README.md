# Cellpose benchmark

Baseline comparison using a custom Cellpose model. See the [installation guide](../README.md) at the repository root before use.

## Installation

Before running `cellpose_benchmark.py`, set up Cellpose following the [official instructions](https://github.com/MouseLand/cellpose) and install additional dependencies:

```bash
pip install -r cellpose_requirements.txt
```

## Run

From the **repository root**:

```bash
python cellpose_benchmark/cellpose_benchmark.py
```

**Input**

- **Images:** `./data/test/` (PNG)
- **Ground truth:** `./data/test/_test_subdataset_gt.json`
- **Model:** `./MODEL/cellpose_model`

**Output** (under `./results/cellpose_pre/`)

- `npy/` — per-image Cellpose segmentation results (`*_seg.npy`)
- `cellpose_pred_coco.json` — COCO-format predictions
- `cellpose_pred_coco_adjust_id.json` — predictions with IDs aligned to ground truth
- Evaluation metrics (AP, AR, F1) printed to the terminal
