# Brain Tumor Detection via Stacked Ensemble Learning

A deep learning pipeline for binary brain tumor classification from MRI scans. Three pre-trained CNNs (EfficientNetB7, Xception, InceptionV3) are fine-tuned as base learners, and their predictions are stacked into a LightGBM meta-learner tuned with Optuna — achieving strong accuracy with built-in explainability via Grad-CAM++ and SHAP.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Outputs](#outputs)
- [Results](#results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

---

## Overview

Brain tumor detection from MRI is a critical task in medical imaging. This project implements a **two-level stacked generalization** approach:

1. **Level 0 (Base Learners):** Three ImageNet-pretrained CNNs are fine-tuned on brain MRI data with ROI cropping and augmentation.
2. **Level 1 (Meta-Learner):** A LightGBM classifier is trained on out-of-fold (OOF) predictions from the base learners, combining their strengths while correcting individual weaknesses.

The pipeline is designed with **no data leakage** — OOF predictions are generated strictly on the training set via K-fold cross-validation, and the test set is never seen during any training or tuning step.

---

## Architecture

```
                    ┌──────────────────┐
                    │   Brain MRI      │
                    │   (224 × 224)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   ROI Cropping   │
                    │   + Augmentation │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
   ┌────────▼───────┐ ┌─────▼──────┐ ┌───────▼────────┐
   │ EfficientNetB7 │ │  Xception  │ │  InceptionV3   │
   │  (fine-tuned)  │ │ (fine-tuned│ │  (fine-tuned)  │
   └────────┬───────┘ └─────┬──────┘ └───────┬────────┘
            │                │                │
            │         OOF Predictions         │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼─────────┐
                    │     LightGBM     │
                    │  (Optuna-tuned)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Tumor / No Tumor│
                    └──────────────────┘
```

---

## Dataset

This project uses the [Brain Tumor Detection dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) from Kaggle.

| Class     | Description                  |
|-----------|------------------------------|
| `yes/`    | MRI scans with brain tumors  |
| `no/`     | MRI scans without tumors     |

The pipeline automatically handles:
- **Duplicate removal** via SHA-1 hashing of pixel content
- **Stratified splitting** into 60% train / 20% validation / 20% test
- **Overlap verification** to confirm zero leakage between splits

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU recommended (CPU works but is slow)

### Setup

```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection

pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.10
numpy
matplotlib
seaborn
scikit-learn
scikit-image
opencv-python
lightgbm
optuna
shap
pandas
scipy
```

> **Note:** `lightgbm`, `optuna`, and `shap` are auto-installed by the script if missing.

---

## Usage

### Google Colab (recommended)

1. Upload `archive.zip` (the Kaggle dataset) to `/content/`.
2. Upload or paste `brain_tumor_pipeline.py` into a Colab cell.
3. Run the cell — all outputs are saved to an `outputs/` directory.

### Local / Server

1. Edit the paths at the top of the script:

```python
ZIP_FILE_PATH    = "/path/to/archive.zip"
EXTRACTION_PATH  = "/path/to/extract/"
OUTPUT_DIR       = "outputs"
```

2. Run:

```bash
python brain_tumor_pipeline.py
```

### Key Configuration

All tunable settings are in Section 0 of the script:

| Parameter        | Default | Description                          |
|------------------|---------|--------------------------------------|
| `IMG_SIZE`       | 224     | Input image resolution               |
| `BATCH_SIZE`     | 32      | Training batch size                  |
| `EPOCHS_FINETUNE`| 15      | Max epochs per base model            |
| `PATIENCE_ES`    | 5       | Early stopping patience              |
| `N_FOLDS`        | 5       | K-fold splits for OOF generation     |
| `N_OPTUNA_TRIALS`| 30      | Optuna hyperparameter search trials  |
| `DPI`            | 600     | Figure resolution for saved images   |
| `RANDOM_SEED`    | 42      | Global seed for reproducibility      |

---

## Pipeline Stages

| #  | Stage                        | Description                                                                 |
|----|------------------------------|-----------------------------------------------------------------------------|
| 1  | Data Loading                 | Extract ZIP, collect file paths, remove duplicates via SHA-1 hashing        |
| 2  | Train/Val/Test Split         | 60/20/20 stratified split with overlap verification                         |
| 3  | ROI Cropping                 | Otsu threshold + contour detection to crop brain region                     |
| 4  | Augmentation                 | Rotation, shift, zoom, flip, brightness (applied to training set only)      |
| 5  | Base Model Building          | Consistent architecture: pretrained CNN → BN → GAP → Dropout → Sigmoid     |
| 6  | Base Model Training          | Fine-tune all three CNNs with early stopping and LR scheduling              |
| 7  | OOF Meta-Features            | K-fold OOF on train set; val/test predicted by fully-trained models         |
| 8  | Soft-Voting Ensemble         | Equal-weight probability averaging across the three base models             |
| 9  | LGBM Meta-Learner            | Optuna-tuned LightGBM trained on OOF features                              |
| 10 | Comprehensive Evaluation     | 12 metrics including MCC, Kappa, G-Mean, Brier score, log loss             |
| 11 | Training Curves              | Real accuracy/loss curves from `model.fit()` history objects                |
| 12 | ROC / Confusion Matrix       | Journal-quality ROC curve, confusion matrix heatmap, classification heatmap |
| 13 | Model Comparison             | Bar chart comparing all base models, ensemble, and meta-learner             |
| 14 | t-SNE                        | 2D visualization of meta-feature separability on the test set               |
| 15 | Grad-CAM++                   | Heatmap overlay showing which brain regions drive predictions               |
| 16 | SHAP                         | Feature importance showing each base model's contribution to the ensemble   |

---

## Outputs

All figures and results are saved to `outputs/`:

```
outputs/
├── training_curves.png          # Accuracy & loss per epoch (all 3 models)
├── training_curves.tif          # TIFF version for journal submission
├── lgbm_loss_curve.png          # Meta-learner log loss over boosting rounds
├── roc_curve.png                # ROC curve with AUC
├── roc_curve.tif
├── confusion_matrix.png         # Test set confusion matrix
├── confusion_matrix.tif
├── classification_heatmap.png   # Precision / recall / F1 heatmap
├── model_comparison.png         # Accuracy bar chart (base vs ensemble vs LGBM)
├── model_comparison.tif
├── comprehensive_metrics.png    # All 12 evaluation metrics
├── tsne_meta_features.png       # t-SNE scatter plot
├── gradcam_examples.png         # Grad-CAM++ overlays on test images
├── gradcam_examples.tif
├── shap_bar.png                 # SHAP feature importance
├── shap_summary.png             # SHAP beeswarm plot
├── augmentation_examples.png    # Visual examples of augmentations
└── sample_predictions.png       # 3×3 grid of predictions vs ground truth
```

All figures are saved at 600 DPI (PNG + TIFF where applicable), suitable for journal or conference submission.

---

## Results

Results will vary depending on the random seed and dataset version. The pipeline prints all metrics to the console and computes them from actual predictions (nothing is hardcoded).

### Metrics Reported

| Metric              | Type          |
|---------------------|---------------|
| Accuracy            | Performance   |
| Precision           | Performance   |
| Recall              | Performance   |
| F1 Score            | Performance   |
| ROC AUC             | Performance   |
| MCC                 | Performance   |
| Cohen's Kappa       | Performance   |
| Balanced Accuracy   | Performance   |
| Specificity         | Performance   |
| G-Mean              | Performance   |
| Brier Score         | Calibration   |
| Log Loss            | Calibration   |

---

## Project Structure

```
brain-tumor-detection/
│
├── brain_tumor_pipeline.py    # Complete pipeline (single self-contained script)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # Project license
│
└── outputs/                   # Generated at runtime (not committed)
    ├── *.png
    └── *.tif
```

---

## Configuration

### Changing the Dataset

The pipeline expects a ZIP file containing this structure:

```
archive.zip
└── Brain_Tumor_Detection/
    ├── yes/      # tumor images (.jpg, .jpeg, .png)
    └── no/       # non-tumor images
```

To use a different dataset, update these variables in Section 0:

```python
ZIP_FILE_PATH  = "/your/path/to/data.zip"
TUMOR_DIR      = "your_folder/positive_class"
NO_TUMOR_DIR   = "your_folder/negative_class"
```

### Adding More Base Models

Add entries to the `BASE_CONFIGS` list:

```python
BASE_CONFIGS = [
    ("EfficientNetB7", EfficientNetB7, 1e-5, 0.5, 0.01),
    ("Xception",       Xception,       1e-5, 0.6, 0.03),
    ("InceptionV3",    InceptionV3,    1e-4, 0.5, 0.01),
    # Add more:
    ("ResNet50",       ResNet50,       1e-4, 0.5, 0.01),
]
```

The OOF generation, ensemble, and meta-learner will automatically adapt.

---

## Reproducibility

The pipeline sets seeds for NumPy, TensorFlow, and scikit-learn via `RANDOM_SEED = 42`. However, perfect reproducibility on GPU requires additional steps due to non-deterministic CUDA operations. To enforce full determinism:

```python
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
```

Add these lines before the TensorFlow import if exact reproducibility is required (may reduce training speed).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{brain_tumor_ensemble_2025,
  title   = {Brain Tumor Detection via Stacked Ensemble Learning with Grad-CAM++ Explainability},
  author  = {Your Name},
  year    = {2025},
  url     = {https://github.com/<your-username>/brain-tumor-detection}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Kaggle Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) by Ahmed Hamada
- Pre-trained weights from [ImageNet](https://www.image-net.org/) via TensorFlow/Keras
- [LightGBM](https://github.com/microsoft/LightGBM), [Optuna](https://optuna.org/), [SHAP](https://github.com/shap/shap)
