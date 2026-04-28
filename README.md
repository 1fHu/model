# CAR-T Cell Killing Activity Classification via Multi-Modal Deep Learning

An end-to-end AI pipeline for classifying CAR-T cell killing efficacy against pancreatic cancer organoids using live-cell tracking data from organ-on-a-chip microfluidic devices.

---

## Overview

This project addresses a critical challenge in cancer immunotherapy: **quantitatively evaluating how effectively CAR-T cells kill pancreatic cancer organoids** at the single-cell level. Instead of relying on manual scoring, the system learns directly from cell trajectory data — combining time-series morphological features with track-level statistics through a multi-modal deep learning architecture.

The pipeline classifies each tracked organoid into one of three killing-response categories (0 = no response, 0.5 = partial, 1.0 = full killing), providing an objective and scalable alternative to expert annotation.

---

## Key Technical Contributions

| Component | Description |
|-----------|-------------|
| **BiLSTM + Attention** | Bidirectional LSTM with a learnable temporal attention mechanism for per-frame morphological time series |
| **TrackNet (MLP)** | Lightweight 3-layer MLP trained on 12 aggregate trajectory statistics |
| **Unified Fusion Model** | Joint end-to-end model fusing both branches with a shared classification head |
| **Soft Score Fusion** | Weight-sweep ensemble that finds the optimal linear combination of both models' probability outputs |
| **SHAP Explainability** | GradientExplainer applied to both the sequence model and the unified fusion model, producing per-feature importance rankings |
| **Ablation Framework** | Automated multi-config ablation study comparing full feature sets against SHAP-guided subsets |
| **Statistical Analysis** | Pairwise two-sample z-tests between killing classes for each feature, with SEM bar plots |

---

## Architecture

```
Raw CSVs (TrackMate format)
        │
        ▼
┌──────────────────────┐
│  Step 1: Data Pipeline│
│  • Load + merge spots │
│    and track CSVs     │
│  • Compute SPEED from │
│    frame differencing │
│  • Per-prefix scaling │
│  • Save .npz arrays   │
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────┐
│ BiLSTM  │  │ TrackNet │
│+Attn    │  │  (MLP)   │
│(Step 3) │  │(Step 2)  │
│SEQ×9    │  │  12 feats│
│→ 3 cls  │  │  → 3 cls │
└────┬────┘  └─────┬────┘
     │             │
     └──────┬──────┘
            ▼
   ┌─────────────────┐
   │  Fusion (Step 4/│
   │  7/8)           │
   │  • Soft ensemble│
   │  • FusionNet    │
   │  • Unified joint│
   └─────────────────┘
            │
            ▼
   ┌─────────────────┐
   │  Explainability │
   │  (Steps 5,6,7,9)│
   │  SHAP analysis  │
   └─────────────────┘
```

---

## Features

### Spot-Level (Time Series) — 9 features per frame
| Category | Features |
|----------|----------|
| Geometric | `RADIUS`, `AREA`, `PERIMETER`, `CIRCULARITY`, `SOLIDITY` |
| Ellipse-based | `ELLIPSE_MAJOR`, `ELLIPSE_MINOR`, `ELLIPSE_ASPECTRATIO` |
| Motion | `SPEED` (computed via frame-to-frame position differencing) |

### Track-Level (Aggregate Statistics) — 12 features
`TRACK_DURATION`, `TRACK_DISPLACEMENT`, `TRACK_MEAN_SPEED`, `TRACK_MAX_SPEED`, `TRACK_MIN_SPEED`, `TRACK_STD_SPEED`, `TOTAL_DISTANCE_TRAVELED`, `MAX_DISTANCE_TRAVELED`, `CONFINEMENT_RATIO`, `MEAN_STRAIGHT_LINE_SPEED`, `LINEARITY_OF_FORWARD_PROGRESSION`, `MEAN_DIRECTIONAL_CHANGE_RATE`

---

## Pipeline Steps

| Script | Purpose |
|--------|---------|
| `Config.py` | Centralized configuration (sequence length, paths, feature lists, random seeds) |
| `Step1_data.py` | Data ingestion, feature engineering, normalization, and dataset export |
| `Step2A_trackmodel.py` | Train and evaluate TrackNet (MLP) on track-level features |
| `Step3A_spotmodel.py` | Train and evaluate BiLSTM+Attention on spot time-series |
| `Step3B_test_spotmodel.py` | Evaluate the spot model on an external held-out test set |
| `Step4A_combined_model.py` | Soft-score fusion with weight sweep across both models |
| `Step5A_SHAP_spot.py` | SHAP GradientExplainer for the BiLSTM spot model |
| `Step6A_SHAP_track.py` | SHAP GradientExplainer for TrackNet |
| `Step7A_SHAP_Fusion.py` | Train FusionNet on concatenated raw + mid-level features; SHAP analysis |
| `Step8_unified_fusion.py` | End-to-end unified model training and external test evaluation |
| `Step9_SHAP_Unified.py` | SHAP analysis on the unified model with beeswarm summary plots |
| `Step10_Ablation.py` | Automated ablation study across feature subsets |
| `Step11_feature_distribution.py` | Feature distribution plots (mean ± SEM) and pairwise z-tests |

---

## Data Format

The pipeline expects **TrackMate-format CSV files** (as exported from ImageJ/Fiji):

```
Data/
├── TRACK/               # Training data — first batch (CART + NCI donors)
│   ├── {ID}_XY{n}_spots.csv
│   └── {ID}_XY{n}_tracks.csv
├── NEW/                 # Held-out test data (Device1–Device8)
│   ├── Device{n}_XY{m}_spots.csv
│   └── Device{n}_XY{m}_tracks.csv
├── CART annotations.xlsx
├── 2nd batch annotations.xlsx
└── PDO size change statistics_20250718.xlsx
```

> **Note:** Raw data is not included in this repository. Contact the authors for a Google Drive link.

---

## Getting Started

### 1. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn shap umap-learn openpyxl
```

### 2. Configure the Run

Edit `Config.py` to set:
- `SEQ_LEN`: sequence length for time-series model (`20`, `100`, or `360`)
- `DATA_DIR`: path to the data folder
- `HIDDEN_SIZE_LSTM`, `DROPOUT`, `EPOCHS`: model hyperparameters

### 3. Run the Pipeline

Execute scripts in order. Steps 3–9 should be re-run for each `SEQ_LEN` value:

```bash
python Step1_data.py          # Generate datasets (run once, or when data changes)
python Step2A_trackmodel.py   # Train TrackNet
python Step3A_spotmodel.py    # Train BiLSTM+Attention
python Step3B_test_spotmodel.py  # External test for spot model
python Step4A_combined_model.py  # Soft fusion evaluation
python Step5A_SHAP_spot.py    # SHAP for spot model
python Step6A_SHAP_track.py   # SHAP for track model
python Step7A_SHAP_Fusion.py  # Mid-level fusion + SHAP
python Step8_unified_fusion.py   # Unified model training + test
python Step9_SHAP_Unified.py  # SHAP for unified model
python Step10_Ablation.py     # Ablation study
python Step11_feature_distribution.py  # Statistical analysis
```

### 4. Outputs

Generated files are saved to:
- `Generated/` — preprocessed datasets (`.npz`, `.csv`), trained models (`.pth`)
- `Results/track/` — TrackNet evaluation plots
- `Results/{SEQ_LEN}/` — BiLSTM evaluation plots and SHAP figures
- `Results/Fusion_{SEQ_LEN}/` — Fusion model plots
- `Results/Unified_{SEQ_LEN}/` — Unified model plots
- `Generated/feature_distribution/` — Feature analysis plots and z-test p-values

---

## Design Decisions

**Why two separate branches instead of one model?**  
Spot-level features (morphology over time) and track-level statistics capture complementary aspects of organoid behavior. The BiLSTM branch can detect transient shape changes that aggregate statistics would smooth out, while the TrackNet branch captures long-range motility patterns not visible frame-by-frame.

**Why configurable `SEQ_LEN`?**  
Different experimental protocols yield different track lengths. Using 20, 100, or 360 frames allows trading off between capturing short-term dynamics and long-range behavioral patterns.

**Why SHAP on the fusion model?**  
The fusion model sees both raw inputs and hidden representations. SHAP on the original 21 features (9 spot + 12 track) allows direct biological interpretation of which physical properties most influence killing classification.

---

## Skills Demonstrated

- **Deep Learning**: Bidirectional LSTM with attention, MLP design, joint multi-input fusion networks
- **Time-Series Analysis**: Variable-length sequence modeling, temporal attention visualization
- **Multi-Modal Learning**: Feature-level and decision-level fusion strategies
- **Explainable AI (XAI)**: SHAP GradientExplainer applied to custom PyTorch models
- **Biomedical Data Engineering**: Parsing microscopy tracking outputs, computing derived biophysical features
- **Experimental Rigor**: Held-out external test sets, ablation studies, class-imbalance handling
- **Statistical Analysis**: Two-sample z-tests, SEM quantification across experimental conditions
- **Software Engineering**: Modular 11-step pipeline, centralized configuration, reproducible seeds

---

## Application Context

Pancreatic ductal adenocarcinoma (PDAC) has one of the lowest survival rates among cancers. CAR-T cell therapy is a promising but highly variable treatment — efficacy differs substantially across patients and T-cell batches. This system enables **automated, objective, and scalable assessment** of CAR-T killing activity directly from live-imaging data, without requiring manual expert annotation of each experiment. The approach is applicable to any organoid-based drug screening or immunotherapy evaluation workflow.
