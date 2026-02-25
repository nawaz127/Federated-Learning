# Federated Learning Framework for Interpretable Lung Cancer Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.0%2B-orange.svg)](https://flower.dev/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Abstract

A privacy-preserving **Federated Learning (FL)** framework for multi-institutional classification of lung CT scans. The system enables collaborative model training across distributed hospital clients without sharing raw patient data, while integrating a comprehensive **Explainable AI (XAI)** pipeline that monitors both diagnostic performance and model trustworthiness in real-time.

### Key Contributions

1. **9 deep learning architectures** — CNNs, Vision Transformers, and custom hybrid models — benchmarked under federated training with IID and non-IID data splits.
2. **Real-time XAI Probing** — Grad-CAM++, LIME, SHAP, and Attention Rollout generate publication-quality heatmaps with faithfulness metrics (Deletion/Insertion AUC) computed every round.
3. **Custom Hybrid Model (LSeTNet)** — a novel CNN-Transformer hybrid with SE attention, CBAM, and learnable positional embeddings, designed from scratch for medical imaging.
4. **Robust FL Infrastructure** — FedAvg / FedProx / Trimmed Mean aggregation, atomic JSON checkpointing, BatchNorm recalibration, and deterministic reproducibility.

---

## Model Zoo

The framework supports **9 architectures** via a unified `model_factory.py`, spanning three families:

### CNN Models

| Model | Class | Parameters | Size (MB) | Backbone Source |
|:------|:------|----------:|----------:|:----------------|
| `resnet50` | `ResNet50` | 24,694,851 | 94.2 | torchvision (ImageNet) |
| `densenet121` | `DenseNet121Medical` | 7,544,707 | 28.8 | torchvision (ImageNet) |
| `mobilenetv3` | `MobileNetV3` | 3,599,539 | 13.7 | torchvision (ImageNet) |

### Transformer Models

| Model | Class | Parameters | Size (MB) | Backbone Source |
|:------|:------|----------:|----------:|:----------------|
| `vit` | `VisionTransformer` | 86,258,435 | 329.0 | timm ViT-Base/16 (ImageNet) |
| `vit_tiny` | `VisionTransformer` | 5,689,283 | 21.7 | timm ViT-Tiny/16 (ImageNet) |
| `swin_tiny` | `SwinTiny` | 27,979,133 | 106.7 | timm Swin-T (ImageNet) |

### Hybrid Models (Custom Architectures)

| Model | Class | Parameters | Size (MB) | Architecture |
|:------|:------|----------:|----------:|:-------------|
| `LSeTNet` | `LSeTNet` | 9,427,109 | 36.0 | CNN (SE + CBAM) → Transformer (custom, no pretrain) |
| `hybridmodel` | `HybridViTCNNMLP` | 97,690,819 | 372.7 | ViT-Base + ResNet18 → MLP fusion |
| `hybridswin` | `HybridSwinDenseNetMLP` | 35,451,005 | 135.2 | Swin-T + DenseNet121 → MLP fusion |

### Architecture Details

**LSeTNet** (Lightweight SE-Transformer Network) — Custom hybrid designed from scratch:
- Initial Conv → 3x Residual-SE Blocks → Final Conv-SE → CBAM spatial attention → Transformer encoder (4-head, 512-dim, learnable positional embeddings) → MLP classifier
- Includes DropPath (stochastic depth), LayerScale, and Pre-LN transformer blocks
- No pretrained backbone — all weights trained from scratch

**HybridViTCNNMLP** — Dual-backbone fusion:
- **ViT branch:** Processes 3-channel RGB input, outputs L2-normalized features
- **CNN branch:** ResNet18 processes 1-channel grayscale (ImageNet weights mean-pooled to 1-ch), outputs L2-normalized features
- Features concatenated → 3-layer MLP classifier

**HybridSwinDenseNetMLP** — Dual-backbone fusion:
- **Swin-T branch:** Processes 3-channel RGB, outputs L2-normalized features
- **DenseNet121 branch:** Processes 1-channel grayscale (conv0 weights mean-pooled), outputs L2-normalized features
- Features concatenated → 3-layer MLP classifier

### Common Model Interface

All 9 models implement a consistent API for the FL pipeline:

| Method | Purpose |
|:-------|:--------|
| `forward(x)` | Standard inference, returns logits `[B, num_classes]` |
| `extract_features(x)` | Returns backbone features before classifier |
| `get_embedding(x)` | Returns embeddings for SHAP / t-SNE analysis |
| `save_features(hook)` | Forward hook storing activations for XAI |
| `remove_feature_hook()` | Removes XAI hook to save memory during training |
| `unfreeze_backbone()` | Unfreezes pretrained backbone for full fine-tuning |

---

## Explainable AI (XAI) Pipeline

The framework integrates 4 XAI methods that run during federated evaluation:

| Method | Type | Works With | Output |
|:-------|:-----|:-----------|:-------|
| **Grad-CAM++** | Gradient-based | All models | Class activation heatmap |
| **LIME** | Perturbation-based | All models | Superpixel importance heatmap |
| **SHAP** | Deep Explainer | All models | Channel-aggregated attribution map |
| **Attention Rollout** | Attention-based | ViT, Swin only | Attention flow heatmap |

### XAI Output Format

For each test sample, the pipeline generates:
- **Per-method panels** (4-5 columns): Original Image → Heatmap → Overlay → Prediction Info (+ LIME Boundaries)
- **Combined panel**: All methods stacked in rows for side-by-side comparison
- **Faithfulness metrics**: Deletion AUC, Insertion AUC, cross-method agreement, temporal stability (SSIM, Pearson)

### Faithfulness Interpretation

| Deletion AUC | Interpretation |
|:-------------|:---------------|
| < 0.3 | High faithfulness — model relies on highlighted regions |
| 0.3 - 0.6 | Moderate faithfulness |
| > 0.6 | Low faithfulness — model may use spurious correlations |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    server.py (Central Orchestrator)              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ MedicalFL    │  │ Aggregation  │  │ Global Evaluation      ││
│  │ Strategy     │  │ FedAvg /     │  │ - Val/Test metrics     ││
│  │ - Round mgmt │  │ FedProx /    │  │ - BN recalibration     ││
│  │ - Config     │  │ TrimmedMean  │  │ - Best model saving    ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
│         ▲                                      │                │
│         │            Model Weights             │                │
│         └──────────────────────────────────────┘                │
└────────────────────────────┬────────────────────────────────────┘
                             │ gRPC (Flower)
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │  client.py  │   │  client.py  │   │  client.py  │
   │  Client 1   │   │  Client 2   │   │  Client 3   │
   │             │   │             │   │             │
   │ ┌─────────┐ │   │ ┌─────────┐ │   │ ┌─────────┐ │
   │ │ Local   │ │   │ │ Local   │ │   │ │ Local   │ │
   │ │ Dataset │ │   │ │ Dataset │ │   │ │ Dataset │ │
   │ └────┬────┘ │   │ └────┬────┘ │   │ └────┬────┘ │
   │      ▼      │   │      ▼      │   │      ▼      │
   │ ┌─────────┐ │   │ ┌─────────┐ │   │ ┌─────────┐ │
   │ │train_   │ │   │ │train_   │ │   │ │train_   │ │
   │ │eval.py  │ │   │ │eval.py  │ │   │ │eval.py  │ │
   │ └────┬────┘ │   │ └────┬────┘ │   │ └────┬────┘ │
   │      ▼      │   │      ▼      │   │      ▼      │
   │ ┌─────────┐ │   │ ┌─────────┐ │   │ ┌─────────┐ │
   │ │XAI Probe│ │   │ │XAI Probe│ │   │ │XAI Probe│ │
   │ │Manager  │ │   │ │Manager  │ │   │ │Manager  │ │
   │ └─────────┘ │   │ └─────────┘ │   │ └─────────┘ │
   └─────────────┘   └─────────────┘   └─────────────┘
```

---

## Repository Structure

```
Federated-Learning/
├── server.py                        # FL server - strategy, aggregation, global eval
├── client.py                        # FL client - local training, validation, XAI probing
├── prediction.py                    # Standalone inference on saved models
├── generate_publication_results.py  # Generate tables/figures for paper
├── run_server_5clients_iid.bat      # Windows batch launcher (server + clients)
├── requirements.txt                 # Python dependencies
│
├── models/                          # Model definitions
│   ├── model_factory.py             # Factory: get_model(), FocalLoss, LabelSmoothingLoss
│   ├── resnet_model.py              # ResNet50 with custom classifier head
│   ├── densenet121.py               # DenseNet121 for medical imaging
│   ├── mobilenetv3.py               # MobileNetV3-Large (lightweight)
│   ├── vit.py                       # Vision Transformer (ViT-Base & ViT-Tiny)
│   ├── swin_tiny.py                 # Swin Transformer Tiny
│   ├── lsetnet.py                   # LSeTNet - custom CNN+Transformer hybrid
│   ├── vit_resnet_hybrid.py         # HybridViTCNNMLP - ViT + ResNet18 fusion
│   └── swint_densenet_hybrid.py     # HybridSwinDenseNetMLP - Swin-T + DenseNet121 fusion
│
├── utils/                           # Utility modules
│   ├── dataloder.py                 # CTScanDataset, CLAHE preprocessing, data splits
│   ├── train_eval.py                # ModelTrainer - training loop, metrics, TensorBoard
│   ├── common_utils.py              # Tensor validation, NaN/Inf guards
│   ├── xai_utils.py                 # Legacy XAI wrappers (GradCAM, LIME, SHAP)
│   ├── xai_config.py                # Canonical XAI configuration
│   ├── xai_faithfulness_tracker.py  # Per-round faithfulness tracking
│   └── xai/                         # Modular XAI components
│       ├── federated_xai_manager.py     # Orchestrates XAI probing in FL loop
│       ├── comprehensive_xai_viz.py     # Publication-quality multi-panel XAI figures
│       ├── gradcam_pp.py                # Grad-CAM++ implementation
│       ├── lime_explainer.py            # LIME explanation wrapper
│       ├── shap_explainer.py            # SHAP DeepExplainer wrapper
│       ├── attention_rollout.py         # Attention Rollout for transformers
│       ├── xai_visualization.py         # Denormalization, overlay, panel helpers
│       ├── xai_metrics.py               # Deletion/Insertion AUC, CAM similarity
│       └── xai_plot.py                  # Additional XAI plotting utilities
│
├── Federated_Dataset/               # Data directory
│   ├── Clients_IID/                 # IID-partitioned data (5 clients)
│   │   └── Client_1/ ... Client_5/
│   ├── Clients_nonIID/              # Non-IID-partitioned data (4 clients)
│   │   └── client_1/ ... client_4/
│   ├── train/                       # Centralized train split
│   ├── val/                         # Centralized validation split (server-side eval)
│   └── test/                        # Held-out test split (final evaluation)
│
├── Result/                          # All experimental outputs
│   ├── clientresult/                # Per-client results
│   │   └── client_X/
│   │       ├── checkpoints/         # Local model weights (.pth)
│   │       ├── metrics/             # Per-round classification reports
│   │       ├── predictions/         # Per-round predictions
│   │       └── xai/                 # XAI visualizations per round
│   │           └── round_Y/
│   │               ├── gradcam_sample_*.png
│   │               ├── lime_sample_*.png
│   │               ├── shap_sample_*.png
│   │               ├── attention_rollout_sample_*.png
│   │               └── combined_xai_sample_*.png
│   └── FLResult/                    # Server-side results
│       └── fl_results_TIMESTAMP/
│           ├── best_model.pth
│           ├── history_round_*.json
│           ├── training_curves.png
│           └── final_classification_report.json
│
└── publication_ready/               # Auto-generated publication materials
    ├── figures/                     # Figures for paper
    ├── tables/                      # LaTeX + CSV tables
    ├── latex/                       # LaTeX source includes
    └── performance_summary.csv
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: NVIDIA RTX 4060 or higher)
- Anaconda or venv

### Setup

```bash
# Clone the repository
git clone https://github.com/nawaz127/Federated-Learning.git
cd Federated_Learning

# Create and activate environment
conda create -n fl_medical python=3.10
conda activate fl_medical

# Install PyTorch (with CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

The system classifies lung CT scans into 3 classes. The `dataloder.py` module handles CLAHE preprocessing and stratified splitting.

```
Federated_Dataset/
├── train/
│   ├── Benign/        # Non-cancerous nodules
│   ├── Malignant/     # Confirmed carcinomas
│   └── Normal/        # Healthy parenchyma
├── val/
│   ├── Benign/
│   ├── Malignant/
│   └── Normal/
└── test/
    ├── Benign/
    ├── Malignant/
    └── Normal/
```

For federated experiments, split data into per-client directories under `Clients_IID/` or `Clients_nonIID/` using the provided augmentation scripts:

```bash
python utils/auto_augment_federated_iid.py      # IID partition
python utils/auto_augment_non_iid.py             # Non-IID partition
```

---

## Usage

### Quick Start (Windows)

```batch
run_server_5clients_iid.bat
```

### Manual Launch

#### 1. Start the Server

```bash
python server.py \
    --rounds 20 \
    --min-clients 3 \
    --model resnet50 \
    --aggregation fedavg \
    --learning-rate 0.0003 \
    --local-epochs 3 \
    --fraction-fit 1.0
```

**Server Arguments:**

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--model` | `resnet50` | Architecture (see Model Zoo above) |
| `--rounds` | `20` | Number of federated rounds |
| `--min-clients` | `3` | Minimum clients per round |
| `--aggregation` | `fedavg` | Strategy: `fedavg`, `fedprox`, `trimmed_mean` |
| `--mu` | `0.01` | FedProx proximal term weight |
| `--learning-rate` | `0.0003` | Client learning rate |
| `--local-epochs` | `3` | Local training epochs per round |
| `--fraction-fit` | `1.0` | Fraction of clients sampled for training |

#### 2. Connect Clients (separate terminals)

```bash
# Client 1
python client.py --client-id 1 --data-dir "./Federated_Dataset/Clients_IID/Client_1"

# Client 2
python client.py --client-id 2 --data-dir "./Federated_Dataset/Clients_IID/Client_2"

# Client 3
python client.py --client-id 3 --data-dir "./Federated_Dataset/Clients_IID/Client_3"
```

### Standalone Prediction

```bash
python prediction.py --model resnet50 --checkpoint Result/FLResult/.../best_model.pth --image path/to/ct_scan.jpg
```

---

## Training Pipeline Details

### Per-Round Flow

1. **Server** broadcasts global model weights + config to selected clients
2. **Each client:**
   - Loads global weights into local model
   - Trains for `local_epochs` on local data (AdamW + CosineAnnealing LR)
   - Evaluates on local validation set (accuracy, F1-macro, per-class metrics)
   - Runs **XAI Probe** (Grad-CAM++ heatmaps + Deletion/Insertion AUC)
   - On final round: runs **heavy XAI** (LIME, SHAP, Attention Rollout) with comprehensive visualization
   - Returns updated weights + metrics to server
3. **Server** aggregates weights (FedAvg / FedProx / Trimmed Mean)
4. **Server** evaluates global model on held-out validation/test set
5. **Server** saves best model checkpoint + round history (atomic JSON writes)

### Medical Preprocessing

- **CLAHE** contrast enhancement for lung CT nodule visibility
- **Albumentations** augmentation: ShiftScaleRotate, HorizontalFlip, GaussNoise, ElasticTransform
- **ImageNet normalization** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Input size: 224x224 RGB (grayscale CT → 3-channel conversion)

### Robustness Features

- **Deterministic seeds** (`torch.manual_seed`, `cudnn.deterministic=True`)
- **NaN/Inf guards** on all model outputs via `validate_tensor()`
- **Atomic JSON writes** (temp file → rename) to prevent checkpoint corruption
- **BatchNorm recalibration** (100 batches + `reset_running_stats()`) after aggregation
- **Mixed precision** (`torch.amp.autocast`) for memory-efficient training on RTX 4060
- **Focal Loss / Label Smoothing** for class-imbalanced medical datasets

---

## Outputs

### Server Results (`Result/FLResult/`)

| File | Content |
|:-----|:--------|
| `best_model.pth` | Best global model weights |
| `history_round_*.json` | Per-round metrics (loss, accuracy, F1, XAI scores) |
| `training_curves.png` | Convergence plots (loss, accuracy, F1 over rounds) |
| `final_classification_report.json` | Per-class precision/recall/F1 |

### Client Results (`Result/clientresult/client_X/`)

| Directory | Content |
|:----------|:--------|
| `checkpoints/` | Local model weights per round |
| `metrics/` | Classification reports per round |
| `predictions/` | Model predictions per round |
| `xai/round_Y/` | XAI visualizations (see below) |

### XAI Visualization Outputs

For each test sample on the final round:

```
xai/round_20/
├── gradcam_sample_0.png            # [Original | Heatmap | Overlay | Prediction]
├── lime_sample_0.png               # [Original | Heatmap | Overlay | Prediction | Boundaries]
├── shap_sample_0.png               # [Original | Heatmap | Overlay | Prediction]
├── attention_rollout_sample_0.png  # (ViT/Swin only)
└── combined_xai_sample_0.png       # All methods in one figure
```

---

## Publication Results

Auto-generate all tables and figures for research papers:

```bash
python generate_publication_results.py
```

Outputs to `publication_ready/`:
- LaTeX tables (overall performance, IID vs non-IID, per-class, XAI faithfulness, statistical significance, rankings)
- CSV companion tables
- XAI visualization figures
- Performance summary

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
