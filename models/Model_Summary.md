
# Model Architecture Summary (DenseNet121, Swin-Tiny, LSeTNet)

> Only the three main models used in the Federated Learning pipeline for Breast Ultrasound Classification (3 classes: Benign, Malignant, Normal).

---

## Quick Comparison Table

| # | Model         | Total Params | Model Size | Normalization         | FedBN Relevant | Input      |
|---|---------------|--------------|------------|----------------------|----------------|------------|
| 1 | DenseNet121   | 7,544,707    | 28.8 MB    | BatchNorm            | Yes            | 1ch → 3ch  |
| 2 | Swin-Tiny     | 27,979,133   | 106.7 MB   | LayerNorm            | No             | 1ch → 3ch  |
| 3 | LSeTNet       | 9,427,109    | 36.0 MB    | BatchNorm + LayerNorm| Yes            | 1ch → 3ch  |

### Parameter Size Ranking (smallest → largest)
```
DenseNet121 :    7.5M  ████████
LSeTNet     :    9.4M  ██████████
Swin-Tiny   :   28.0M  █████████████████████████████
```

---

## Detailed Parameter Breakdown

### 1. ResNet50

| Component           | Parameters     |
|---------------------|----------------|
| **Total**           | 24,694,851     |
| Backbone            | 23,508,032     |
| Classifier Head     | 1,186,819      |
| Conv2d layers       | 23,454,912     |
| Linear layers       | 1,181,187      |
| BatchNorm layers    | 58,752         |

- **Source:** `torchvision.models.resnet50` (ImageNet1K_V1 pretrained)
- **Backbone:** Standard ResNet50 — 4 residual stages, each with Bottleneck blocks (Conv 1×1 → Conv 3×3 → Conv 1×1 + skip connection)
- **GradCAM++ Target:** `backbone.layer4` (last residual stage)
- **XAI Hook:** `backbone.layer4` (registers forward hook for feature maps)
- **Default Dropout:** 0.5

**Classifier Head Architecture:**
```
BatchNorm1d(2048)
→ Dropout(0.5)
→ Linear(2048 → 512)
→ ReLU
→ BatchNorm1d(512)
→ Dropout(0.25)
→ Linear(512 → 256)
→ ReLU
→ BatchNorm1d(256)
→ Dropout(0.125)
→ Linear(256 → 3)
```

---


### 1. DenseNet121

| Component           | Parameters     |
|---------------------|----------------|
| **Total**           | 7,544,707      |
| Backbone            | 6,953,856      |
| Classifier Head     | 590,851        |
| Conv2d layers       | 6,870,208      |
| Linear layers       | 590,851        |
| BatchNorm layers    | 83,648         |

- **Source:** `torchvision.models.densenet121` (DenseNet121_Weights.IMAGENET1K_V1 pretrained)
- **Backbone:** DenseNet-121 — 4 dense blocks with dense connections (each layer receives all preceding feature maps)
- **GradCAM++ Target:** Last Conv2d layer (auto-detected via `find_last_conv_layer`)
- **XAI Hook:** `densenet.features.norm5`
- **Default Dropout:** 0.5
- **Input:** Grayscale → 3ch via channel repeat

**Classifier Head Architecture:**
```
Linear(1024 → 512)
→ ReLU
→ Dropout(0.5)
→ Linear(512 → 128)
→ ReLU
→ Linear(128 → 3)
```

---

### 3. MobileNetV3-Large

| Component           | Parameters     |
|---------------------|----------------|
| **Total**           | 3,599,539      |
| Backbone            | 2,971,952      |
| Classifier Head     | 627,587        |
| Conv2d layers       | 2,947,552      |
| Linear layers       | 624,131        |
| BatchNorm layers    | 27,856         |

- **Source:** `torchvision.models.mobilenet_v3_large` (MobileNet_V3_Large_Weights.IMAGENET1K_V1 pretrained)
- **Backbone:** MobileNetV3-Large — inverted residual blocks with Squeeze-and-Excite (SE) attention + h-swish activation
- **GradCAM++ Target:** Last Conv2d layer (auto-detected)
- **XAI Hook:** `backbone.features[-1]` (last feature block)
- **Default Dropout:** 0.5
- **Input:** Grayscale → 3ch via channel repeat

**Classifier Head Architecture:**
```
BatchNorm1d(960)
→ Dropout(0.5)
→ Linear(960 → 512)
→ ReLU
→ BatchNorm1d(512)
→ Dropout(0.25)
→ Linear(512 → 256)
→ ReLU
→ BatchNorm1d(256)
→ Dropout(0.125)
→ Linear(256 → 3)
```

---

### 4. ViT-Base (Vision Transformer)

| Component           | Parameters     |
|---------------------|----------------|
| **Total**           | **86,258,435** |
| Backbone            | 85,798,656     |
| Classifier Head     | 459,779        |
| Conv2d layers       | 590,592        |
| Linear layers       | 85,477,379     |
| LayerNorm layers    | 38,400         |
| Other (embeddings)  | 152,064        |

- **Source:** `timm.create_model("vit_base_patch16_224")` pretrained on ImageNet
- **Backbone:** ViT-Base — 12 transformer encoder layers, 12 attention heads, 768-dim embeddings, 16×16 patch embedding → 196 patches + 1 CLS token = 197 tokens
- **GradCAM++ Target:** `vit.norm` (final LayerNorm before classifier)
- **Attention Rollout:** Extracts attention weights via hooks on `block.attn.attn_drop` (12 layers)
- **XAI Hook:** `vit.norm`
- **Default Dropout:** 0.3
- **Drop Path Rate:** 0.1 (stochastic depth)
- **Input:** Grayscale → 3ch via channel repeat

**Classifier Head Architecture:**
```
Linear(768 → 512)
→ ReLU
→ Dropout(0.3)
→ Linear(512 → 128)
→ ReLU
→ Linear(128 → 3)
```

**Why ViT is 10× larger than DenseNet121:**
```
ViT self-attention: Each of 12 layers has:
  - Q, K, V projections: 768 × 768 × 3 = 1,769,472 params
  - Output projection: 768 × 768 = 589,824 params
  - MLP: 768 × 3072 × 2 = 4,718,592 params
  Per layer total: ~7.1M params × 12 layers = ~85M params
```

---


### 2. Swin-Tiny (Shifted Window Transformer)

| Component           | Parameters     |
|---------------------|----------------|
| **Total**           | 27,979,133     |
| Backbone            | 27,519,354     |
| Classifier Head     | 459,779        |
| Conv2d layers       | 4,704          |
| Linear layers       | 27,926,339     |
| LayerNorm layers    | 24,768         |
| Other (embeddings)  | 23,322         |

- **Source:** `timm.create_model("swin_tiny_patch4_window7_224")` pretrained on ImageNet
- **Backbone:** Swin-Tiny — hierarchical transformer with shifted window attention, 4 stages with [2, 2, 6, 2] blocks, dimensions [96, 192, 384, 768]
- **GradCAM++ Target:** `swin.norm` (final LayerNorm)
- **Attention Rollout:** Uses windowed attention — standard rollout not directly compatible
- **XAI Hook:** `swin.norm`
- **Default Dropout:** 0.3
- **Drop Path Rate:** 0.1
- **Input:** Grayscale → 3ch via channel repeat

**Classifier Head Architecture:**
```
Linear(768 → 512)
→ ReLU
→ Dropout(0.3)
→ Linear(512 → 128)
→ ReLU
→ Linear(128 → 3)
```

---

### 6. Hybrid ViT-ResNet18

| Component           | Parameters     |
|---------------------|----------------|
| **Total**           | **97,690,819** |
| Backbone (combined) | 96,968,896     |
| Classifier Head     | 721,923        |
| Conv2d layers       | 11,751,232     |
| Linear layers       | 85,739,523     |
| BatchNorm layers    | 9,600          |
| LayerNorm layers    | 38,400         |
| Other (embeddings)  | 152,064        |

- **Source:** `timm` (ViT-Base) + `torchvision` (ResNet18)
- **Backbone:** Dual-branch fusion
  - **Branch 1:** ViT-Base `vit_base_patch16_224` (86M params) — takes 3ch input
  - **Branch 2:** ResNet18 (11M params) — adapted to 1ch grayscale input (conv1 weight averaged across RGB channels)
  - Features are L2-normalized before concatenation
- **GradCAM++ Target:** Last Conv2d in `cnn` branch (ResNet18)
- **XAI Hooks:** `vit.norm` + `cnn.layer4`
- **Default Dropout:** 0.3
- **Input:** 1ch → 3ch for ViT branch, 1ch for CNN branch

**Classifier Head Architecture:**
```
Linear(768 + 512 = 1280 → 512)
→ ReLU
→ Dropout(0.3)
→ Linear(512 → 128)
→ ReLU
→ Linear(128 → 3)
```

---

### 7. Hybrid Swin-DenseNet121

| Component           | Parameters     |
|---------------------|----------------|
| **Total**           | 35,451,005     |
| Backbone (combined) | 34,466,938     |
| Classifier Head     | 984,067        |
| Conv2d layers       | 6,868,640      |
| Linear layers       | 28,450,627     |
| BatchNorm layers    | 83,648         |
| LayerNorm layers    | 24,768         |
| Other (embeddings)  | 23,322         |

- **Source:** `timm` (Swin-Tiny) + `torchvision` (DenseNet121)
- **Backbone:** Dual-branch fusion
  - **Branch 1:** Swin-Tiny `swin_tiny_patch4_window7_224` (28M params) — takes 3ch input
  - **Branch 2:** DenseNet121 (7M params) — adapted to 1ch grayscale (conv0 weight averaged)
  - Features are L2-normalized before concatenation
- **GradCAM++ Target:** Last Conv2d in `densenet.features`
- **XAI Hooks:** `swin.norm` + `densenet.features.norm5`
- **Default Dropout:** 0.3
- **Input:** 1ch → 3ch for Swin branch, 1ch for DenseNet branch

**Classifier Head Architecture:**
```
Linear(768 + 1024 = 1792 → 512)
→ ReLU
→ Dropout(0.3)
→ Linear(512 → 128)
→ ReLU
→ Linear(128 → 3)
```

---


### 3. LSeTNet (Lightweight SE-Transformer Network)

| Component           | Parameters     |
|---------------------|----------------|
| **Total**           | 9,427,109      |
| Backbone            | 8,898,722      |
| Classifier Head     | 528,387        |
| Conv2d layers       | 7,210,786      |
| Linear layers       | 1,392,643      |
| BatchNorm layers    | 7,552          |
| LayerNorm layers    | 2,048          |
| Other (SE, CBAM, pos embed) | 814,080|

- **Source:** Custom architecture (no pretrained backbone)
- **Backbone:** Hybrid CNN-Transformer
  - Initial Conv2d(1, 32, 3×3) → BN → GELU
  - 3× Residual SE Blocks (Squeeze-and-Excite attention)
  - Conv2d(256, 512, 1×1) → CBAM (Channel + Spatial attention) → BN
  - Transformer Encoder (1 block default): Multi-head self-attention + LayerNorm + MLP + DropPath + LayerScale
  - Global Average Pooling
- **GradCAM++ Target:** Last Conv2d layer (auto-detected)
- **XAI Hook:** CBAM output layer
- **Default Dropout:** 0.5
- **Drop Path Rate:** 0.1
- **Layer Scale Init:** 1e-5
- **Input:** Native 1ch grayscale

**Classifier Head Architecture:**
```
Linear(512 → 1024)
→ ReLU
→ Dropout(0.5)
→ Linear(1024 → 3)
```

---


## GPU Memory Estimation (Training, fp32, per client)

| Model       | Model Size | ~Training Memory (batch=8) | 4 Clients Total | Fits 8GB? |
|-------------|------------|----------------------------|-----------------|-----------|
| DenseNet121 | 28.8 MB    | ~1.5 GB                    | ~6.0 GB         | Yes       |
| LSeTNet     | 36.0 MB    | ~1.8 GB                    | ~7.2 GB         | Tight     |
| Swin-Tiny   | 106.7 MB   | ~3.0 GB                    | ~12.0 GB        | No (bs=4) |

### 8 GB VRAM Settings (4 clients sharing GPU)

| Model       | batch_size | accum_steps | effective_batch | Gradient Checkpointing |
|-------------|------------|-------------|-----------------|------------------------|
| DenseNet121 | 8          | 1           | 8               | Not needed             |
| LSeTNet     | 8          | 1           | 8               | Not needed             |
| Swin-Tiny   | 4          | 2           | 8               | Recommended            |

---

## Federated Learning Compatibility Notes

### FedBN (Federated BatchNorm)
- **Applies to:** ResNet50, DenseNet121, MobileNetV3, LSeTNet, Hybrid ViT-RN18, Hybrid Swin-DN
- **Does NOT apply to:** ViT-Base, Swin-Tiny (use LayerNorm only — no running stats to exclude)
- FedBN excludes `running_mean`, `running_var`, `num_batches_tracked` from aggregation

### Normalization Behavior
| Model | Norm Type | FedBN Effect | XAI Eval Mode |
|-------|-----------|-------------|---------------|
| CNNs (ResNet, DenseNet, MobileNetV3) | BatchNorm | Excludes running stats | BN → eval(), track_running_stats=True |
| Transformers (ViT, Swin) | LayerNorm | No-op (all params aggregated) | LN → eval() only |
| Hybrids | Both | Excludes BN running stats | BN → eval()+track, LN → eval() |
| LSeTNet | Both | Excludes BN running stats | BN → eval()+track, LN → eval() |

### GradCAM++ Compatibility
| Model | Target Layer | Activation Shape | Spatial CAM |
|-------|-------------|------------------|-------------|
| ResNet50 | `backbone.layer4` | [2048, 7, 7] | 7×7 |
| DenseNet121 | Last Conv2d | [1024, 7, 7] | 7×7 |
| MobileNetV3 | Last Conv2d | [960, 7, 7] | 7×7 |
| ViT-Base | `vit.norm` | [197, 768] → reshaped to [768, 14, 14] | 14×14 |
| Swin-Tiny | `swin.norm` | [49, 768] → reshaped to [768, 7, 7] | 7×7 |


### Recommended Training Hyperparameters

| Model       | Learning Rate | Batch Size | Accum Steps | Effective Batch | Scheduler      |
|-------------|--------------|------------|-------------|-----------------|---------------|
| DenseNet121 | 3e-4         | 8          | 1           | 8               | warmup_cosine |
| Swin-Tiny   | 1e-4         | 4          | 2           | 8               | warmup_cosine |
| LSeTNet     | 3e-4         | 8          | 1           | 8               | warmup_cosine |
