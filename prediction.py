"""
Standalone Prediction & XAI Visualization Script
=================================================
Loads a trained FL checkpoint, runs inference on a single image (or a
directory of images), and generates publication-quality XAI overlays
using the same preprocessing pipeline as training.

Usage:
    # Single image
    python prediction.py --model resnet50 --checkpoint Result/FLResult/.../best_model.pth --image path/to/ct_scan.jpg

    # All images in a directory
    python prediction.py --model resnet50 --checkpoint Result/FLResult/.../best_model.pth --image-dir Federated_Dataset/test/Malignant

    # Choose XAI method
    python prediction.py --model vit --checkpoint model.pth --image scan.jpg --method gradcampp --alpha 0.45
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_factory import get_model

logger = logging.getLogger(__name__)

# ── Constants (must match training pipeline in utils/dataloder.py) ──────────
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
IMG_SIZE = 224

# ImageNet normalization — same as training pipeline
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Preprocessing (mirrors CTScanDataset exactly) ──────────────────────────
def preprocess_ct_image(
    img_path: str,
    img_size: int = IMG_SIZE,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Load and preprocess a CT scan image exactly as the training pipeline does:
      1. Load as grayscale
      2. Resize to img_size x img_size
      3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
      4. Normalize to [0, 255]
      5. Convert grayscale → 3-channel RGB
      6. Scale to [0, 1], apply ImageNet normalization
      7. Return as [1, 3, H, W] tensor

    Returns:
        orig_rgb: Original image as RGB uint8 numpy array (for display)
        x: Preprocessed tensor [1, 3, H, W] ready for model input
    """
    # Step 1: Load as grayscale
    img_u8 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise ValueError(f"Could not load image: {img_path}")

    # Step 2: Resize
    img_u8 = cv2.resize(img_u8, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    # Step 3: CLAHE (same as _apply_medical_preprocessing in dataloder.py)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_u8 = clahe.apply(img_u8)

    # Step 4: Normalize to [0, 255]
    img_u8 = cv2.normalize(img_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 5: Grayscale → 3-channel RGB (same as CTScanDataset._load_and_preprocess)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    # Keep a copy for display
    orig_rgb = img_rgb.copy()

    # Step 6: ImageNet normalization (same as val_transform in dataloder.py)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # Step 7: HWC → CHW → batch dimension
    x = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()  # [1, 3, H, W]

    return orig_rgb, x


# ── Model Loading ──────────────────────────────────────────────────────────
def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    """Load checkpoint with flexible key handling (supports various save formats)."""
    sd = torch.load(path, map_location=device, weights_only=False)

    # Unwrap nested dicts
    if isinstance(sd, dict):
        for key in ["model_state_dict", "state_dict", "weights", "model"]:
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        # Filter non-tensor entries if mixed
        if not all(isinstance(v, torch.Tensor) for v in sd.values()):
            sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)} or sd

    # Strip common prefixes
    new_sd = {}
    for k, v in sd.items():
        k_clean = k
        if k_clean.startswith("module."):
            k_clean = k_clean[7:]
        if k_clean.startswith("model."):
            k_clean = k_clean[6:]
        new_sd[k_clean] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    logger.info(f"Checkpoint loaded from {path}")


# ── Target Layer Selection (reuses pipeline logic) ─────────────────────────
def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Select the correct Grad-CAM target layer for each model architecture.
    Mirrors the logic in client.py's get_gradcam_target_layer().
    """
    # CNN models — use last conv layer or specific block
    if model_name == "resnet50":
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            return model.backbone.layer4
    elif model_name == "densenet121":
        if hasattr(model, "densenet") and hasattr(model.densenet, "features"):
            return _find_last_conv(model.densenet.features)
    elif model_name == "mobilenetv3":
        if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            return model.backbone.features[-1]
    elif model_name == "LSeTNet":
        return _find_last_conv(model)

    # Transformer models — use final norm layer
    elif model_name in ("vit", "vit_tiny"):
        if hasattr(model, "vit") and hasattr(model.vit, "norm"):
            return model.vit.norm
    elif model_name == "swin_tiny":
        if hasattr(model, "swin") and hasattr(model.swin, "norm"):
            return model.swin.norm

    # Hybrid models — use CNN branch's last conv
    elif model_name == "hybridmodel":
        if hasattr(model, "cnn"):
            return _find_last_conv(model.cnn)
    elif model_name == "hybridswin":
        if hasattr(model, "densenet") and hasattr(model.densenet, "features"):
            return _find_last_conv(model.densenet.features)

    # Fallback
    layer = _find_last_conv(model)
    if layer is not None:
        return layer
    raise RuntimeError(f"Cannot determine Grad-CAM target layer for model: {model_name}")


def _find_last_conv(module: nn.Module) -> nn.Module | None:
    """Find the last Conv2d layer in a module."""
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


# ── Visualization Helpers ──────────────────────────────────────────────────
def normalize01(a: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    a = a.astype(np.float32)
    a_min, a_max = a.min(), a.max()
    if a_max - a_min < 1e-8:
        return np.zeros_like(a)
    return (a - a_min) / (a_max - a_min)


def overlay_heatmap(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a heatmap on an RGB image."""
    H, W = img_rgb.shape[:2]
    heat_resized = cv2.resize(heatmap, (W, H))
    heat_colored = cv2.applyColorMap(
        (heat_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1.0, heat_rgb, alpha, 0)


# ── CAM Engines ────────────────────────────────────────────────────────────
class CAMBase:
    """Base class for gradient-based CAM methods with proper hook management."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_act)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out

    def _save_grad(self, module, gin, gout):
        self.gradients = gout[0]

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _post_process(self, cam: torch.Tensor) -> np.ndarray:
        cam = torch.relu(cam).detach().cpu().numpy()
        return normalize01(cam)


class GradCAMEngine(CAMBase):
    """Standard Grad-CAM."""

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        A = self.activations
        dYdA = self.gradients

        # Handle both 4D (CNN: [B,C,H,W]) and 3D (Transformer: [B,L,D]) activations
        if A.dim() == 4:
            weights = dYdA[0].mean(dim=(1, 2))
            cam = (weights.view(-1, 1, 1) * A[0]).sum(dim=0)
        elif A.dim() == 3:
            # Transformer output: [B, L, D]
            weights = dYdA[0].mean(dim=0)  # [D]
            cam_1d = (A[0] * weights).sum(dim=-1)  # [L]
            # Reshape to 2D (skip CLS token if present)
            n_tokens = cam_1d.shape[0]
            if int(n_tokens**0.5) ** 2 == n_tokens:
                side = int(n_tokens**0.5)
            else:
                side = int((n_tokens - 1) ** 0.5)
                cam_1d = cam_1d[1:]  # Remove CLS token
            cam = cam_1d.view(side, side)
        else:
            cam = A[0].mean(dim=0) if A.dim() > 1 else A[0]

        return self._post_process(cam)


class GradCAMPPEngine(CAMBase):
    """Grad-CAM++ with improved weighting."""

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        A = self.activations[0]
        dY = self.gradients[0]

        if A.dim() == 3:
            # Transformer: [L, D] — fall back to Grad-CAM weighting
            weights = dY.mean(dim=0)
            cam_1d = (A * weights).sum(dim=-1)
            n_tokens = cam_1d.shape[0]
            if int(n_tokens**0.5) ** 2 == n_tokens:
                side = int(n_tokens**0.5)
            else:
                side = int((n_tokens - 1) ** 0.5)
                cam_1d = cam_1d[1:]
            cam = cam_1d.view(side, side)
        else:
            # CNN: [C, H, W] — full Grad-CAM++ formula
            dY2 = dY**2
            dY3 = dY2 * dY
            eps = 1e-6
            sum_A = (A * dY3).sum(dim=(1, 2))
            denom = 2.0 * dY2 + sum_A[:, None, None]
            denom = torch.where(denom != 0, denom, torch.ones_like(denom) * eps)
            alpha = dY2 / denom
            weights = (alpha * torch.relu(dY)).sum(dim=(1, 2))
            cam = (weights.view(-1, 1, 1) * A).sum(dim=0)

        return self._post_process(cam)


def smoothgrad_campp(
    model: nn.Module,
    target_layer: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    samples: int = 10,
    noise_std: float = 0.1,
) -> np.ndarray:
    """SmoothGrad-CAM++: Average Grad-CAM++ over noisy copies."""
    cams = []
    base = x.clone().detach()
    for _ in range(samples):
        engine = GradCAMPPEngine(model, target_layer)
        noise = torch.randn_like(base) * noise_std
        cam = engine.generate(base + noise, class_idx=class_idx)
        cams.append(cam)
        engine.close()
    return normalize01(np.mean(np.stack(cams, axis=0), axis=0))


# ── Guided Backpropagation ─────────────────────────────────────────────────
class GuidedBackprop:
    """Guided Backpropagation — hooks into ReLU/GELU for guided gradients."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self._register()

    def _register(self):
        def relu_backward_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        for m in self.model.modules():
            if isinstance(m, (nn.ReLU, nn.GELU)):
                self.handles.append(m.register_full_backward_hook(relu_backward_hook))

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        # Sum absolute gradients across channels: [1, 3, H, W] → [H, W]
        gb = x.grad.detach().abs().squeeze(0).mean(dim=0).cpu().numpy()
        return normalize01(gb)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ── Predict & Explain ──────────────────────────────────────────────────────
def predict_and_explain(
    model: nn.Module,
    model_name: str,
    img_path: str,
    device: torch.device,
    method: str = "gradcampp",
    alpha: float = 0.45,
    smooth_samples: int = 10,
    smooth_noise: float = 0.10,
    out_dir: str = "./prediction_output",
) -> dict:
    """Run inference + XAI explanation on a single image."""
    os.makedirs(out_dir, exist_ok=True)

    # Preprocess (same pipeline as training)
    orig_rgb, x = preprocess_ct_image(img_path)
    x = x.to(device)

    # Get target layer for CAM
    target_layer = get_gradcam_target_layer(model, model_name)

    # ── Inference ──
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # ── XAI Explanation ──
    if method == "gradcam":
        engine = GradCAMEngine(model, target_layer)
        heatmap = engine.generate(x, class_idx=pred_idx)
        engine.close()
    elif method == "gradcampp":
        engine = GradCAMPPEngine(model, target_layer)
        heatmap = engine.generate(x, class_idx=pred_idx)
        engine.close()
    elif method == "smoothgrad_campp":
        heatmap = smoothgrad_campp(
            model, target_layer, x, pred_idx,
            samples=smooth_samples, noise_std=smooth_noise,
        )
    elif method == "guided_gradcam":
        engine = GradCAMPPEngine(model, target_layer)
        heat_cam = engine.generate(x, class_idx=pred_idx)
        engine.close()
        gb = GuidedBackprop(model)
        gb_map = gb.generate(x, class_idx=pred_idx)
        gb.close()
        # Resize to match
        h, w = heat_cam.shape
        gb_resized = cv2.resize(gb_map, (w, h))
        heatmap = normalize01(heat_cam * gb_resized)
    else:
        raise ValueError(f"Unknown XAI method: {method}. "
                         f"Choose from: gradcam, gradcampp, smoothgrad_campp, guided_gradcam")

    # ── Create overlay ──
    overlay_img = overlay_heatmap(orig_rgb, heatmap, alpha=alpha)

    # ── Save outputs ──
    stem = Path(img_path).stem
    overlay_path = os.path.join(out_dir, f"{stem}_{method}_overlay.png")
    heatmap_path = os.path.join(out_dir, f"{stem}_{method}_heatmap.npy")
    probs_path = os.path.join(out_dir, f"{stem}_probs.json")

    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    np.save(heatmap_path, heatmap)
    with open(probs_path, "w") as f:
        json.dump(
            {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
            f, indent=2,
        )

    # ── Save publication-quality panel ──
    panel_path = os.path.join(out_dir, f"{stem}_{method}_panel.png")
    _save_prediction_panel(
        orig_rgb, heatmap, overlay_img, pred_name, pred_idx,
        confidence, probs, method, panel_path,
    )

    logger.info(f"Prediction: {pred_name} ({confidence:.1%}) | Saved to {out_dir}")

    return {
        "image": img_path,
        "pred_idx": pred_idx,
        "pred_name": pred_name,
        "confidence": confidence,
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "overlay_path": overlay_path,
        "panel_path": panel_path,
    }


def _save_prediction_panel(
    orig_rgb: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    pred_name: str,
    pred_idx: int,
    confidence: float,
    probs: np.ndarray,
    method: str,
    save_path: str,
) -> None:
    """Save a 4-panel figure: [Original | Heatmap | Overlay | Prediction Info]."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original CT Scan", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Heatmap
    H, W = orig_rgb.shape[:2]
    heat_resized = cv2.resize(heatmap, (W, H))
    im = axes[1].imshow(heat_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"{method.upper()} Heatmap", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Prediction info
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].axis("off")
    axes[3].set_facecolor("#f8f9fa")
    axes[3].set_title("Prediction", fontsize=13, fontweight="bold")

    y = 0.85
    axes[3].text(0.5, y, "Predicted:", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="#7f8c8d")
    y -= 0.10
    axes[3].text(0.5, y, pred_name, ha="center", va="center",
                 fontsize=18, fontweight="bold", color="#2c3e50")
    y -= 0.14
    axes[3].text(0.5, y, f"Confidence: {confidence:.1%}", ha="center", va="center",
                 fontsize=14, fontweight="bold",
                 color="#2ecc71" if confidence > 0.7 else "#e67e22")

    # Class probabilities bar
    y -= 0.18
    for i, name in enumerate(CLASS_NAMES):
        bar_y = y - i * 0.12
        axes[3].text(0.05, bar_y, f"{name}:", ha="left", va="center",
                     fontsize=10, color="#2c3e50")
        axes[3].text(0.95, bar_y, f"{probs[i]:.1%}", ha="right", va="center",
                     fontsize=10, fontweight="bold",
                     color="#2ecc71" if i == pred_idx else "#95a5a6")

    fig.suptitle("Lung CT Scan — Prediction & Explanation",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── CLI Entry Point ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict & Explain lung CT scans using trained FL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "densenet121", "mobilenetv3",
                                 "vit", "vit_tiny", "swin_tiny",
                                 "LSeTNet", "hybridmodel", "hybridswin"],
                        help="Model architecture (must match the checkpoint)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single CT scan image")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Path to a directory of CT scan images")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of output classes")
    parser.add_argument("--method", type=str, default="gradcampp",
                        choices=["gradcam", "gradcampp", "smoothgrad_campp", "guided_gradcam"],
                        help="XAI visualization method")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="Heatmap overlay opacity (0-1)")
    parser.add_argument("--smooth-samples", type=int, default=10,
                        help="Number of SmoothGrad samples (for smoothgrad_campp)")
    parser.add_argument("--smooth-noise", type=float, default=0.10,
                        help="SmoothGrad noise std (for smoothgrad_campp)")
    parser.add_argument("--out-dir", type=str, default="./prediction_output",
                        help="Output directory for results")
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Provide either --image or --image-dir")

    # Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model = get_model(
        args.model, num_classes=args.num_classes, pretrained=False,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    logger.info(f"Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for f in sorted(Path(args.image_dir).iterdir()):
            if f.suffix.lower() in img_exts:
                image_paths.append(str(f))
        logger.info(f"Found {len(image_paths)} images in {args.image_dir}")

    if not image_paths:
        logger.error("No images found to process")
        return

    # Run predictions
    results = []
    for i, img_path in enumerate(image_paths):
        logger.info(f"[{i+1}/{len(image_paths)}] Processing: {img_path}")
        try:
            res = predict_and_explain(
                model=model,
                model_name=args.model,
                img_path=img_path,
                device=device,
                method=args.method,
                alpha=args.alpha,
                smooth_samples=args.smooth_samples,
                smooth_noise=args.smooth_noise,
                out_dir=args.out_dir,
            )
            results.append(res)
        except Exception as e:
            logger.error(f"Failed on {img_path}: {type(e).__name__}: {e}")

    # Save summary
    summary_path = os.path.join(args.out_dir, "prediction_summary.json")
    summary = []
    for r in results:
        summary.append({k: v for k, v in r.items() if k != "heat"})
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Done — {len(results)}/{len(image_paths)} predictions saved to {args.out_dir}")


if __name__ == "__main__":
    main()
