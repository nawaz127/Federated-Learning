from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.xai.xai_metrics import _resize_cam

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_imagenet(img: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor or array back to [0,1] RGB image.
    Accepts CHW or HWC. Returns HWC in float32.
    
    Handles two cases:
    1. ImageNet-normalized float tensor (values ~[-2, +3]): reverses normalization
    2. uint8 image [0, 255]: scales to [0, 1] then reverses normalization
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)

    if img.ndim == 2:
        img = img[:, :, None]

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = img.astype(np.float32)

    # Detect uint8-range images: values [0, 255] have max >> 10
    # ImageNet-normalized float tensors typically have max ~2-4, NEVER > 10
    if img.max() > 10.0:
        img = img / 255.0

    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return img


def normalize_imagenet(img: np.ndarray) -> np.ndarray:
    """
    Normalize a [0,1] RGB image with ImageNet statistics.
    """
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def overlay_gradcam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a CAM heatmap on an image. Returns uint8 BGR image for saving with cv2.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)

    cam_resized = _resize_cam(cam, image.shape[1], image.shape[0])
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = heatmap * alpha + image * (1 - alpha)
    return overlay.astype(np.uint8)


def save_gradcam_panel(
    x: torch.Tensor,
    cam: np.ndarray,
    out_dir: str | Path,
    round_num: int,
    idx: int,
    true_label: int | None = None,
    pred_label: int | None = None,
    auc: float | None = None,
) -> Path:
    """
    Save a 3-panel Grad-CAM visualization.
    """
    out_dir = Path(out_dir) / f"round_{round_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = denormalize_imagenet(x[0])
    cam_resized = _resize_cam(cam, base.shape[1], base.shape[0])

    overlay = overlay_gradcam(base, cam_resized)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(base)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM++")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    title = "Overlay"
    if true_label is not None and pred_label is not None:
        title = f"Overlay (GT={true_label}, Pred={pred_label})"
    if auc is not None:
        title = f"{title}\nDel-AUC={auc:.3f}"
    axes[2].set_title(title)
    axes[2].axis("off")

    out_path = out_dir / f"gradcam_{idx}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
