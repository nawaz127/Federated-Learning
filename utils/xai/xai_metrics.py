import base64
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def _sanitize_cam(cam: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert CAM to finite, contiguous float32 2D array."""
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    cam = np.asarray(cam)
    cam = np.squeeze(cam)

    if cam.ndim != 2:
        raise ValueError(f"CAM must be 2D after squeeze, got shape={cam.shape}")
    if cam.size == 0:
        raise ValueError("CAM is empty")

    cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    cam = cam.astype(np.float32, copy=False)
    if not cam.flags["C_CONTIGUOUS"]:
        cam = np.ascontiguousarray(cam)
    return cam


def _resize_cam(cam: np.ndarray | torch.Tensor, width: int, height: int) -> np.ndarray:
    """Resize CAM robustly, with torch interpolate fallback if OpenCV rejects dtype/layout."""
    cam = _sanitize_cam(cam)
    if cam.shape == (height, width):
        return cam

    try:
        return cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    except cv2.error:
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        cam_t = F.interpolate(cam_t, size=(height, width), mode="bilinear", align_corners=False)
        return cam_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _zscore(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    mean = cam.mean()
    std = cam.std()
    if std < 1e-8:
        return cam - mean
    return (cam - mean) / (std + 1e-8)


def compute_cam_similarity(cam1: np.ndarray, cam2: np.ndarray) -> dict[str, float]:
    """
    Compute similarity metrics between two CAMs.
    Returns dict with keys: ssim, pearson, cosine.
    """
    cam1 = _sanitize_cam(cam1)
    cam2 = _sanitize_cam(cam2)

    if cam1.shape != cam2.shape:
        cam2 = _resize_cam(cam2, cam1.shape[1], cam1.shape[0])

    cam1n = _zscore(cam1)
    cam2n = _zscore(cam2)

    cam1_flat = cam1n.flatten()
    cam2_flat = cam2n.flatten()

    pearson = float(np.corrcoef(cam1_flat, cam2_flat)[0, 1])
    if np.isnan(pearson):
        pearson = 0.0

    dot = float(np.dot(cam1_flat, cam2_flat))
    norm1 = float(np.linalg.norm(cam1_flat))
    norm2 = float(np.linalg.norm(cam2_flat))
    cosine = 0.0 if (norm1 == 0.0 or norm2 == 0.0) else float(dot / (norm1 * norm2))

    data_range = float(cam2n.max() - cam2n.min())
    if data_range <= 0:
        data_range = 1.0
    try:
        ssim_score = float(ssim(cam1n, cam2n, data_range=data_range))
    except Exception:
        ssim_score = 0.0

    return {"pearson": pearson, "ssim": ssim_score, "cosine": cosine}


def compute_deletion_auc(
    model: torch.nn.Module,
    x: torch.Tensor,
    cam: np.ndarray,
    class_idx: int,
    device: torch.device,
    steps: int = 10,
) -> float:
    """
    Deletion AUC faithfulness metric.
    Lower values = better faithfulness (rapid confidence drop).
    """
    model.eval()

    with torch.no_grad():
        x_mod = x.clone().to(device)
        _, _, H, W = x_mod.shape

        cam = _resize_cam(cam, W, H)

        cam_flat = cam.reshape(-1)
        indices = np.argsort(-cam_flat)

        num_pixels = H * W
        pixels_per_step = max(num_pixels // steps, 1)

        scores: list[float] = []
        for step_i in range(steps + 1):
            logits = model(x_mod)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            scores.append(float(probs[0, class_idx].item()))

            if step_i == steps:
                break

            start_idx = step_i * pixels_per_step
            end_idx = min((step_i + 1) * pixels_per_step, num_pixels)
            pixels_to_delete = indices[start_idx:end_idx]

            for pixel_idx in pixels_to_delete:
                row = int(pixel_idx // W)
                col = int(pixel_idx % W)
                x_mod[0, :, row, col] = -1.0

        fractions = np.linspace(0.0, 1.0, len(scores))
        if hasattr(np, "trapezoid"):
            auc = float(np.trapezoid(scores, fractions))
        else:
            auc = float(np.trapz(scores, fractions))

        return auc


def compute_insertion_auc(
    model: torch.nn.Module,
    x: torch.Tensor,
    cam: np.ndarray,
    class_idx: int,
    device: torch.device,
    steps: int = 10,
) -> float:
    """
    Insertion AUC faithfulness metric.
    Higher values = better faithfulness (rapid confidence increase).
    """
    model.eval()

    with torch.no_grad():
        x_mod = x.clone().to(device)
        _, _, H, W = x_mod.shape

        cam = _resize_cam(cam, W, H)

        cam_flat = cam.reshape(-1)
        # Insert most relevant pixels first for standard insertion AUC behavior.
        indices = np.argsort(-cam_flat)

        num_pixels = H * W
        pixels_per_step = max(num_pixels // steps, 1)

        mask = torch.zeros((1, 1, H, W), device=device, dtype=x_mod.dtype)
        scores: list[float] = []
        for step_i in range(steps + 1):
            masked_input = x_mod * mask
            logits = model(masked_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            scores.append(float(probs[0, class_idx].item()))

            if step_i == steps:
                break

            start_idx = step_i * pixels_per_step
            end_idx = min((step_i + 1) * pixels_per_step, num_pixels)
            pixels_to_insert = indices[start_idx:end_idx]

            for pixel_idx in pixels_to_insert:
                row = int(pixel_idx // W)
                col = int(pixel_idx % W)
                mask[0, :, row, col] = 1.0

        fractions = np.linspace(0.0, 1.0, len(scores))
        if hasattr(np, "trapezoid"):
            auc = float(np.trapezoid(scores, fractions))
        else:
            auc = float(np.trapz(scores, fractions))

        return auc


def compute_xai_consistency(cam_t: np.ndarray, cam_t_plus_1: np.ndarray) -> float:
    """
    SECTION 9 — FEDERATED XAI STABILITY METRICS
    Cross-round CAM consistency: consistency = cosine_similarity(cam_t, cam_t+1)
    """
    sim = compute_cam_similarity(cam_t, cam_t_plus_1)
    return sim["cosine"]


def compute_cross_method_agreement(cam_list: list[np.ndarray]) -> dict[str, float]:
    """
    SECTION 3.2 — CROSS-METHOD AGREEMENT
    Compute agreement between multiple XAI methods for the same sample.
    """
    if not cam_list or len(cam_list) < 2:
        return {}

    agreements = {}
    for i in range(len(cam_list)):
        for j in range(i + 1, len(cam_list)):
            sim = compute_cam_similarity(cam_list[i], cam_list[j])
            agreements[f"agreement_{i}_{j}"] = sim["cosine"]

    agreements["agreement_mean"] = float(np.mean(list(agreements.values())))
    return agreements


def encode_cam_stack(cams: list[np.ndarray], downsample: int = 32) -> str:
    """
    Encode a CAM stack to a JSON string with base64 payload.
    """
    if not cams:
        return ""

    resized = []
    for cam in cams:
        cam = _resize_cam(cam, downsample, downsample)
        resized.append(cam)

    arr = np.stack(resized, axis=0).astype(np.float32)
    payload = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }
    return json.dumps(payload)


def decode_cam_stack(payload: str) -> np.ndarray | None:
    """
    Decode a CAM stack from a JSON string with base64 payload.
    """
    if not payload:
        return None
    try:
        data = json.loads(payload)
        raw = base64.b64decode(data["data"])
        arr = np.frombuffer(raw, dtype=np.dtype(data["dtype"]))
        return arr.reshape(data["shape"]).astype(np.float32)
    except Exception:
        return None
