import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from utils.common_utils import validate_tensor

logger = logging.getLogger(__name__)


class _ModelWrapper(torch.nn.Module):
    """Thin nn.Module wrapper so GradientExplainer accepts the model."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


def _reshape_shap_values(shap_values_raw, sample_shape):
    """Robustly reshape SHAP values back to image dimensions.

    Handles the different return formats across SHAP versions:
      - list of arrays (one per class), each (1, n_features)
      - single ndarray  (n_classes, 1, n_features)  or  (1, n_features)
      - shap.Explanation object
    Returns a list of arrays [class0, class1, ...] each shaped like sample_shape,
    or a single array shaped like sample_shape for single-output.
    """
    n_pixels = int(np.prod(sample_shape[1:]))  # C*H*W

    # --- shap.Explanation object (SHAP >= 0.42) --------------------------
    if hasattr(shap_values_raw, "values"):
        sv = np.asarray(shap_values_raw.values)
    elif isinstance(shap_values_raw, list):
        # Already a list of arrays, one per class
        return [np.asarray(sv).reshape(sample_shape) for sv in shap_values_raw]
    else:
        sv = np.asarray(shap_values_raw)

    # --- single-output (regression / binary) ------------------------------
    if sv.size == n_pixels or sv.size == int(np.prod(sample_shape)):
        return sv.reshape(sample_shape)

    # --- multi-class output -----------------------------------------------
    # sv might be (n_classes, 1, n_features) or (1, n_features, n_classes)
    if sv.ndim == 3 and sv.shape[0] * sv.shape[1] * sv.shape[2] == sv.size:
        # Try (n_classes, batch, features)
        if sv.shape[-1] == n_pixels:
            return [sv[i].reshape(sample_shape) for i in range(sv.shape[0])]
        # Try (batch, features, n_classes)
        if sv.shape[1] == n_pixels:
            return [sv[:, :, i].reshape(sample_shape) for i in range(sv.shape[2])]

    # Last resort: infer n_classes from total size
    n_classes = sv.size // n_pixels
    if n_classes >= 1 and sv.size == n_classes * n_pixels:
        sv_flat = sv.reshape(n_classes, *sample_shape)
        return [sv_flat[i] for i in range(n_classes)]

    # If nothing works, just return raw
    logger.warning(f"SHAP reshape heuristic failed: sv.shape={sv.shape}, sample_shape={sample_shape}")
    return sv


def compute_shap_explanation(model: torch.nn.Module, background: torch.Tensor, sample: torch.Tensor, save_path: str = None):
    """
    SECTION 6 — SHAP INTEGRATION
    Implement SHAP for CNN and hybrid transformer.
    
    Uses GradientExplainer (PyTorch-native) instead of DeepExplainer which
    requires TensorFlow. GradientExplainer computes expected gradients,
    which is equivalent to Integrated Gradients with a distribution of baselines.
    """
    model.eval()
    
    # Ensure background and sample are on the correct device
    device = next(model.parameters()).device
    background = background.to(device)
    sample = sample.to(device)

    # Wrap model in nn.Module so GradientExplainer can inspect it
    wrapped = _ModelWrapper(model)
    wrapped.eval()

    # Use GradientExplainer — PyTorch-native, no TensorFlow dependency
    # Falls back to KernelExplainer if GradientExplainer also fails
    try:
        explainer = shap.GradientExplainer(wrapped, background)
        shap_values = explainer.shap_values(sample)
    except Exception as e:
        logger.warning(f"SHAP GradientExplainer failed: {e}. Trying KernelExplainer.")
        # KernelExplainer is model-agnostic but slower
        # Flatten image for kernel explainer
        def _forward_flat(x_flat):
            x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device)
            x_tensor = x_tensor.reshape(-1, *sample.shape[1:])
            with torch.no_grad():
                return wrapped(x_tensor).cpu().numpy()
        
        bg_flat = background.cpu().numpy().reshape(background.shape[0], -1)
        sample_flat = sample.cpu().numpy().reshape(1, -1)
        explainer = shap.KernelExplainer(_forward_flat, bg_flat[:4])  # Use fewer bg samples
        shap_values_flat = explainer.shap_values(sample_flat, nsamples=100)
        # Reshape back to image dimensions (handles multi-class correctly)
        shap_values = _reshape_shap_values(shap_values_flat, sample.shape)
    
    # GradientExplainer returns shape (batch, C, H, W, num_classes) or a list.
    # Normalise into a list-of-arrays (one per class), each (1, C, H, W).
    if isinstance(shap_values, list):
        pass  # already per-class list
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 5:
            # (batch, C, H, W, num_classes) — split on last axis
            shap_values = [shap_values[..., i] for i in range(shap_values.shape[-1])]
        # else keep as-is (single-output)
    
    # Convert to numpy for visualization if needed
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Simple visualization: plot the attribution for the predicted class
        with torch.no_grad():
            preds = wrapped(sample)
            class_idx = torch.argmax(preds, dim=1).item()
        
        # shap_values is a list (one per class) or single array
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[class_idx])
        else:
            sv = np.asarray(shap_values)

        # sv shape: [1, C, H, W] or [C, H, W] or [H, W]
        # Sum over channels for heatmap
        sv = np.squeeze(sv)  # remove batch dim if present
        if sv.ndim == 3:
            heatmap = np.sum(np.abs(sv), axis=0)  # sum over channels
        elif sv.ndim == 2:
            heatmap = np.abs(sv)
        else:
            heatmap = np.abs(sv).reshape(int(np.sqrt(sv.size)), -1)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # Revert normalization for display if possible, or just show grayscale
        img_disp = sample[0].cpu().numpy().transpose(1, 2, 0)
        if img_disp.shape[2] == 1:
            plt.imshow(img_disp[:, :, 0], cmap='gray')
        else:
            # Simple de-norm for ImageNet if needed, or just clip
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
            plt.imshow(img_disp)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.colorbar()
        plt.title(f"SHAP Attribution (Class {class_idx})")
        
        plt.savefig(save_path)
        plt.close()
        
    return shap_values

def aggregate_federated_shap(client_shap_values_list):
    """
    Add federated SHAP aggregation support.
    Simple average of SHAP values across clients.
    """
    if not client_shap_values_list:
        return None
    
    # Assuming all SHAP values have the same structure
    # If it's a list of lists (one list per class per client)
    num_clients = len(client_shap_values_list)
    num_classes = len(client_shap_values_list[0])
    
    aggregated_shap = []
    for c_idx in range(num_classes):
        class_shap = [client_sv[c_idx] for client_sv in client_shap_values_list]
        avg_shap = np.mean(class_shap, axis=0)
        aggregated_shap.append(avg_shap)
        
    return aggregated_shap
