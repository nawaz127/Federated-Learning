import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _reshape_transformer_activations(
    A: torch.Tensor, G: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape transformer activations from [num_tokens, embed_dim] → [embed_dim, grid_h, grid_w].

    Automatically detects whether a CLS token is present:
      - If num_tokens is a perfect square → no CLS (e.g. Swin Transformer).
      - If num_tokens - 1 is a perfect square → CLS at index 0 (e.g. ViT).
    """
    num_tokens, embed_dim = A.shape
    grid_size = int(math.isqrt(num_tokens))

    if grid_size * grid_size == num_tokens:
        # Perfect square — no CLS token (e.g. Swin)
        A_patches = A
        G_patches = G
    else:
        # Try stripping CLS token at index 0
        n_minus_1 = num_tokens - 1
        grid_size = int(math.isqrt(n_minus_1))
        if grid_size * grid_size == n_minus_1:
            A_patches = A[1:]
            G_patches = G[1:]
        else:
            raise ValueError(
                f"Cannot reshape {num_tokens} transformer tokens to a spatial grid. "
                f"Expected n² (no CLS) or n²+1 (with CLS) tokens."
            )

    # [num_patches, embed_dim] → [embed_dim, grid_h, grid_w]
    A_spatial = A_patches.permute(1, 0).reshape(embed_dim, grid_size, grid_size)
    G_spatial = G_patches.permute(1, 0).reshape(embed_dim, grid_size, grid_size)
    return A_spatial, G_spatial


def find_last_conv_layer(model: nn.Module) -> nn.Module | None:
    """Return the last nn.Conv2d layer in the model, or None if not found."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module | None:
    """Dynamically select the target layer for Grad-CAM++ based on model type."""
    if model_name == "resnet50":
        # Use layer4 (the last residual block) instead of the last individual Conv2d.
        # Individual Conv2d inside bottleneck blocks produce flat gradients due to
        # skip connections. layer4 output has proper spatial activation + gradient flow.
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            return model.backbone.layer4
        return find_last_conv_layer(model)

    if model_name in ["customcnn", "densenet121", "mobilenetv3", "LSeTNet"]:
        return find_last_conv_layer(model)

    if model_name == "vit":
        if hasattr(model, "vit") and hasattr(model.vit, "norm"):
            return model.vit.norm
        if hasattr(model, "vit") and hasattr(model.vit, "blocks") and len(model.vit.blocks) > 0:
            return model.vit.blocks[-1].norm1

    if model_name == "swin_tiny":
        if hasattr(model, "swin") and hasattr(model.swin, "norm"):
            return model.swin.norm
        if hasattr(model, "swin") and hasattr(model.swin, "layers") and len(model.swin.layers) > 0:
            return model.swin.layers[-1].blocks[-1].norm

    if model_name == "hybridmodel":
        if hasattr(model, "cnn"):
            return find_last_conv_layer(model.cnn)

    if model_name == "hybridswin":
        if hasattr(model, "densenet"):
            return find_last_conv_layer(model.densenet.features)

    return find_last_conv_layer(model)


def compute_gradcam_pp(
    model: nn.Module,
    x: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int | None = None,
) -> np.ndarray:
    """
    Compute Grad-CAM++ heatmap for a single image batch x (shape [1, C, H, W]).
    Returns a numpy array of shape [H, W] normalized to [0,1].
    """
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=False)

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks not triggered - check target_layer")

        A = activations[0][0]  # CNN: [C, H, W] | Transformer: [num_tokens, embed_dim]
        G = gradients[0][0]

        # Handle transformer activations (2D from LayerNorm) → spatial [C, H, W]
        if A.ndim == 2:
            logger.debug(
                f"Transformer activations detected: {A.shape}. "
                "Reshaping to spatial [C, H, W] for GradCAM++."
            )
            A, G = _reshape_transformer_activations(A, G)

        grads2 = G ** 2
        grads3 = G ** 3

        sum_activations = torch.sum(A, dim=(1, 2), keepdim=True)

        eps = 1e-7
        denominator = 2 * grads2 + sum_activations * grads3 + eps
        denominator = torch.clamp(denominator, min=eps)

        alpha = grads2 / denominator

        positive_gradients = torch.relu(G)

        weights = torch.sum(alpha * positive_gradients, dim=(1, 2))

        cam = torch.sum(weights.view(-1, 1, 1) * A, dim=0)
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        
        # NUMERICAL STABILITY: Check for NaN/Inf before normalization
        if not np.isfinite(cam_np).all():
            logger.error(f"GradCAM++ produced non-finite values: NaN={np.isnan(cam_np).sum()}, Inf={np.isinf(cam_np).sum()}")
            logger.error(f"  CAM shape: {cam_np.shape}, min={np.nanmin(cam_np)}, max={np.nanmax(cam_np)}")
            # DO NOT return zeros - raise exception for explicit failure
            raise RuntimeError("GradCAM++ numerical instability detected - cannot generate valid heatmap")
        
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        else:
            # All values are same (flat CAM) - this is valid but indicates no saliency
            logger.warning("GradCAM++ produced flat heatmap (no gradient signal)")
            # Return flat map with warning, not zeros
            cam_np = np.ones_like(cam_np) * 0.5  # Uniform 0.5 indicates no saliency
        
        # Final validation
        if not np.isfinite(cam_np).all():
            logger.error("GradCAM++ normalization produced non-finite values")
            raise RuntimeError("GradCAM++ normalization failed")

        return cam_np

    finally:
        handle_f.remove()
        handle_b.remove()
        model.zero_grad(set_to_none=True)
