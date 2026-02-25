import torch
import logging

logger = logging.getLogger(__name__)

def check_nan_inf_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        logger.warning(f"Tensor '{name}' contains NaN or Inf values. Shape: {tensor.shape}")

def validate_tensor(tensor, name, strict=False):
    """
    SECTION 11 — NUMERICAL STABILITY GLOBAL SAFETY LAYER
    Validates tensor for NaN/Inf values.
    
    By default, replaces NaN/Inf with 0 and logs a warning so that
    BN recalibration and evaluation forward passes can complete.
    Set strict=True to raise RuntimeError instead (e.g. before sending
    parameters to the server).
    
    Returns:
        The (possibly cleaned) tensor.
    """
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        msg = (f"Numerical instability in {name}: "
               f"{nan_count} NaN, {inf_count} Inf out of {tensor.numel()} elements")
        if strict:
            logger.critical(msg)
            raise RuntimeError(msg)
        else:
            logger.warning(msg + " — replacing with 0 to allow forward pass to continue")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor

