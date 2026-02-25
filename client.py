import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import time
import warnings
from collections import OrderedDict

import cv2
import flwr as fl
import grpc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.model_factory import FocalLoss, LabelSmoothingLoss, get_model
from utils.dataloder import (
    CTScanDataset,
    get_class_weights,
    get_medical_transforms,
)
from utils.train_eval import ModelTrainer
from utils.xai.federated_xai_manager import FederatedXAIManager

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# gRPC stability / payloads
os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "30000")
os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "10000")
os.environ.setdefault("GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA", "0")
os.environ.setdefault("GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", "1")
os.environ.setdefault("GRPC_MAX_RECEIVE_MESSAGE_LENGTH", str(200 * 1024 * 1024))
os.environ.setdefault("GRPC_MAX_SEND_MESSAGE_LENGTH",    str(200 * 1024 * 1024))

logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE_DIR = os.path.abspath(os.path.join("Result", "clientresult"))
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


# Small CPU/runtime knobs
def _set_runtime_knobs(num_threads: int = 4) -> None:
    """
    Configure CPU threads to reduce contention on CPU-only boxes.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# XAI helpers (Grad-CAM)
def _normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    amin = a.min()
    a = a - amin
    amax = a.max() + 1e-12
    a = a / amax
    return a

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the last nn.Conv2d layer in the model, or None if not found."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv

def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module | None:
    """Dynamically selects the target layer for Grad-CAM++ based on model type."""
    # ResNet50: Use layer4 block, not the last individual Conv2d.
    # Individual Conv2d inside bottleneck blocks produce flat/zero gradients
    # due to skip connections. layer4 output gives proper gradient flow.
    if model_name == "resnet50":
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
            return model.backbone.layer4
        return find_last_conv_layer(model)

    # Other CNN models
    if model_name in ["customcnn", "densenet121", "mobilenetv3", "LSeTNet"]:
        return find_last_conv_layer(model)

    # ViT (timm model)
    elif model_name in ("vit", "vit_tiny"):
        # Common choices: model.vit.norm or last block's norm1
        if hasattr(model, 'vit') and hasattr(model.vit, 'norm'):
            return model.vit.norm
        elif hasattr(model, 'vit') and hasattr(model.vit, 'blocks') and len(model.vit.blocks) > 0:
            return model.vit.blocks[-1].norm1 # Last block's first norm layer

    # Swin Transformer (timm model)
    elif model_name == "swin_tiny":
        # Common choices: model.swin.norm or last block's norm
        if hasattr(model, 'swin') and hasattr(model.swin, 'norm'):
            return model.swin.norm
        elif hasattr(model, 'swin') and hasattr(model.swin, 'layers') and len(model.swin.layers) > 0:
            return model.swin.layers[-1].blocks[-1].norm # Norm in the last block of the last stage

    # Hybrid models
    elif model_name == "hybridmodel": # HybridViTCNNMLP (ViT + ResNet)
        # Target the last conv layer of the CNN branch (ResNet part)
        if hasattr(model, 'cnn'):
            return find_last_conv_layer(model.cnn)

    elif model_name == "hybridswin": # HybridSwinDenseNetMLP (Swin + DenseNet)
        # Target the last conv layer of the CNN branch (DenseNet part)
        if hasattr(model, 'densenet'):
            # For DenseNet, features.denseblock4.denselayer24.conv2 is a common last conv.
            # A more generic approach is to find the last conv in its features
            return find_last_conv_layer(model.densenet.features)

    logger.warning(f"Could not find a specific Grad-CAM target layer for model: {model_name}. "
                   "Falling back to `find_last_conv_layer` on the whole model.")
    return find_last_conv_layer(model) # Fallback

def compute_gradcam_pp(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_layer: torch.nn.Module,
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
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=False)

        if not activations or not gradients:
            raise RuntimeError("Hooks not triggered - check target_layer")

        A = activations[0][0]  # CNN: [C, H, W] | Transformer: [num_tokens, embed_dim]
        G = gradients[0][0]

        # Handle transformer activations (2D from LayerNorm) → spatial [C, H, W]
        if A.ndim == 2:
            from utils.xai.gradcam_pp import _reshape_transformer_activations
            A, G = _reshape_transformer_activations(A, G)

        # Grad-CAM++ weight calculation
        grads2 = G ** 2
        grads3 = G ** 3
        
        sum_activations = torch.sum(A, dim=(1, 2), keepdim=True)

        eps = 1e-7
        denominator = (2 * grads2 + sum_activations * grads3 + eps)
        alpha = grads2 / denominator
        
        positive_gradients = F.relu(G)
        weights = torch.sum(alpha * positive_gradients, dim=(1, 2))

        cam = torch.sum(weights.view(-1, 1, 1) * A, dim=0)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        return cam_np

    finally:
        handle_f.remove()
        handle_b.remove()
        model.zero_grad(set_to_none=True)


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
    Expected range: 0.1-0.3 (good), 0.5-0.7 (moderate), >0.7 (poor).
    """
    model.eval()

    with torch.no_grad():
        x_mod = x.clone().to(device)
        B, C, H, W = x_mod.shape

        if cam.shape != (H, W):
            cam = cv2.resize(cam, (W, H))

        cam_flat = cam.reshape(-1)
        indices = np.argsort(-cam_flat)

        num_pixels = H * W
        pixels_per_step = max(num_pixels // steps, 1)

        mask = torch.ones((1, 1, H, W), device=device, dtype=x_mod.dtype)
        scores: list[float] = []

        for step_i in range(steps + 1):
            masked_input = x_mod * mask

            logits = model(masked_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            score = float(probs[0, class_idx].item())
            scores.append(score)

            if step_i == steps:
                break

            start_idx = step_i * pixels_per_step
            end_idx = min((step_i + 1) * pixels_per_step, num_pixels)
            pixels_to_delete = indices[start_idx:end_idx]

            for pixel_idx in pixels_to_delete:
                row = int(pixel_idx // W)
                col = int(pixel_idx % W)
                x_mod[0, :, row, col] = -1.0  # assuming input normalized in [-1, 1]

        fractions = np.linspace(0.0, 1.0, len(scores))
        # auc = float(np.trapz(scores, fractions))
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
    Expected range: 0.7-0.9 (good), 0.5-0.7 (moderate), <0.5 (poor).
    """
    model.eval()

    with torch.no_grad():
        x_mod = x.clone().to(device)
        B, C, H, W = x_mod.shape

        if cam.shape != (H, W):
            cam = cv2.resize(cam, (W, H))

        cam_flat = cam.reshape(-1)
        indices = np.argsort(cam_flat) # Sort ascending for insertion

        num_pixels = H * W
        pixels_per_step = max(num_pixels // steps, 1)

        mask = torch.zeros((1, 1, H, W), device=device, dtype=x_mod.dtype) # Start with empty mask
        scores: list[float] = []

        for step_i in range(steps + 1):
            masked_input = x_mod * mask

            logits = model(masked_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            score = float(probs[0, class_idx].item())
            scores.append(score)

            if step_i == steps:
                break

            start_idx = step_i * pixels_per_step
            end_idx = min((step_i + 1) * pixels_per_step, num_pixels)
            pixels_to_insert = indices[start_idx:end_idx]

            for pixel_idx in pixels_to_insert:
                row = int(pixel_idx // W)
                col = int(pixel_idx % W)
                mask[0, :, row, col] = 1.0 # Insert pixels

        fractions = np.linspace(0.0, 1.0, len(scores))
        if hasattr(np, "trapezoid"):
            auc = float(np.trapezoid(scores, fractions)) # trapezoid is alias for trapz in newer numpy
        else:
            auc = float(np.trapz(scores, fractions))

        return auc

def compute_cam_consistency(cam1: np.ndarray, cam2: np.ndarray) -> float:
    """
    Computes cosine similarity between two Grad-CAM heatmaps for localization consistency.
    """
    cam1_flat = cam1.flatten()
    cam2_flat = cam2.flatten()

    dot_product = np.dot(cam1_flat, cam2_flat)
    norm_cam1 = np.linalg.norm(cam1_flat)
    norm_cam2 = np.linalg.norm(cam2_flat)

    if norm_cam1 == 0 or norm_cam2 == 0:
        return 0.0 # Return 0 consistency if either CAM is flat (all zeros)

    return dot_product / (norm_cam1 * norm_cam2)

def save_gradcam_overlay(
    x: torch.Tensor,
    cam: np.ndarray,
    out_dir: str,
    round_num: int,
    idx: int,
    true_label: int,
    pred_label: int,
    auc: float | None = None,
) -> None:
    """Save an overlay of Grad-CAM++ heatmap on top of the input image."""
    _ensure_dir(out_dir)

    img = x[0].detach().cpu()

    if img.shape[0] == 1:
        base = img[0].numpy()
        is_grayscale = True
    else:
        base = img.permute(1, 2, 0).numpy()
        is_grayscale = False

    base = base - base.min()
    if base.max() > 0:
        base = base / (base.max() + 1e-8)

    cam_resized = cv2.resize(cam, (base.shape[1], base.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    if is_grayscale:
        axes[0].imshow(base, cmap="gray")
    else:
        axes[0].imshow(base)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM++")
    axes[1].axis("off")

    if is_grayscale:
        axes[2].imshow(base, cmap="gray")
    else:
        axes[2].imshow(base)
    axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)

    match_symbol = "✓" if pred_label == true_label else "✗"
    title = f"Overlay {match_symbol}\nTrue: {true_label}, Pred: {pred_label}"
    if auc is not None:
        title += f"\nDel-AUC: {auc:.4f}"
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()

    fname = f"round{round_num:03d}_sample{idx:03d}_true{true_label}_pred{pred_label}"
    if auc is not None:
        fname += f"_auc{auc:.3f}"
    fname += ".png"

    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculates the Kullback-Leibler (KL) Divergence, ensuring numerical stability."""
    epsilon = 1e-10 # Small constant to avoid log(0)
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    return np.sum(p * np.log(p / q))

def _flatten_parameters(parameters: list[np.ndarray]) -> np.ndarray:
    """Flattens and concatenates a list of NumPy arrays into a single 1D array."""
    return np.concatenate([p.flatten() for p in parameters])

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates the cosine similarity between two 1D NumPy arrays."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # Handle zero-norm vectors

    return dot_product / (norm_vec1 * norm_vec2)

# Flower Client
class MedicalFLClient(fl.client.NumPyClient):
    """
    Federated Learning client for medical image classification (PyTorch).
    """
    def __init__(
        self,
        client_id: int,
        data_dir: str,
        device: torch.device,
        model_name: str = "customcnn",
        num_classes: int = 3,
        batch_size: int = 16,
        local_epochs: int = 8,
        num_workers: int = 0,  # MUST be 0 on Windows to prevent DataLoader deadlock
        results_base_dir: str = RESULTS_BASE_DIR,
    ):
        self.client_id = client_id
        self.data_dir = data_dir
        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.num_workers = num_workers

        # Per-client folders
        self.results_base_dir = results_base_dir
        os.makedirs(self.results_base_dir, exist_ok=True)
        self.client_root = os.path.join(self.results_base_dir, f"client_{client_id}")
        self.ckpt_dir = os.path.join(self.client_root, "checkpoints")
        self.log_dir = os.path.join(self.client_root, "logs")
        self.xai_dir = os.path.join(self.client_root, "xai")
        self.pred_dir = os.path.join(self.client_root, "predictions")
        self.metrics_dir = os.path.join(self.client_root, "metrics")
        for d in [self.client_root, self.ckpt_dir, self.log_dir, self.xai_dir, self.pred_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

        # Model
        self.model = get_model(model_name, num_classes, pretrained=True)
        self.model.to(device)
        self.model = self.model.float()  # Ensure model is always in float32

        # Choose a conv layer for Grad-CAM (optional)
        self.target_layer = get_gradcam_target_layer(self.model, model_name)
        if self.target_layer is None:
            logger.warning("No Conv2d layer found for Grad-CAM. XAI probe will be skipped.")
        self.shared_xai_loader = None
        self.shared_xai_dir = None
        self.xai_manager = None
        if self.target_layer is not None:
            self.xai_manager = FederatedXAIManager(
                model=self.model,
                target_layer=self.target_layer,
                device=self.device,
                client_id=self.client_id,
                save_dir=self.xai_dir,
            )

        # DataLoaders
        logger.info(f"Client {self.client_id}: Loading data from {data_dir}")

        # Load the full client dataset
        full_client_dataset = CTScanDataset(data_dir, transform=None, subset='full')
        all_samples = full_client_dataset.samples
        all_labels = [s[1] for s in all_samples] # Extract labels for stratification

        # Define split ratios
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        # Split into train and temp (val + test)
        train_samples, temp_samples, train_labels, temp_labels = train_test_split(
            all_samples, all_labels, train_size=train_ratio, stratify=all_labels, random_state=42
        )

        # Split temp into validation and test
        # Calculate val_size relative to temp_samples
        if len(temp_samples) > 0: # Avoid division by zero if temp_samples is empty
            relative_val_ratio = val_ratio / (val_ratio + test_ratio)
            val_samples, test_samples, _, _ = train_test_split(
                temp_samples, temp_labels, train_size=relative_val_ratio, stratify=temp_labels, random_state=42
            )
        else:
            val_samples = []
            test_samples = []

        # Create dataset instances for each split using lightweight copies
        # (avoids 3 redundant directory scans and misleading log messages)
        import copy
        train_dataset = copy.copy(full_client_dataset)
        train_dataset.transform = get_medical_transforms(subset='train')
        train_dataset.subset = 'train_local'
        train_dataset.samples = train_samples

        val_dataset = copy.copy(full_client_dataset)
        val_dataset.transform = get_medical_transforms(subset='val')
        val_dataset.subset = 'val_local'
        val_dataset.samples = val_samples

        test_dataset = copy.copy(full_client_dataset)
        test_dataset.transform = get_medical_transforms(subset='test')
        test_dataset.subset = 'test_local'
        test_dataset.samples = test_samples
        
        # Data validation: Check for overlap between splits
        train_paths = set([str(path) for path, _ in train_samples])
        val_paths = set([str(path) for path, _ in val_samples])
        test_paths = set([str(path) for path, _ in test_samples])
        
        train_val_overlap = train_paths & val_paths
        train_test_overlap = train_paths & test_paths
        val_test_overlap = val_paths & test_paths
        
        if train_val_overlap:
            logger.error(f"Client {self.client_id}: FOUND {len(train_val_overlap)} OVERLAPPING FILES between train and val!")
        if train_test_overlap:
            logger.error(f"Client {self.client_id}: FOUND {len(train_test_overlap)} OVERLAPPING FILES between train and test!")
        if val_test_overlap:
            logger.error(f"Client {self.client_id}: FOUND {len(val_test_overlap)} OVERLAPPING FILES between val and test!")
        
        if not (train_val_overlap or train_test_overlap or val_test_overlap):
            logger.info(f"Client {self.client_id}: ✓ Data splits verified - no overlaps detected")
        
        # Log class distributions for each split
        train_labels_list = [label for _, label in train_samples]
        val_labels_list = [label for _, label in val_samples]
        test_labels_list = [label for _, label in test_samples]
        
        from collections import Counter
        logger.info(f"Client {self.client_id}: Train class distribution: {dict(Counter(train_labels_list))}")
        logger.info(f"Client {self.client_id}: Val class distribution: {dict(Counter(val_labels_list))}")
        logger.info(f"Client {self.client_id}: Test class distribution: {dict(Counter(test_labels_list))}")

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, persistent_workers=False, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, persistent_workers=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, persistent_workers=False
        )

        # Class weights
        self.class_weights = get_class_weights(self.train_loader)
        try:
            cw_log = self.class_weights.tolist()
        except Exception:
            cw_log = self.class_weights
        logger.info(f"Client {self.client_id}: Class weights: {cw_log}")

        # Trainer
        self.trainer = ModelTrainer(self.model, device, self.ckpt_dir, self.log_dir)

        # Defaults (can be overridden by server config)
        self.learning_rate = 0.0003  # Tuned for batch_size=16, 3 IID clients
        self.weight_decay = 1e-4
        self.fedbn = False
        self.communication_bytes_sent = 0
        self.communication_bytes_received = 0
        self.prev_cams = {} # For XAI localization consistency

        # For temporarily disabling dropout during XAI
        self.original_dropout_ps = {}

        logger.info(f"Client {self.client_id} initialized successfully")
        logger.info(f"  - Training samples:   {len(self.train_loader.dataset)}")
        logger.info(f"  - Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"  - Test samples:       {len(self.test_loader.dataset)}")

    def _store_dropout_states(self):
        self.original_dropout_ps = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                self.original_dropout_ps[name] = module.p
                module.p = 0.0 # Temporarily disable dropout

    def _restore_dropout_states(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout) and name in self.original_dropout_ps:
                module.p = self.original_dropout_ps[name]
        self.original_dropout_ps = {} # Clear states

    def _get_shared_probe_loader(self, shared_dir: str) -> DataLoader | None:
        if not shared_dir or not os.path.isdir(shared_dir):
            logger.warning(f"Shared XAI probe dir not found: {shared_dir}")
            return None

        if self.shared_xai_loader is not None and self.shared_xai_dir == shared_dir:
            return self.shared_xai_loader

        dataset = CTScanDataset(shared_dir, transform=get_medical_transforms(subset="val"), subset="xai_shared")
        # DO NOT sort samples — sorting by path puts Benign first always,
        # causing XAI to only explain Benign class images.
        # Shuffle instead so all classes are represented.
        import random as _rand
        _rand.shuffle(dataset.samples)
        self.shared_xai_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle to get diverse class samples
            num_workers=0,
        )
        self.shared_xai_dir = shared_dir
        logger.info(f"Client {self.client_id}: Using shared XAI probe set from {shared_dir} "
                    f"with {len(self.shared_xai_loader.dataset)} samples.")
        return self.shared_xai_loader

    def _recalibrate_batchnorm(self, loader: DataLoader, num_batches: int = 20) -> None:
        """
        Recompute BatchNorm running statistics using local data after loading
        newly aggregated global weights.
        This improves global evaluation stability and reduces single-class collapse.
        """
        bn_layers = [
            m for m in self.model.modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
        ]
        if not bn_layers:
            return

        # Keep dropout and most layers in eval behavior, but allow BN stats update.
        self.model.eval()
        for m in bn_layers:
            m.train()

        nan_batches = 0
        with torch.no_grad():
            seen = 0
            for data, _ in loader:
                data = data.to(self.device).float()
                try:
                    _ = self.model(data)
                except RuntimeError as e:
                    if "Numerical instability" in str(e):
                        nan_batches += 1
                        if nan_batches > 5:
                            logger.warning(f"Client {self.client_id}: Too many NaN batches during BN recalibration, aborting")
                            break
                        continue
                    raise
                seen += 1
                if seen >= num_batches:
                    break

        self.model.eval()
        logger.info(f"Client {self.client_id}: BatchNorm recalibrated using {seen} local batches before evaluation"
                    + (f" ({nan_batches} batches had NaN)" if nan_batches else ""))

    # Flower NumPyClient API ----
    def get_parameters(self) -> list[np.ndarray]:
        """
        Return model parameters to the server for aggregation.
        
        CRITICAL FIX: Transmit float32 to server, compress only for communication tracking.
        This ensures aggregation correctness and prevents precision loss that degrades
        convergence and FedProx behavior.
        
        FEDBN FIX: Exclude BatchNorm running statistics from aggregation.
        BatchNorm statistics are local to each client's data distribution and should
        NOT be aggregated in medical federated learning (heterogeneous data).
        """
        # Get trainable parameters only (excludes BN running stats)
        current_parameters = []
        for name, param in self.model.state_dict().items():
            # Exclude BatchNorm running statistics
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                continue
            # Include all other parameters (weights, biases, BN gamma/beta)
            current_parameters.append(param.detach().cpu().numpy().astype(np.float32))
        
        # Track communication bytes sent (using actual data size)
        self.communication_bytes_sent += sum(p.nbytes for p in current_parameters)

        return current_parameters

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """
        Set model parameters from server aggregation.
        
        FEDBN FIX: Only update parameters that were aggregated (excluding BN running stats).
        Each client maintains its own BatchNorm statistics for local data distribution.
        """
        incoming_ndarrays = parameters

        # Decompress from float16 to float32 if necessary
        incoming_ndarrays_decompressed = []
        for arr in incoming_ndarrays:
            if arr.dtype == np.float16:
                incoming_ndarrays_decompressed.append(arr.astype(np.float32))
            else:
                incoming_ndarrays_decompressed.append(arr)

        # Load parameters back into model, EXCLUDING BN running stats
        params_dict = {}
        param_idx = 0
        for name, param in self.model.state_dict().items():
            # Skip BatchNorm running statistics - they stay client-local
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                params_dict[name] = param  # Keep local BN stats
                # CRITICAL: Do NOT increment param_idx here - no parameter consumed from incoming array
                continue
            # Update aggregated parameters
            if param_idx < len(incoming_ndarrays_decompressed):
                incoming = incoming_ndarrays_decompressed[param_idx]
                if tuple(incoming.shape) != tuple(param.shape):
                    raise RuntimeError(
                        f"Client {self.client_id}: Parameter shape mismatch for '{name}': "
                        f"expected {tuple(param.shape)}, got {tuple(incoming.shape)}"
                    )
                params_dict[name] = torch.tensor(incoming, dtype=param.dtype)
                param_idx += 1

        if param_idx != len(incoming_ndarrays_decompressed):
            raise RuntimeError(
                f"Client {self.client_id}: Consumed {param_idx} parameters but received "
                f"{len(incoming_ndarrays_decompressed)}"
            )
        
        self.model.load_state_dict(params_dict, strict=False)
        # Ensure model is in float32 to match data precision
        self.model = self.model.float()
        
        # Track communication bytes received
        self.communication_bytes_received += sum(arr.nbytes for arr in incoming_ndarrays_decompressed)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
        logger.info(f"Client {self.client_id}: Starting local training round")

        # Proactively clear GPU memory before each round to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            import gc; gc.collect()

        # Reset communication counters for the current round
        self.communication_bytes_sent = 0
        self.communication_bytes_received = 0

        # Store initial global parameters for parameter cosine similarity
        initial_global_parameters_flattened = _flatten_parameters(parameters)

        self.set_parameters(parameters)

        # Server-configurable knobs
        local_epochs   = int(config.get("local_epochs", self.local_epochs))
        learning_rate  = float(config.get("learning_rate", self.learning_rate))
        weight_decay   = float(config.get("weight_decay", self.weight_decay))
        loss_function  = str(config.get("loss_function", "crossentropy")).lower()
        optimizer_name = str(config.get("optimizer", "adamw")).lower()
        scheduler_name = str(config.get("scheduler", "plateau")).lower()
        use_scheduler  = bool(config.get("use_scheduler", True))
        personalization_layer = bool(config.get("personalization_layer", False))
        self.fedbn = bool(config.get("fedbn", False)) # Update fedbn status from config
        global_class_distribution_raw = config.get("global_class_distribution", None)
        # Deserialize from JSON string (Flower config only supports Scalar types)
        if isinstance(global_class_distribution_raw, str):
            import json
            global_class_distribution = json.loads(global_class_distribution_raw)
        else:
            global_class_distribution = global_class_distribution_raw

        if self.fedbn and hasattr(self.model, 'freeze_bn'):
            self.model.freeze_bn()
        elif hasattr(self.model, 'unfreeze_bn'):
            # Ensure BN layers are unfrozen if FedBN is not active for this round
            self.model.unfreeze_bn()

        # Personalization layer: Freeze backbone if enabled before local training
        if personalization_layer:
            logger.info("Client personalization: Freezing backbone layers.")
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "head" not in name: # Commonly used names for classification layers
                    param.requires_grad = False

        # FedProx parameters
        aggregation = config.get("aggregation", "fedavg")
        mu = float(config.get("mu", 0.0))

        # Gradient accumulation — auto-set for large models on limited VRAM
        accumulation_steps = int(config.get("accumulation_steps", 1))
        if accumulation_steps <= 1:
            # Auto-detect: use accumulation for transformer models with small batch
            model_name = getattr(self, 'model_name', '').lower()
            is_large_model = any(k in model_name for k in ['vit', 'hybrid', 'swin'])
            if is_large_model and self.batch_size <= 4:
                accumulation_steps = max(1, 8 // self.batch_size)
                logger.info(f"Auto gradient accumulation: steps={accumulation_steps}, "
                           f"effective_batch={self.batch_size * accumulation_steps}")

        global_params = None
        if aggregation == "fedprox":
            # Store a copy of the global model's parameters for FedProx
            global_params = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
            }

        # Loss
        if loss_function == "focal":
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss_function == "label_smoothing":
            criterion = LabelSmoothingLoss(num_classes=self.num_classes, smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        # Train
        train_history = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=self.class_weights,
            use_scheduler=use_scheduler,
            patience=10,
            criterion=criterion,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            global_model_weights=global_params,
            mu=mu,
            accumulation_steps=accumulation_steps,
        )

        # Unfreeze all parameters after local training for consistency
        if personalization_layer:
            for param in self.model.parameters():
                param.requires_grad = True

        rnd = config.get("server_round", 0)

        # Evaluate on test
        # test_metrics = self.trainer.evaluate(self.test_loader)
        test_metrics = self.trainer.evaluate(
            self.test_loader,
            save_name=f"round_{rnd}_local_trained_matrix.png"
        )

        # Save per-round metrics and predictions
        try:
            round_metrics_path = os.path.join(self.metrics_dir, f"round_{rnd}_metrics.json")
            serializable_metrics = {
                k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
                for k, v in test_metrics.items()
                if not isinstance(v, (np.ndarray, list))
            }
            with open(round_metrics_path, "w") as mf:
                json.dump(serializable_metrics, mf, indent=2)

            round_pred_path = os.path.join(self.pred_dir, f"round_{rnd}_predictions.json")
            pred_data = {}
            if "predictions" in test_metrics:
                pred_data["predictions"] = [int(p) for p in test_metrics["predictions"]]
            if "labels" in test_metrics:
                pred_data["labels"] = [int(l) for l in test_metrics["labels"]]
            if pred_data:
                with open(round_pred_path, "w") as pf:
                    json.dump(pred_data, pf, indent=2)
        except Exception as e:
            logger.warning(f"Client {self.client_id}: Failed to save per-round metrics/predictions: {e}")

        # XAI probe (optional)
        xai_metrics = self._xai_probe(
            self.val_loader,
            config=config,
            num_samples=16,
            save_k=3,
            epoch=rnd,
        ) if self.target_layer else self._empty_xai_metrics()

        # Save checkpoint
        best_model_path = os.path.join(self.ckpt_dir, f"client_{self.client_id}_best_model.pth")
        torch.save(self.model.state_dict(), best_model_path)
        logger.info(f"Client {self.client_id}: Best model saved to {best_model_path}")

        # Scalar metrics only (keep payload small)
        metrics = {
            "train_loss": float(train_history["train_loss"][-1]),
            "train_accuracy": float(train_history["train_accuracy"][-1]),
            "train_f1": float(train_history.get("train_f1_macro", [0.0])[-1]),
            "val_loss": float(train_history["val_loss"][-1]),
            "val_accuracy": float(train_history["val_accuracy"][-1]),
            "val_f1": float(train_history["val_f1_macro"][-1]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1_macro"]),
            "num_examples": int(len(self.train_loader.dataset)),
            "grad_norm": float(train_history.get("grad_norm", [0.0])[-1]),
            "max_activation": float(train_history.get("max_activation", [0.0])[-1]),
        }
        
        # Add XAI metrics (ensure all values are native Python types)
        for key, value in xai_metrics.items():
            if isinstance(value, (int, float, str, bytes, bool)):
                metrics[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
            else:
                metrics[key] = str(value)

        # Add personalized metrics if personalization layer is active
        if personalization_layer:
            metrics["personalized_test_accuracy"] = float(test_metrics["accuracy"])
            metrics["personalized_test_f1"] = float(test_metrics["f1_macro"])
            logger.info(f"Client {self.client_id}: Personalized Test Acc: {metrics['personalized_test_accuracy']:.4f}, F1: {metrics['personalized_test_f1']:.4f}")

        # --- KL Divergence for Data Heterogeneity ---
        if global_class_distribution is not None:
            # Calculate local class distribution (use .samples to avoid loading images)
            local_labels = [s[1] for s in self.train_loader.dataset.samples]
            local_class_counts = np.bincount(local_labels, minlength=self.num_classes)
            local_class_probs = local_class_counts / local_class_counts.sum()

            # Ensure global_class_distribution is a numpy array
            global_class_probs = np.asarray(global_class_distribution, dtype=np.float64)

            # Compute KL divergence
            kl_div = _kl_divergence(local_class_probs, global_class_probs)
            metrics["kl_divergence"] = float(kl_div)  # Convert to native Python float
            logger.info(f"Client {self.client_id}: KL Divergence = {kl_div:.4f}")
        else:
            logger.warning(f"Client {self.client_id}: global_class_distribution not provided in config. Skipping KL divergence calculation.")
        # --- End KL Divergence ---

        # Client drift monitoring
        if global_params is not None:
            client_drift = 0.0
            layer_drift = {}
            for name, param in self.model.named_parameters():
                if name in global_params:
                    # Individual layer drift
                    ld = float(torch.norm(param - global_params[name]).item())
                    layer_drift[name] = ld
                    # Total client drift
                    client_drift += ld
            metrics["client_drift"] = float(client_drift)
            
            # Don't include layer_drift dict in metrics (too large, causes serialization issues)
            # metrics["layer_drift"] = layer_drift  # REMOVED

            # Separate CNN and Transformer drift
            cnn_drift_total = 0.0
            transformer_drift_total = 0.0
            for name, ld_value in layer_drift.items():
                if name.startswith(("cnn.", "densenet.", "backbone.", "features.")):
                    cnn_drift_total += ld_value
                elif name.startswith(("vit.", "swin.", "encoder.", "transformer.")):
                    transformer_drift_total += ld_value
            metrics["cnn_drift"] = float(cnn_drift_total)
            metrics["transformer_drift"] = float(transformer_drift_total)

        # Compute outgoing parameters BEFORE reporting metrics
        # (get_parameters() increments communication_bytes_sent internally)
        outgoing_params = self.get_parameters()
        
        metrics["communication_bytes_sent"] = int(self.communication_bytes_sent)
        metrics["communication_bytes_received"] = int(self.communication_bytes_received)

        # Parameter Cosine Similarity between current local model and initial global model
        # FEDBN FIX: Exclude BN stats when computing cosine similarity
        # Must match what get_parameters() returns (trainable params only)
        current_local_parameters = []
        for name, param in self.model.state_dict().items():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                continue
            current_local_parameters.append(param.detach().cpu().numpy())
        
        current_local_parameters_flattened = _flatten_parameters(current_local_parameters)

        parameter_cosine_similarity = _cosine_similarity(
            initial_global_parameters_flattened, current_local_parameters_flattened
        )
        metrics["parameter_cosine_similarity"] = float(parameter_cosine_similarity)
        logger.info(f"Client {self.client_id}: Parameter Cosine Similarity = {parameter_cosine_similarity:.4f}")

        logger.info(f"Client {self.client_id}: Local training completed")
        
        # Validate weights before sending
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                logger.critical(f"Client {self.client_id}: Parameter {name} contains NaN/Inf before sending to server.")
                raise RuntimeError(f"Numerical instability in {name}")

        # CRITICAL: Synchronize CUDA and clear cache to prevent GPU lock between rounds
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return outgoing_params, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
        logger.info(f"Client {self.client_id}: Starting evaluation")
        
        # Clear GPU cache before evaluation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        self.set_parameters(parameters)

        # Stabilize global-model evaluation by aligning BN running stats to local data.
        # This mitigates one-class collapse right after parameter aggregation.
        try:
            self._recalibrate_batchnorm(self.train_loader, num_batches=20)
        except Exception as e:
            logger.warning(f"Client {self.client_id}: BN recalibration skipped due to: {type(e).__name__}: {e}")

        rnd = config.get("server_round", 0)

        #test_metrics = self.trainer.evaluate(self.test_loader)
        test_metrics = self.trainer.evaluate(
            self.test_loader,
            save_name=f"round_{rnd}_global_weights_matrix.png"
        )

        add_xai = self._xai_probe(
            self.val_loader,
            config=config,
            num_samples=12,
            save_k=0,
            epoch=rnd,
        ) if self.target_layer else self._empty_xai_metrics()
        test_metrics.update(add_xai)
        test_metrics["communication_bytes_sent"] = self.communication_bytes_sent
        test_metrics["communication_bytes_received"] = self.communication_bytes_received

        logger.info(f"Client {self.client_id}: Evaluation completed")
        logger.info(f"  - Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  - Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"  - XAI Del-AUC Mean: {test_metrics['xai_del_auc_mean']:.4f}")

        # ── GPU cleanup after evaluation ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Filter metrics for Flower: only scalars and strings (no lists/arrays)
        flower_metrics = {}
        for k, v in test_metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                flower_metrics[k] = float(v)
            elif isinstance(v, (int, float)):
                flower_metrics[k] = float(v)
            elif isinstance(v, str):
                flower_metrics[k] = v
            elif isinstance(v, bytes):
                flower_metrics[k] = v
            # Skip lists, arrays, dicts — Flower ConfigRecord can't serialize them

        return (
            float(test_metrics.get("loss", 0.0)),
            int(len(self.test_loader.dataset)),
            flower_metrics,
        )


    def _empty_xai_metrics(self) -> dict[str, float | str]:
        return {
            "xai_del_auc_mean": float("nan"),
            "xai_del_auc_std": float("nan"),
            "xai_ins_auc_mean": float("nan"),
            "xai_ins_auc_std": float("nan"),
            "xai_cam_consistency_mean": float("nan"),
            "xai_cam_consistency_std": float("nan"),
            "xai_temporal_stability_mean": float("nan"),
            "xai_temporal_stability_std": float("nan"),
            "xai_temporal_pearson_mean": float("nan"),
            "xai_temporal_pearson_std": float("nan"),
            "xai_samples": 0.0,
            "xai_cam_stack_b64": "",
        }

    def _xai_probe(
        self,
        loader: DataLoader,
        config: dict | None = None,
        num_samples: int = 16,
        save_k: int = 3,
        epoch: int = 0,
    ) -> dict[str, float | str]:
        config = config or {}

        if self.xai_manager is None:
            logger.warning("XAI manager not initialized, skipping XAI probe.")
            return self._empty_xai_metrics()

        if not bool(config.get("xai_probe", True)):
            return self._empty_xai_metrics()

        shared_dir = config.get("xai_shared_probe_dir", None)
        if shared_dir:
            shared_loader = self._get_shared_probe_loader(shared_dir)
            if shared_loader is not None:
                loader = shared_loader

        if loader is None:
            logger.warning("No loader available for XAI probe.")
            return self._empty_xai_metrics()

        # CRITICAL FIX: Keep model in train mode for GradCAM to work
        # GradCAM needs gradient flow, which is disabled in eval mode
        self.model.train()  # Changed from eval() to train()

        # Only put BatchNorm in eval mode to use running stats (not training mode stats)
        # LayerNorm has no running stats — just set eval mode for consistency
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                m.track_running_stats = True
            elif isinstance(m, nn.LayerNorm):
                m.eval()

        self._store_dropout_states()

        try:
            num_samples = int(config.get("xai_samples", num_samples))
            save_k = int(config.get("xai_save_k", save_k))
            cam_downsample = int(config.get("xai_cam_downsample", 32))

            run_heavy = bool(config.get("xai_run_heavy", False))
            run_shap = bool(config.get("xai_run_shap", True))
            run_lime = bool(config.get("xai_run_lime", True))
            run_attention = bool(config.get("xai_run_attention", True))

            result = self.xai_manager.run_probe(
                loader=loader,
                round_num=epoch,
                num_samples=num_samples,
                save_k=save_k,
                cam_downsample=cam_downsample,
                run_heavy=run_heavy,
                run_shap=run_shap,
                run_lime=run_lime,
                run_attention=run_attention,
                shared_probe=bool(shared_dir),
            )
            return result
        except Exception as e:
            logger.error(f"Client {self.client_id}: XAI probe failed: {type(e).__name__}: {e}")
            return self._empty_xai_metrics()
        finally:
            self._restore_dropout_states()
            # CRITICAL: Clear CUDA graph and cache after XAI to prevent GPU lock
            self.model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


def create_client(client_id: int, data_dir: str, model_name: str = "customcnn",
                  batch_size: int = 16, local_epochs: int = 50, num_workers: int = 0) -> MedicalFLClient:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    client = MedicalFLClient(
        client_id=client_id,
        data_dir=data_dir,
        device=device,
        model_name=model_name,
        num_classes=3,
        batch_size=batch_size,
        local_epochs=local_epochs,
        num_workers=num_workers,
    )
    return client


def run_flower(server_address: str, client: MedicalFLClient) -> None:
    """Start Flower with simple auto-reconnect on transient UNAVAILABLE."""
    while True:
        try:
            fl.client.start_client(
                server_address=server_address,
                client=client.to_client(),
            )
            break  # finished cleanly
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.warning("Server UNAVAILABLE (gRPC 14). Reconnecting in 5s...")
                time.sleep(5)
                continue
            else:
                raise


def main():
    # Use spawn to avoid forking a multi-threaded process (Flower/gRPC)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set in this process
        pass

    _set_runtime_knobs(num_threads=4)

    # Set deterministic seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Deterministic at the cost of some speed

    # SECTION 12 — RTX 4060 OPTIMIZATION
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="Federated Learning Client for Medical Imaging")
    parser.add_argument("--client-id", type=int, default=1, help="Client ID")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to client data directory")
    parser.add_argument("--server-address", type=str, default="localhost:5050", help="FL server address")
    parser.add_argument("--model", type=str, default="resnet50", # Changed default to resnet50
                        choices=["resnet50", "hybridmodel", "mobilenetv3", "hybridswin", "densenet121", "LSeTNet", "swin_tiny", "vit", "vit_tiny"],
                        help="Model architecture")
    parser.add_argument("--train-local", action="store_true", help="Run local training only (no FL server)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (CPU-only: 32 for ResNet/DenseNet, 64 for small CNN/EfficientNetB0)")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local epochs per round")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (set 0 if problems)")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")

    client = create_client(
        client_id=args.client_id,
        data_dir=args.data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        num_workers=args.num_workers,
    )

    if args.train_local:
        logger.info("Running standalone local training (no FL server)")
        updated_params, num_examples, train_metrics = client.fit(client.get_parameters(), config={})
        test_loss, test_examples, test_metrics = client.evaluate(updated_params, config={})
        logger.info("Local training and evaluation completed:")
        logger.info(f"  - Final train metrics: {train_metrics}")
        logger.info(f"  - Final test metrics:  {test_metrics}")
        return

    logger.info(f"Starting FL client {args.client_id} connecting to {args.server_address}")
    run_flower(args.server_address, client)

if __name__ == "__main__":
    main()
