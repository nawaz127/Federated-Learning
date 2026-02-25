"""
Comprehensive XAI Visualization Module
======================================
Generates publication-quality multi-panel XAI figures for each method:
  Panel 1: Original Test Image
  Panel 2: XAI Heatmap
  Panel 3: Overlay (Original + Heatmap)
  Panel 4: Prediction info (True vs Predicted)

Supports: Grad-CAM++, LIME, SHAP, Attention Rollout
"""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from utils.xai.xai_visualization import denormalize_imagenet, overlay_gradcam
from utils.xai.xai_metrics import _resize_cam

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Benign", "Malignant", "Normal"]


def _get_class_name(idx: int) -> str:
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"Class {idx}"


def _save_single_method_panel(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    method_name: str,
    true_label: int,
    pred_label: int,
    confidence: float | None,
    save_path: Path,
    extra_panel: np.ndarray | None = None,
    extra_title: str = "",
):
    """
    Save a 4-panel figure:
      [Original] [Heatmap] [Overlay] [Prediction Info]
    """
    # Resize heatmap to match original
    h, w = original_img.shape[:2]
    heatmap_resized = _resize_cam(heatmap, w, h)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Create overlay
    overlay = overlay_gradcam(original_img.copy(), heatmap_resized, alpha=0.45)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Build figure
    n_panels = 4 if extra_panel is None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Suptitle
    true_name = _get_class_name(true_label)
    pred_name = _get_class_name(pred_label)
    correct = true_label == pred_label
    status = "CORRECT" if correct else "WRONG"
    status_color = "#2ecc71" if correct else "#e74c3c"

    fig.suptitle(
        f"{method_name} Explanation",
        fontsize=16, fontweight="bold", y=1.02,
    )

    # Panel 1: Original
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Heatmap
    im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"{method_name} Heatmap", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Prediction Info
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].axis("off")
    axes[3].set_facecolor("#f8f9fa")

    info_texts = [
        ("True Label:", true_name, "#2c3e50"),
        ("Predicted:", pred_name, "#2c3e50"),
        ("Status:", status, status_color),
    ]
    if confidence is not None:
        info_texts.append(("Confidence:", f"{confidence:.1%}", "#2c3e50"))

    y_pos = 0.85
    for label, value, color in info_texts:
        axes[3].text(0.5, y_pos, label, ha="center", va="center",
                     fontsize=12, fontweight="bold", color="#7f8c8d",
                     transform=axes[3].transAxes)
        y_pos -= 0.12
        axes[3].text(0.5, y_pos, value, ha="center", va="center",
                     fontsize=16, fontweight="bold", color=color,
                     transform=axes[3].transAxes)
        y_pos -= 0.16

    axes[3].set_title("Prediction", fontsize=13, fontweight="bold")

    # Optional 5th panel (e.g. LIME boundary)
    if extra_panel is not None and n_panels == 5:
        axes[4].imshow(extra_panel)
        axes[4].set_title(extra_title, fontsize=13, fontweight="bold")
        axes[4].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {method_name} visualization → {save_path}")


def _save_combined_panel(
    original_img: np.ndarray,
    method_results: dict,
    true_label: int,
    pred_label: int,
    confidence: float | None,
    save_path: Path,
    sample_idx: int = 0,
):
    """
    Save a combined figure with ALL methods side by side.
    Layout:
      Row 0: Original  | GradCAM Heatmap  | GradCAM Overlay
      Row 1: LIME       | SHAP             | Attention Rollout
    Or a single row per method with columns: [Method Heatmap] [Method Overlay]
    """
    methods = list(method_results.keys())
    n_methods = len(methods)
    if n_methods == 0:
        return

    h, w = original_img.shape[:2]
    true_name = _get_class_name(true_label)
    pred_name = _get_class_name(pred_label)
    correct = true_label == pred_label

    # Layout: n_methods rows x 3 cols (Original/Heatmap/Overlay) + 1 col for info
    fig = plt.figure(figsize=(20, 5 * n_methods))
    gs = gridspec.GridSpec(n_methods, 4, width_ratios=[1, 1, 1, 0.6], wspace=0.15, hspace=0.3)

    for row, method_name in enumerate(methods):
        heatmap = method_results[method_name]
        heatmap_resized = _resize_cam(heatmap, w, h)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
        overlay = overlay_gradcam(original_img.copy(), heatmap_resized, alpha=0.45)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Col 0: Original
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(original_img)
        ax0.set_title("Original" if row == 0 else "", fontsize=12)
        ax0.set_ylabel(method_name, fontsize=14, fontweight="bold", rotation=90, labelpad=15)
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Col 1: Heatmap
        ax1 = fig.add_subplot(gs[row, 1])
        im = ax1.imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
        ax1.set_title("Heatmap" if row == 0 else "", fontsize=12)
        ax1.axis("off")

        # Col 2: Overlay
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(overlay_rgb)
        ax2.set_title("Overlay" if row == 0 else "", fontsize=12)
        ax2.axis("off")

        # Col 3: Info (only first row)
        if row == 0:
            ax3 = fig.add_subplot(gs[:, 3])
            ax3.axis("off")
            ax3.set_facecolor("#f8f9fa")

            status = "CORRECT" if correct else "WRONG"
            status_color = "#2ecc71" if correct else "#e74c3c"

            info_text = f"True: {true_name}\nPred: {pred_name}\n{status}"
            if confidence is not None:
                info_text += f"\nConf: {confidence:.1%}"

            ax3.text(0.5, 0.5, info_text, ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     transform=ax3.transAxes,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bdc3c7"))

    fig.suptitle(
        f"XAI Explanations — Sample {sample_idx}",
        fontsize=18, fontweight="bold", y=1.01,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved combined XAI panel → {save_path}")


def run_comprehensive_xai(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    loader,
    device: torch.device,
    round_num: int,
    save_dir: str | Path,
    client_id: int = 0,
    num_samples: int = 1,  # Minimum sample for XAI (SHAP, GradCAM, LIME)
    preprocess_fn=None,
    run_gradcam: bool = True,
    run_lime: bool = True,
    run_shap: bool = True,
    run_attention: bool = True,
) -> dict:
    """
    Run all XAI methods on test samples and save comprehensive visualizations.

    For each sample, saves:
      - Individual per-method panels (gradcam_sample_X.png, lime_sample_X.png, etc.)
      - Combined all-methods panel (combined_xai_sample_X.png)

    Returns dict with XAI metrics.
    """
    import torch
    from utils.xai.gradcam_pp import compute_gradcam_pp
    from utils.xai.lime_explainer import compute_lime_explanation
    from utils.xai.shap_explainer import compute_shap_explanation
    from utils.xai.attention_rollout import AttentionRollout

    save_dir = Path(save_dir) / f"round_{round_num}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Collect samples — stratified random sampling from ALL classes
    # This ensures XAI visualizations cover Benign, Malignant, AND Normal
    # Cap collection to avoid loading entire dataset into GPU memory
    import random as _rand
    max_collect = num_samples * 3  # Collect 3x what we need for class coverage
    all_tensors_by_class: dict[int, list[tuple[torch.Tensor, int]]] = {}
    total_collected = 0
    for images, labels in loader:
        images = images.to(device)
        for i in range(images.size(0)):
            if total_collected >= max_collect:
                break
            lbl = int(labels[i].item())
            all_tensors_by_class.setdefault(lbl, []).append(
                (images[i:i+1].detach().cpu(), lbl)  # CPU to save GPU memory
            )
            total_collected += 1
        if total_collected >= max_collect:
            break

    # Build a balanced list: round-robin across classes, then random fill
    sample_tensors = []
    sample_labels = []
    available_classes = sorted(all_tensors_by_class.keys())
    if available_classes:
        # Shuffle within each class for randomness
        for cls in available_classes:
            _rand.shuffle(all_tensors_by_class[cls])
        # Round-robin pick
        per_class = max(1, num_samples // len(available_classes))
        remainder = num_samples - per_class * len(available_classes)
        for cls in available_classes:
            picked = all_tensors_by_class[cls][:per_class]
            for t, l in picked:
                sample_tensors.append(t)
                sample_labels.append(l)
        # Fill remainder from random classes
        leftover = []
        for cls in available_classes:
            leftover.extend(all_tensors_by_class[cls][per_class:])
        _rand.shuffle(leftover)
        for t, l in leftover[:max(0, remainder)]:
            sample_tensors.append(t)
            sample_labels.append(l)
    # Final shuffle so classes are interleaved
    combined = list(zip(sample_tensors, sample_labels))
    _rand.shuffle(combined)
    sample_tensors = [c[0] for c in combined]
    sample_labels = [c[1] for c in combined]
    sample_tensors = sample_tensors[:num_samples]
    sample_labels = sample_labels[:num_samples]
    logger.info(f"Client {client_id}: XAI stratified sampling — "
                f"classes represented: {sorted(set(sample_labels))}, "
                f"total samples: {len(sample_tensors)}")

    if not sample_tensors:
        logger.warning(f"Client {client_id}: No samples for XAI visualization")
        return {}

    # Collect background data for SHAP (limit to 8 to reduce memory/compute)
    bg_tensors = []
    if run_shap:
        for images, _ in loader:
            bg_tensors.append(images.to(device))
            if sum(t.size(0) for t in bg_tensors) >= 8:
                break
        bg_data = torch.cat(bg_tensors, dim=0)[:8] if bg_tensors else None

    all_metrics = {
        "xai_methods_run": [],
        "xai_samples_processed": 0,
    }

    # Sequential processing: process one sample at a time, clear GPU cache after each
    for idx, (x, y_true) in enumerate(zip(sample_tensors, sample_labels)):
        x = x.to(device)

        # Get prediction
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            confidence = float(probs.max().item())
            pred_label = int(torch.argmax(probs, dim=1).item())

        # Denormalize for visualization
        original_img = denormalize_imagenet(x[0])  # HWC, float32 [0,1]

        method_heatmaps = {}

        # ─── Grad-CAM++ ───
        if run_gradcam:
            try:
                x_cam = x.clone().requires_grad_(True)
                cam = compute_gradcam_pp(model, x_cam, target_layer, class_idx=pred_label)
                method_heatmaps["Grad-CAM++"] = cam

                _save_single_method_panel(
                    original_img=original_img,
                    heatmap=cam,
                    method_name="Grad-CAM++",
                    true_label=y_true,
                    pred_label=pred_label,
                    confidence=confidence,
                    save_path=save_dir / f"gradcam_sample_{idx}.png",
                )
                if "Grad-CAM++" not in all_metrics["xai_methods_run"]:
                    all_metrics["xai_methods_run"].append("Grad-CAM++")
            except Exception as e:
                logger.error(f"Client {client_id}: Grad-CAM++ failed for sample {idx}: {e}")

        # ─── LIME ───
        if run_lime:
            try:
                image_for_lime = original_img.copy()
                explanation = compute_lime_explanation(
                    model=model,
                    image=image_for_lime,
                    device=device,
                    preprocess_fn=preprocess_fn,
                    num_samples=1000,
                    top_labels=3,
                )

                # Get the positive-only mask for heatmap
                top_label = explanation.top_labels[0]
                temp, mask = explanation.get_image_and_mask(
                    top_label, positive_only=True, num_features=10, hide_rest=False
                )

                # Create a proper heatmap from LIME's local model weights
                # Get the explanation as a dict of {segment_id: weight}
                local_exp = explanation.local_exp[top_label]
                segments = explanation.segments

                # Build a continuous heatmap from segment weights
                lime_heatmap = np.zeros(segments.shape, dtype=np.float32)
                for seg_id, weight in local_exp:
                    lime_heatmap[segments == seg_id] = weight

                # Normalize: positive contributions as heatmap
                lime_heatmap_pos = np.maximum(lime_heatmap, 0)
                if lime_heatmap_pos.max() > 0:
                    lime_heatmap_pos = lime_heatmap_pos / lime_heatmap_pos.max()

                method_heatmaps["LIME"] = lime_heatmap_pos

                # Also create a boundary visualization for extra panel
                from skimage.segmentation import mark_boundaries
                boundary_img = mark_boundaries(
                    temp / 255.0 if temp.max() > 1 else temp, mask
                )

                _save_single_method_panel(
                    original_img=original_img,
                    heatmap=lime_heatmap_pos,
                    method_name="LIME",
                    true_label=y_true,
                    pred_label=pred_label,
                    confidence=confidence,
                    save_path=save_dir / f"lime_sample_{idx}.png",
                    extra_panel=boundary_img,
                    extra_title="LIME Boundaries",
                )
                if "LIME" not in all_metrics["xai_methods_run"]:
                    all_metrics["xai_methods_run"].append("LIME")
            except Exception as e:
                logger.error(f"Client {client_id}: LIME failed for sample {idx}: {e}")

        # ─── SHAP ───
        if run_shap and bg_data is not None:
            try:
                # Guard 1: Skip SHAP if GPU memory is low (< 1.5GB free)
                # Instead, force SHAP to run on CPU to avoid OOM
                import threading
                shap_result = [None]
                shap_error = [None]

                def _run_shap():
                    try:
                        cpu_model = model.cpu()
                        cpu_bg = bg_data[:4].cpu()
                        cpu_x = x.cpu()
                        shap_result[0] = compute_shap_explanation(cpu_model, cpu_bg, cpu_x)
                    except Exception as e:
                        shap_error[0] = e

                shap_thread = threading.Thread(target=_run_shap)
                shap_thread.start()
                shap_thread.join(timeout=120)  # 2-minute timeout

                if shap_thread.is_alive():
                    logger.warning(f"Client {client_id}: SHAP timed out after 120s for sample {idx}")
                    # Thread will eventually finish, but we move on
                    raise TimeoutError("SHAP computation timed out")

                if shap_error[0] is not None:
                    raise shap_error[0]

                shap_values = shap_result[0]
            finally:
                import torch
                torch.cuda.empty_cache()

            # Extract attribution map for predicted class
            if 'shap_values' in locals():
                if isinstance(shap_values, list):
                    sv = shap_values[pred_label]
                else:
                    sv = shap_values[pred_label]

                # sv shape: [1, C, H, W] — sum absolute values over channels
                shap_heatmap = np.sum(np.abs(sv[0]), axis=0)
                shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min() + 1e-8)

                method_heatmaps["SHAP"] = shap_heatmap

                _save_single_method_panel(
                    original_img=original_img,
                    heatmap=shap_heatmap,
                    method_name="SHAP",
                    true_label=y_true,
                    pred_label=pred_label,
                    confidence=confidence,
                    save_path=save_dir / f"shap_sample_{idx}.png",
                )
                if "SHAP" not in all_metrics["xai_methods_run"]:
                    all_metrics["xai_methods_run"].append("SHAP")
            # except block removed: already handled above

        # ─── Attention Rollout (Transformer models only) ───
        if run_attention:
            # Check if model supports attention rollout (transformers only)
            model_class_name = type(model).__name__.lower()
            is_transformer = any(kw in model_class_name for kw in [
                'vit', 'swin', 'transformer', 'hybrid', 'vitresnet', 'swintdensenet',
                'lsetnet',
            ])
            # Also check if model has 'return_attention' support via forward signature
            if not is_transformer:
                import inspect
                try:
                    sig = inspect.signature(model.forward)
                    is_transformer = 'return_attention' in sig.parameters
                except (ValueError, TypeError):
                    pass
            
            if not is_transformer:
                logger.info(f"Client {client_id}: Skipping Attention Rollout — "
                           f"{type(model).__name__} is a CNN (no attention mechanism)")
            else:
                try:
                    rollout_engine = AttentionRollout(model)
                    attention_map = rollout_engine(x, image_size=x.shape[-1])

                    if attention_map is not None:
                        if isinstance(attention_map, torch.Tensor):
                            att_heatmap = attention_map.detach().cpu().numpy()
                        else:
                            att_heatmap = attention_map

                        method_heatmaps["Attention Rollout"] = att_heatmap

                        _save_single_method_panel(
                            original_img=original_img,
                            heatmap=att_heatmap,
                            method_name="Attention Rollout",
                            true_label=y_true,
                            pred_label=pred_label,
                            confidence=confidence,
                            save_path=save_dir / f"attention_rollout_sample_{idx}.png",
                        )
                        if "Attention Rollout" not in all_metrics["xai_methods_run"]:
                            all_metrics["xai_methods_run"].append("Attention Rollout")
                    else:
                        logger.warning(f"Client {client_id}: Attention Rollout returned None for sample {idx} "
                                       f"(model may not support return_attention)")
                except Exception as e:
                    logger.warning(f"Client {client_id}: Attention Rollout failed for sample {idx}: {e}")

        # ─── Combined panel (all methods for this sample) ───
        if method_heatmaps:
            try:
                _save_combined_panel(
                    original_img=original_img,
                    method_results=method_heatmaps,
                    true_label=y_true,
                    pred_label=pred_label,
                    confidence=confidence,
                    save_path=save_dir / f"combined_xai_sample_{idx}.png",
                    sample_idx=idx,
                )
            except Exception as e:
                logger.error(f"Client {client_id}: Failed to save combined panel for sample {idx}: {e}")

        all_metrics["xai_samples_processed"] = idx + 1

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_metrics["xai_methods_run"] = ", ".join(all_metrics["xai_methods_run"])
    logger.info(f"Client {client_id}: Comprehensive XAI complete — "
                f"{all_metrics['xai_samples_processed']} samples, "
                f"methods: {all_metrics['xai_methods_run']}")

    return all_metrics
