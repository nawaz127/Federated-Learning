from pathlib import Path
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger(__name__)

from utils.xai.gradcam_pp import compute_gradcam_pp
from utils.xai.xai_metrics import (
    compute_cam_similarity,
    compute_deletion_auc,
    compute_insertion_auc,
    encode_cam_stack,
    compute_cross_method_agreement,
)
from utils.xai.xai_visualization import denormalize_imagenet, save_gradcam_panel, overlay_gradcam
from utils.xai.shap_explainer import compute_shap_explanation
from utils.xai.lime_explainer import compute_lime_explanation
from utils.xai.attention_rollout import AttentionRollout
from utils.xai.comprehensive_xai_viz import run_comprehensive_xai


class FederatedXAIManager:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        device: torch.device,
        client_id: int,
        save_dir: str,
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.client_id = client_id

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.prev_cams: dict[int, np.ndarray] = {}

    def _fallback_cam_from_input(self, x: torch.Tensor) -> np.ndarray:
        """Deterministic fallback CAM from input intensity when GradCAM fails."""
        # x shape: [1, C, H, W]
        cam = x.detach().float().abs().mean(dim=1).squeeze(0).cpu().numpy()
        cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        cam = cam - float(cam.min())
        denom = float(cam.max())
        if denom > 1e-8:
            cam = cam / denom
        else:
            cam = np.full_like(cam, 0.5, dtype=np.float32)
        return cam

    def _preprocess_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if t.max() > 1.5:
            t = t / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
        return (t - mean) / std

    def run_probe(
        self,
        loader,
        round_num: int,
        num_samples: int = 16,
        save_k: int = 0,
        cam_downsample: int = 32,
        run_heavy: bool = False,
        run_shap: bool = True,
        run_lime: bool = True,
        run_attention: bool = True,
        shared_probe: bool = False,
    ) -> dict[str, float | str]:
        # Log XAI probe configuration
        logger.info(f"Client {self.client_id}: Starting XAI probe for round {round_num}")
        logger.info(f"Client {self.client_id}: Target layer: {type(self.target_layer).__name__}")
        logger.info(f"Client {self.client_id}: Number of samples: {num_samples}")
        
        # Keep model in train mode for GradCAM gradient flow.
        # Only set BatchNorm layers to eval mode so they use running stats.
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                m.eval()

        del_aucs: list[float] = []
        ins_aucs: list[float] = []
        cosine_scores: list[float] = []
        ssim_scores: list[float] = []
        pearson_scores: list[float] = []
        cams: list[np.ndarray] = []

        sample_tensors: list[torch.Tensor] = []
        sample_labels: list[tuple[int, int]] = []

        seen = 0
        saved = 0

        # ── Stratified random sampling across ALL classes ──
        # Collect all available samples grouped by class
        # Cap collection to avoid loading entire dataset into GPU memory
        import random as _rand
        max_collect = num_samples * 3  # Collect 3x what we need for good class coverage
        all_by_class: dict[int, list[tuple[torch.Tensor, int]]] = {}
        total_collected = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            for i in range(images.size(0)):
                if total_collected >= max_collect:
                    break
                lbl = int(labels[i].item())
                all_by_class.setdefault(lbl, []).append(
                    (images[i:i+1].detach().cpu(), lbl)  # Move to CPU to save GPU memory
                )
                total_collected += 1
            if total_collected >= max_collect:
                break

        # Build balanced sample list: round-robin from each class
        available_classes = sorted(all_by_class.keys())
        probe_queue: list[tuple[torch.Tensor, int]] = []
        if available_classes:
            for cls in available_classes:
                _rand.shuffle(all_by_class[cls])
            per_class = max(1, num_samples // len(available_classes))
            for cls in available_classes:
                probe_queue.extend(all_by_class[cls][:per_class])
            # Fill remainder
            leftover = []
            for cls in available_classes:
                leftover.extend(all_by_class[cls][per_class:])
            _rand.shuffle(leftover)
            probe_queue.extend(leftover[:max(0, num_samples - len(probe_queue))])
        _rand.shuffle(probe_queue)
        probe_queue = probe_queue[:num_samples]
        logger.info(f"Client {self.client_id}: XAI probe stratified sampling — "
                    f"classes: {sorted(set(l for _, l in probe_queue))}, "
                    f"total: {len(probe_queue)}")

        for x, y_true in probe_queue:
            x = x.to(self.device)  # Move back to GPU for computation
            with torch.no_grad():
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pred_idx = int(torch.argmax(logits, dim=1).item())

            class_idx = pred_idx

            x_cam = x.clone().requires_grad_(True)
            try:
                cam = compute_gradcam_pp(
                    model=self.model,
                    x=x_cam,
                    target_layer=self.target_layer,
                    class_idx=class_idx,
                )
            except Exception as e:
                logger.warning(
                    f"GradCAM++ failed for sample {seen} (class={class_idx}): "
                    f"{type(e).__name__}: {str(e)}. Using fallback CAM."
                )
                cam = self._fallback_cam_from_input(x)

            try:
                del_auc = compute_deletion_auc(
                    model=self.model,
                    x=x.detach(),
                    cam=cam,
                    class_idx=class_idx,
                    device=self.device,
                    steps=10,
                )
                ins_auc = compute_insertion_auc(
                    model=self.model,
                    x=x.detach(),
                    cam=cam,
                    class_idx=class_idx,
                    device=self.device,
                    steps=10,
                )
            except Exception as e:
                logger.error(f"Deletion/Insertion AUC failed for sample {seen} (class={class_idx}): {type(e).__name__}: {str(e)}")
                del_auc = float("nan")
                ins_auc = float("nan")

            del_aucs.append(del_auc)
            ins_aucs.append(ins_auc)
            cams.append(cam)
            sample_tensors.append(x.detach())
            sample_labels.append((y_true, pred_idx))

            # ── Per-sample GPU cleanup to prevent memory accumulation ──
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if seen % 4 == 0:
                    torch.cuda.empty_cache()

            if seen in self.prev_cams:
                sim = compute_cam_similarity(self.prev_cams[seen], cam)
                cosine_scores.append(sim["cosine"])
                ssim_scores.append(sim["ssim"])
                pearson_scores.append(sim["pearson"])

            self.prev_cams[seen] = cam

            if saved < save_k:
                try:
                    save_gradcam_panel(
                        x=x.detach(),
                        cam=cam,
                        out_dir=self.save_dir,
                        round_num=round_num,
                        idx=seen,
                        true_label=y_true,
                        pred_label=pred_idx,
                        auc=del_auc,
                    )
                    saved += 1
                except Exception as e:
                    logger.warning(f"Failed to save GradCAM panel for sample {seen}: {type(e).__name__}: {str(e)}")

            seen += 1

        metrics = self._summarize_metrics(
            del_aucs,
            ins_aucs,
            cosine_scores,
            ssim_scores,
            pearson_scores,
        )
        metrics["xai_samples"] = float(len(cams))
        
        # Log XAI probe results
        logger.info(f"Client {self.client_id}: XAI probe completed - {len(cams)}/{num_samples} samples processed")
        if len(cams) > 0:
            logger.info(f"Client {self.client_id}: Deletion AUC: {metrics['xai_del_auc_mean']:.4f}, Insertion AUC: {metrics['xai_ins_auc_mean']:.4f}")
        else:
            logger.warning(f"Client {self.client_id}: XAI probe completed with zero valid CAM samples.")


        if shared_probe:
            metrics["xai_cam_stack_b64"] = encode_cam_stack(cams, downsample=cam_downsample)
        else:
            metrics["xai_cam_stack_b64"] = ""

        if run_heavy and sample_tensors:
            round_dir = self.save_dir / f"round_{round_num}"
            round_dir.mkdir(parents=True, exist_ok=True)
            # Limit heavy XAI to 3 samples (one per class if available)
            # to prevent OOM / PC hang on last round
            heavy_limit = min(3, len(sample_tensors))
            logger.info(f"Client {self.client_id}: Heavy XAI limited to {heavy_limit} samples (prevents OOM)")
            heavy_metrics = self._run_heavy_xai(
                round_dir=round_dir,
                sample_tensors=sample_tensors[:heavy_limit],
                sample_labels=sample_labels[:heavy_limit],
                run_shap=run_shap,
                run_lime=run_lime,
                run_attention=run_attention,
                loader=loader,
                round_num=round_num,
            )
            metrics.update(heavy_metrics)

        return metrics

    def _summarize_metrics(
        self,
        del_aucs: list[float],
        ins_aucs: list[float],
        cosine_scores: list[float],
        ssim_scores: list[float],
        pearson_scores: list[float],
    ) -> dict[str, float]:
        def _stats(arr: list[float]) -> tuple[float, float]:
            if not arr:
                return 0.0, 0.0
            a = np.asarray(arr, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return 0.0, 0.0
            return float(np.mean(a)), float(np.std(a))

        del_mean, del_std = _stats(del_aucs)
        ins_mean, ins_std = _stats(ins_aucs)
        cos_mean, cos_std = _stats(cosine_scores)
        ssim_mean, ssim_std = _stats(ssim_scores)
        pear_mean, pear_std = _stats(pearson_scores)

        return {
            "xai_del_auc_mean": del_mean,
            "xai_del_auc_std": del_std,
            "xai_ins_auc_mean": ins_mean,
            "xai_ins_auc_std": ins_std,
            "xai_cam_consistency_mean": cos_mean,
            "xai_cam_consistency_std": cos_std,
            "xai_temporal_stability_mean": ssim_mean,
            "xai_temporal_stability_std": ssim_std,
            "xai_temporal_pearson_mean": pear_mean,
            "xai_temporal_pearson_std": pear_std,
        }

    def _run_heavy_xai(
        self,
        round_dir: Path,
        sample_tensors: list[torch.Tensor],
        sample_labels: list[tuple[int, int]],
        run_shap: bool,
        run_lime: bool,
        run_attention: bool,
        loader=None,
        round_num: int = 0,
    ) -> dict[str, float]:
        """Run comprehensive XAI visualizations using the new module.

        Produces publication-quality multi-panel figures for each method
        (Grad-CAM++, LIME, SHAP, Attention Rollout) and a combined panel.
        Also computes cross-method agreement metrics.
        """
        heavy_cams = []

        # ── 1. Comprehensive visualization (the new module) ──
        try:
            comp_metrics = run_comprehensive_xai(
                model=self.model,
                target_layer=self.target_layer,
                loader=loader,
                device=self.device,
                round_num=round_num,
                save_dir=self.save_dir,
                client_id=self.client_id,
                num_samples=min(3, len(sample_tensors)),  # Cap at 3 to prevent OOM
                preprocess_fn=self._preprocess_tensor,
                run_gradcam=True,
                run_lime=run_lime,
                run_shap=run_shap,
                run_attention=run_attention,
            )
            logger.info(f"Client {self.client_id}: Comprehensive XAI visualization saved")
        except Exception as e:
            logger.error(f"Client {self.client_id}: Comprehensive XAI failed: {e}")
            comp_metrics = {}

        # ── 2. Cross-method agreement (legacy metric) ──
        sample = sample_tensors[0].to(self.device)
        y_true, y_pred = sample_labels[0]

        # Grad-CAM++
        try:
            gradcam_pp = compute_gradcam_pp(self.model, sample, self.target_layer, class_idx=y_pred)
            if gradcam_pp is not None:
                heavy_cams.append(gradcam_pp)
        except Exception:
            pass

        if run_shap:
            try:
                bg_count = min(8, len(sample_tensors))
                background = torch.cat(sample_tensors[:bg_count], dim=0).to(self.device)
                shap_values = compute_shap_explanation(self.model, background, sample)
                if isinstance(shap_values, list):
                    sv = shap_values[y_pred]
                else:
                    sv = shap_values[y_pred]
                shap_cam = np.sum(np.abs(sv[0]), axis=0)
                shap_cam = (shap_cam - shap_cam.min()) / (shap_cam.max() - shap_cam.min() + 1e-8)
                heavy_cams.append(shap_cam)
            except Exception:
                pass

        if run_lime:
            try:
                image = denormalize_imagenet(sample[0])
                explanation = compute_lime_explanation(
                    model=self.model,
                    image=image,
                    device=self.device,
                    preprocess_fn=self._preprocess_tensor,
                    num_samples=1000,
                    top_labels=1,
                )
                top_label = explanation.top_labels[0]
                local_exp = explanation.local_exp[top_label]
                segments = explanation.segments
                lime_heatmap = np.zeros(segments.shape, dtype=np.float32)
                for seg_id, weight in local_exp:
                    lime_heatmap[segments == seg_id] = weight
                lime_pos = np.maximum(lime_heatmap, 0)
                if lime_pos.max() > 0:
                    lime_pos = lime_pos / lime_pos.max()
                heavy_cams.append(lime_pos)
            except Exception:
                pass

        if run_attention:
            try:
                rollout_engine = AttentionRollout(self.model)
                attention_map = rollout_engine(sample, image_size=sample.shape[-1])
                if attention_map is not None:
                    att_cam = attention_map.detach().cpu().numpy() if isinstance(attention_map, torch.Tensor) else attention_map
                    heavy_cams.append(att_cam)
            except Exception:
                pass

        # Compute agreement metrics
        result = {}
        if len(heavy_cams) >= 2:
            agreements = compute_cross_method_agreement(heavy_cams)
            result["xai_cross_method_agreement_mean"] = agreements.get("agreement_mean", float("nan"))
        else:
            result["xai_cross_method_agreement_mean"] = float("nan")

        # Merge comprehensive metrics
        result.update(comp_metrics)
        return result
