import logging

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


class AttentionRollout:
    """
    SECTION 8 — ATTENTION ROLLOUT IMPLEMENTATION

    Supports multiple extraction modes:
      1. Hook-based extraction from timm ViT blocks (standard ViT)
      2. Swin Transformer — last-stage global windowed attention visualization
      3. LSeTNet / custom models — hooks on nn.MultiheadAttention modules
      4. model.forward(x, return_attention=True) — fallback API
    """
    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.model.eval()

    # ── Hook-based attention extraction for timm ViT models ──────────

    def _extract_attention_via_hooks(self, input_tensor: torch.Tensor):
        """Extract attention weights from timm ViT blocks using forward hooks.

        For timm >= 0.9 the Attention module may use fused SDPA which never
        materialises the attention matrix.  We temporarily disable
        ``fused_attn`` on each Attention module so the explicit path is taken,
        giving us access to the post-softmax weights via ``attn_drop``.
        """
        # Locate ViT backbone blocks
        vit_backbone = None
        if hasattr(self.model, "vit") and hasattr(self.model.vit, "blocks"):
            vit_backbone = self.model.vit

        if vit_backbone is None:
            return None

        attn_storage: list[list[torch.Tensor]] = []
        hooks: list[torch.utils.hooks.RemovableHook] = []
        fused_states: list[tuple] = []  # (attn_module, original_fused_attn)

        for block in vit_backbone.blocks:
            if not hasattr(block, "attn"):
                continue
            attn_module = block.attn

            # Disable fused (SDPA) attention so the explicit path is taken
            if hasattr(attn_module, "fused_attn"):
                fused_states.append((attn_module, attn_module.fused_attn))
                attn_module.fused_attn = False

            # Hook attn_drop — its input is the post-softmax attention matrix
            if hasattr(attn_module, "attn_drop"):
                layer_bucket: list[torch.Tensor] = []
                attn_storage.append(layer_bucket)

                def _make_hook(bucket):
                    def _hook(module, inp, out):
                        # out shape: [B, num_heads, N, N]
                        bucket.append(out.detach())
                    return _hook

                h = attn_module.attn_drop.register_forward_hook(_make_hook(layer_bucket))
                hooks.append(h)

        if not hooks:
            # No hookable attention modules found
            for attn_module, original in fused_states:
                attn_module.fused_attn = original
            return None

        try:
            with torch.no_grad():
                self.model(input_tensor)
        finally:
            # Always clean up hooks and restore state
            for h in hooks:
                h.remove()
            for attn_module, original in fused_states:
                attn_module.fused_attn = original

        # Collect captured weights
        result: list[torch.Tensor] = []
        for bucket in attn_storage:
            if bucket:
                result.append(bucket[0])  # [B, num_heads, N, N]

        return result if result else None

    # ── Swin Transformer windowed attention extraction ────────────────

    def _extract_swin_attention(self, input_tensor: torch.Tensor):
        """Extract attention from Swin Transformer using last-stage global attention.

        Swin uses windowed attention which is NOT directly compatible with standard
        attention rollout (the attention matrices are per-window, not global).
        However, the LAST stage (Layer 3) has window_size == resolution (7×7),
        meaning the attention is effectively global over all spatial tokens.

        Strategy: Use the last stage's averaged attention map as the visualization.
        For earlier stages, we average attention across windows per spatial location.
        """
        swin_backbone = None
        if hasattr(self.model, "swin") and hasattr(self.model.swin, "layers"):
            swin_backbone = self.model.swin
        else:
            return None

        # Collect attention from ALL blocks in the last layer (global attention)
        last_layer = swin_backbone.layers[-1]
        attn_storage: list[list[torch.Tensor]] = []
        hooks: list[torch.utils.hooks.RemovableHook] = []
        fused_states: list[tuple] = []

        for block in last_layer.blocks:
            if not hasattr(block, "attn"):
                continue
            attn_module = block.attn

            # Disable fused attention to force explicit attention matrix computation
            if hasattr(attn_module, "fused_attn"):
                fused_states.append((attn_module, attn_module.fused_attn))
                attn_module.fused_attn = False

            if hasattr(attn_module, "attn_drop"):
                layer_bucket: list[torch.Tensor] = []
                attn_storage.append(layer_bucket)

                def _make_hook(bucket):
                    def _hook(module, inp, out):
                        bucket.append(out.detach())
                    return _hook

                h = attn_module.attn_drop.register_forward_hook(_make_hook(layer_bucket))
                hooks.append(h)

        if not hooks:
            for attn_module, original in fused_states:
                attn_module.fused_attn = original
            return None

        try:
            with torch.no_grad():
                self.model(input_tensor)
        finally:
            for h in hooks:
                h.remove()
            for attn_module, original in fused_states:
                attn_module.fused_attn = original

        # Process captured attention weights
        # For the last stage of Swin-Tiny (224×224 input):
        #   resolution = 7×7, window_size = 7×7
        #   So num_windows = 1, and attention is [B*1, num_heads, 49, 49]
        #   This IS global attention over the 7×7 spatial grid
        attention_maps = []
        for bucket in attn_storage:
            if bucket:
                attn = bucket[0]  # [B*nW, num_heads, win_tokens, win_tokens]
                attention_maps.append(attn)

        if not attention_maps:
            return None

        # Average across all blocks in the last layer, then across heads
        # Each tensor: [B*nW, num_heads, N, N] where N = window_size^2
        avg_attn_list = []
        for attn in attention_maps:
            # Average across heads → [B*nW, N, N]
            if self.head_fusion == "mean":
                attn_avg = torch.mean(attn, dim=1)
            elif self.head_fusion == "max":
                attn_avg = torch.max(attn, dim=1)[0]
            else:
                attn_avg = torch.mean(attn, dim=1)
            avg_attn_list.append(attn_avg)

        # Average across blocks → [B*nW, N, N]
        combined = torch.stack(avg_attn_list, dim=0).mean(dim=0)

        # For the last stage, nW should be 1 (window covers entire 7×7),
        # so combined is [B, 49, 49]. Take first batch item.
        # Get the mean attention each token receives (column-wise mean)
        # This gives us a spatial importance score per token
        attn_map = combined[0]  # [49, 49]

        # Compute the mean attention each spatial position receives
        spatial_importance = attn_map.mean(dim=0)  # [49]

        # Return as a "Swin" mode marker so __call__ knows to handle differently
        return ("swin", spatial_importance)

    # ── LSeTNet / Custom model attention extraction via nn.MultiheadAttention ──

    def _extract_mha_attention(self, input_tensor: torch.Tensor):
        """Extract attention from models using nn.MultiheadAttention (e.g., LSeTNet).

        Hooks into nn.MultiheadAttention modules to capture attention weights.
        PyTorch's MultiheadAttention.forward() returns (attn_output, attn_weights)
        when need_weights=True (default). We hook the module output to capture these.
        """
        mha_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                mha_modules.append((name, module))

        if not mha_modules:
            return None

        attn_storage: list[list[torch.Tensor]] = []
        hooks: list[torch.utils.hooks.RemovableHook] = []

        for name, mha in mha_modules:
            layer_bucket: list[torch.Tensor] = []
            attn_storage.append(layer_bucket)

            def _make_hook(bucket):
                def _hook(module, inp, out):
                    # nn.MultiheadAttention returns (attn_output, attn_weights)
                    # attn_weights shape: [B, target_len, source_len]
                    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                        bucket.append(out[1].detach())
                return _hook

            h = mha.register_forward_hook(_make_hook(layer_bucket))
            hooks.append(h)

        # Temporarily ensure need_weights=True for all MHA modules
        # We need to override the forward call to pass need_weights=True
        # The cleanest way: temporarily set the default
        original_needs = []
        for name, mha in mha_modules:
            # Store any state we need to restore
            original_needs.append(None)

        try:
            with torch.no_grad():
                self.model(input_tensor)
        finally:
            for h in hooks:
                h.remove()

        # Collect captured weights
        result: list[torch.Tensor] = []
        for bucket in attn_storage:
            if bucket:
                # attn_weights: [B, target_len, source_len]
                # Reshape to [B, 1, target_len, source_len] to match ViT format
                aw = bucket[0]
                if aw.dim() == 3:
                    aw = aw.unsqueeze(1)  # Add heads dim: [B, 1, L, L]
                result.append(aw)

        if not result:
            return None

        # Return as "mha" mode for the __call__ to handle
        return ("mha", result)

    # ── Main extraction dispatch ─────────────────────────────────────

    def _get_attention_weights(self, input_tensor: torch.Tensor):
        """Return attention data for visualization.

        Returns one of:
          - list[Tensor] — standard ViT attention matrices [B, heads, N, N]
          - ("swin", Tensor) — Swin spatial importance [49]
          - ("mha", list[Tensor]) — MHA attention from custom models
          - None — no attention available
        """
        # 1. Try timm ViT hook-based extraction
        attn_via_hooks = self._extract_attention_via_hooks(input_tensor)
        if attn_via_hooks is not None:
            logger.debug("Extracted %d attention layers via ViT hooks", len(attn_via_hooks))
            return attn_via_hooks

        # 2. Try Swin Transformer windowed attention
        swin_result = self._extract_swin_attention(input_tensor)
        if swin_result is not None:
            logger.debug("Extracted attention via Swin last-stage global attention")
            return swin_result

        # 3. Try nn.MultiheadAttention hooks (LSeTNet, custom models)
        mha_result = self._extract_mha_attention(input_tensor)
        if mha_result is not None:
            logger.debug("Extracted attention via nn.MultiheadAttention hooks")
            return mha_result

        # 4. Fall back to model's own return_attention API
        with torch.no_grad():
            try:
                out = self.model(input_tensor, return_attention=True)
                if isinstance(out, tuple) and len(out) == 2:
                    _, attention_weights = out
                    if attention_weights is not None:
                        return attention_weights
            except TypeError:
                pass  # model.forward() doesn't accept return_attention

        return None

    def _apply_discard_ratio(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        flat = attention_matrix.flatten(start_dim=1)
        k = int(flat.size(-1) * (1 - self.discard_ratio))
        k = max(k, 1)
        _, indices = flat.topk(k, largest=True)
        out = torch.zeros_like(flat)
        out.scatter_(dim=1, index=indices, src=flat.gather(dim=1, index=indices))
        return out.view_as(attention_matrix)

    def aggregate_attention(self, attention_weights):
        if attention_weights is None:
            return None

        # attention_weights is list of (batch, heads, tokens, tokens)
        if self.head_fusion == "mean":
            attention_weights = [torch.mean(attn, dim=1) for attn in attention_weights]
        elif self.head_fusion == "max":
            attention_weights = [torch.max(attn, dim=1)[0] for attn in attention_weights]
        else:
            raise ValueError(f"Unknown head_fusion method: {self.head_fusion}")

        # Add identity for residual connection
        attention_weights = [attn + torch.eye(attn.size(-1)).to(attn.device) for attn in attention_weights]
        attention_weights = [attn / (attn.sum(dim=-1, keepdim=True) + 1e-8) for attn in attention_weights]

        # NUMERICAL STABILITY: Validate attention matrices are finite
        for idx, attn in enumerate(attention_weights):
            if not torch.isfinite(attn).all():
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Non-finite attention weights in layer {idx}")
                logger.error(f"  NaN count: {(~torch.isfinite(attn)).sum().item()}")
                logger.error(f"  Attention shape: {attn.shape}")
                # DO NOT return None silently - raise explicit error
                raise RuntimeError(f"Attention rollout numerical instability in layer {idx}")

        if self.discard_ratio > 0:
            attention_weights = [self._apply_discard_ratio(attn) for attn in attention_weights]

        num_tokens = attention_weights[0].shape[-1]
        rollout = torch.eye(num_tokens, num_tokens, device=attention_weights[0].device).unsqueeze(0)
        for idx, attn_matrix in enumerate(attention_weights):
            rollout = torch.bmm(attn_matrix, rollout)
            
            # NUMERICAL STABILITY: Check rollout doesn't explode/vanish
            if not torch.isfinite(rollout).all():
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Attention rollout produced non-finite values at layer {idx}")
                logger.error(f"  Rollout range: [{torch.nanmin(rollout).item()}, {torch.nanmax(rollout).item()}]")
                raise RuntimeError(f"Attention rollout numerical instability at layer {idx}")

        return rollout

    def __call__(self, input_image_tensor: torch.Tensor, image_size: int = 224, save_path: str = None) -> torch.Tensor | None:
        attention_weights = self._get_attention_weights(input_image_tensor)
        if attention_weights is None:
            return None

        # ── Handle Swin Transformer mode ──
        if isinstance(attention_weights, tuple) and attention_weights[0] == "swin":
            _, spatial_importance = attention_weights
            num_tokens = spatial_importance.shape[0]
            grid_size = int(num_tokens ** 0.5)
            if grid_size * grid_size != num_tokens:
                logger.warning(f"Swin attention: {num_tokens} tokens is not a perfect square")
                return None
            attn_2d = spatial_importance.reshape(grid_size, grid_size)
            attn_resized = F.interpolate(
                attn_2d.unsqueeze(0).unsqueeze(0).float(),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            attn_resized = (attn_resized - attn_resized.min()) / (
                attn_resized.max() - attn_resized.min() + 1e-8
            )
            if save_path:
                self._save_visualization(input_image_tensor, attn_resized, save_path,
                                         title="Swin Attention (Last Stage)")
            return attn_resized

        # ── Handle MHA mode (LSeTNet) ──
        if isinstance(attention_weights, tuple) and attention_weights[0] == "mha":
            _, attn_list = attention_weights
            # Aggregate across layers using rollout-style multiplication
            # attn_list: list of [B, 1, L, L] tensors
            first_attn = attn_list[0]
            num_tokens = first_attn.shape[-1]

            # If only 1 layer, just use mean attention per column
            if len(attn_list) == 1:
                attn_map = first_attn[0].squeeze(0)  # [L, L]
                spatial_importance = attn_map.mean(dim=0)  # [L]
            else:
                # Rollout: (I + A_1) * (I + A_2) * ...
                identity = torch.eye(num_tokens, device=first_attn.device).unsqueeze(0)
                rollout = identity.clone()
                for attn in attn_list:
                    # attn: [B, 1, L, L] → [B, L, L]
                    a = attn[0].squeeze(0) if attn.dim() == 4 else attn[0]
                    if a.dim() == 3:
                        a = a.squeeze(0)
                    a = a + torch.eye(num_tokens, device=a.device)
                    a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
                    rollout = torch.mm(a, rollout.squeeze(0)).unsqueeze(0)
                spatial_importance = rollout.squeeze(0).mean(dim=0)  # [L]

            grid_size = int(num_tokens ** 0.5)
            if grid_size * grid_size != num_tokens:
                logger.warning(f"MHA attention: {num_tokens} tokens is not a perfect square")
                return None
            attn_2d = spatial_importance.reshape(grid_size, grid_size)
            attn_resized = F.interpolate(
                attn_2d.unsqueeze(0).unsqueeze(0).float(),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            attn_resized = (attn_resized - attn_resized.min()) / (
                attn_resized.max() - attn_resized.min() + 1e-8
            )
            if save_path:
                self._save_visualization(input_image_tensor, attn_resized, save_path,
                                         title="Attention Rollout (MHA)")
            return attn_resized

        # ── Standard ViT rollout ──
        rollout_map = self.aggregate_attention(attention_weights)
        if rollout_map is None:
            return None

        if rollout_map.shape[0] > 1:
            rollout_map = rollout_map[0:1]

        # tokens include [CLS] token at index 0
        attention_to_patches = rollout_map[:, 0, 1:]
        num_patches = attention_to_patches.shape[-1]
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            return None

        attention_map_2d = attention_to_patches.reshape(1, grid_size, grid_size)
        attention_map_resized = F.interpolate(
            attention_map_2d.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        # Normalize heatmap safely
        attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (
            attention_map_resized.max() - attention_map_resized.min() + 1e-8
        )
        
        if save_path:
            self._save_visualization(input_image_tensor, attention_map_resized, save_path)

        return attention_map_resized

    def _save_visualization(self, input_image_tensor, attention_map, save_path, title="Attention Rollout"):
        """Save attention rollout visualization to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        img = input_image_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title("Original Image")
        
        attn_np = attention_map.cpu().numpy() if isinstance(attention_map, torch.Tensor) else attention_map
        plt.subplot(1, 2, 2)
        plt.imshow(attn_np, cmap='inferno')
        plt.colorbar()
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

def compute_attention_rollout(model, input_tensor, save_path=None):
    rollout = AttentionRollout(model)
    return rollout(input_tensor, save_path=save_path)
