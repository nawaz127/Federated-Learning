
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from utils.common_utils import validate_tensor
from models.residual_se_block import ResidualSEBlock

# Ensure project root is in sys.path for direct execution
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Helper for DropPath
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect implementation that is typically used in
    Transformers units.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# Helper for LayerScale
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
        return x * self.gamma

# CBAM Modules
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim_multiplier=4, dropout=0.1, drop_path_rate=0.1, layer_scale_init_values=1e-5):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        ff_dim = embed_dim * ff_dim_multiplier # Calculate internal FFN dimension
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(), # GELU is common in modern transformers
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.ls1 = LayerScale(embed_dim, init_values=layer_scale_init_values) if layer_scale_init_values else nn.Identity()
        self.ls2 = LayerScale(embed_dim, init_values=layer_scale_init_values) if layer_scale_init_values else nn.Identity()

    def forward_flat(self, x_flat):
        # Attention block (Pre-LN)
        x_norm1 = self.layernorm1(x_flat)
        attn_output, _ = self.att(x_norm1, x_norm1, x_norm1)
        x_flat = x_flat + self.ls1(self.drop_path(attn_output))

        # FFN block (Pre-LN)
        x_norm2 = self.layernorm2(x_flat)
        ffn_output = self.ffn(x_norm2)
        ffn_output = self.dropout(ffn_output)
        x_flat = x_flat + self.ls2(self.drop_path(ffn_output))

        x_flat = validate_tensor(x_flat, "TransformerBlock output")
        return x_flat

    def forward(self, x):
        # x is 4D (B, C, H, W). Reshape for MultiheadAttention
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1).transpose(1, 2) # (B, H*W, C)
        
        x_flat = self.forward_flat(x_flat)

        # Reshape back to 4D
        return x_flat.transpose(1, 2).view(b, c, h, w)


class LSeTNet(nn.Module):
    def __init__(self, num_classes=3, img_size=224, in_channels=3, embed_dim=512, num_heads=8, ff_dim_multiplier=4, num_transformer_blocks=2, dropout_rate=0.5, freeze_backbone: bool = False, drop_path_rate: float = 0.2, layer_scale_init_values: float = 1e-5):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.in_channels = in_channels # Store in_channels
        self.embed_dim = embed_dim  # Fixed: Use argument instead of hardcoded 64
        # Lightweight custom CNN backbone with GroupNorm
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Kaiming initialization for all layers
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.feature_maps = None  # For XAI visualization hooks
        self._feature_hook_handle = None  # Track hook for cleanup

        # Linear projection to match transformer embed dim
        self.cnn_to_transformer = nn.Linear(128, self.embed_dim)
        # Positional Encoding for Transformer
        # Assuming spatial dimension after backbone and conv_block5 is (img_size // 16)
        spatial_dim = self.img_size // 16
        self.pos_embedding = nn.Parameter(torch.empty(1, spatial_dim**2, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Initial CNN layer with GroupNorm
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            SEBlock(32)
        )

        # ResidualSEBlocks (with DropPath and now GroupNorm internally)
        self.res_blocks = nn.Sequential(
            ResidualSEBlock(32, 64, drop_path_rate=drop_path_rate),
            ResidualSEBlock(64, 128, drop_path_rate=drop_path_rate)
        )

        # Final CNN layer before Transformer with GroupNorm
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            SEBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Spatial Attention (CBAM)
        self.cbam = CBAM(128)

        # Register hook for XAI on the last CNN layer before transformer
        self._feature_hook_handle = self.cbam.register_forward_hook(self._save_features_hook)

        # Normalization before Transformer (GroupNorm)
        self.norm = nn.GroupNorm(8, 128)
        self.pre_transformer_norm = nn.LayerNorm(128)

        # Transformer Blocks
        # Linear DropPath decay (Swin-style)
        if num_transformer_blocks > 1:
            dpr = torch.linspace(0, drop_path_rate, num_transformer_blocks).tolist()
        else:
            dpr = [drop_path_rate]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                ff_dim_multiplier=ff_dim_multiplier,
                dropout=dropout_rate,
                drop_path_rate=dpr[i],
                layer_scale_init_values=layer_scale_init_values
            ) for i in range(num_transformer_blocks)
        ])


        # Classification Head (Using LayerNorm or GroupNorm instead of BatchNorm1d)
        # Concatenating Avg and Max pool gives 2 * embed_dim channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * self.embed_dim),
            nn.Linear(2 * self.embed_dim, num_classes)
        )

        # Initialize classifier weights for better stability
        if isinstance(self.classifier[1], nn.Linear):
            nn.init.xavier_uniform_(self.classifier[1].weight)
            if self.classifier[1].bias is not None:
                nn.init.zeros_(self.classifier[1].bias)

        # Freeze backbone if required (must be after all layers are initialized)
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        # Freeze everything except the classifier
        for name, param in self.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("LSeTNet backbone frozen.")

    def unfreeze_backbone(self):
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True
        print("LSeTNet backbone UN-FROZEN.")

    def freeze_bn(self):
        # Update: Freezing GroupNorm/LayerNorm for consistency (though less critical than BN)
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        print("LSeTNet normalization layers frozen.")

    def unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
        print("LSeTNet normalization layers UN-FROZEN.")

    def _save_features_hook(self, module, input, output):
        self.feature_maps = output

    # Alias for consistency with other models
    save_features = _save_features_hook

    def remove_feature_hook(self):
        """Remove the forward hook to save memory during training when XAI is not needed."""
        if self._feature_hook_handle is not None:
            self._feature_hook_handle.remove()
            self._feature_hook_handle = None


    def forward(self, x):
        # Handle 1-channel to 3-channel conversion if needed
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1) # Convert grayscale to RGB

        # 1. Custom CNN Backbone
        x = self.backbone(x)

        # 2. Transition and Residual Blocks
        x = self.conv_block1(x)
        x = self.res_blocks(x)
        x = self.conv_block5(x)

        # 3. Spatial Attention
        x = self.cbam(x)

        # 4. Normalization and Reshape for Transformer
        x = self.norm(x)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2) # (B, H*W, C)
        x = self.pre_transformer_norm(x)

        # 5. Linear Projection and Positional Embedding
        x = self.cnn_to_transformer(x)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # 6. Transformer Blocks
        for block in self.transformer_blocks:
            x = block.forward_flat(x) if hasattr(block, 'forward_flat') else self._apply_block_flat(block, x)

        # 7. Reshape back to 4D for Global Pooling
        x = x.transpose(1, 2).view(b, self.embed_dim, h, w)

        # 8. AdaptiveConcatPool: concat avg and max pool
        avg_pool = self.global_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1).view(x.size(0), -1)

        # 9. Classifier
        x = self.classifier(x)
        x = validate_tensor(x, "LSeTNet output")
        return x

    def _apply_block_flat(self, block, x_flat):
        # Utility to apply block to flat sequence and keep it flat
        b, l, c = x_flat.shape
        h = w = int(l**0.5)
        x_4d = x_flat.transpose(1, 2).view(b, c, h, w)
        x_out_4d = block(x_4d)
        return x_out_4d.view(b, c, -1).transpose(1, 2)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the LSeTNet backbone before the classification head.
        """
        # Handle 1-channel to 3-channel conversion if needed
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1) # Convert grayscale to RGB

        x = self.backbone(x)
        x = self.conv_block1(x)
        x = self.res_blocks(x)
        x = self.conv_block5(x)
        x = self.cbam(x)
        x = self.norm(x)

        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.pre_transformer_norm(x)
        x = self.cnn_to_transformer(x)
        x = x + self.pos_embedding[:, :x.size(1), :]

        for block in self.transformer_blocks:
            x = block.forward_flat(x) if hasattr(block, 'forward_flat') else self._apply_block_flat(block, x)

        # Features before final dense layers (using same ConcatPool logic)
        x = x.transpose(1, 2).view(b, self.embed_dim, h, w)
        avg_pool = self.global_pool(x)
        max_pool = self.global_max_pool(x)
        return torch.cat([avg_pool, max_pool], dim=1).view(x.size(0), -1)

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (features) from the model for tasks like SHAP or t-SNE.
        """
        return self.extract_features(x)


def LSeTNet_model(
    num_classes: int = 3,
    img_size: int = 224,
    in_channels: int = 3,
    pretrained: bool = False,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
    drop_path_rate: float = 0.1,
    layer_scale_init_values: float = 1e-5,
    embed_dim: int = 512,
    num_heads: int = 8, # Default of 8 is common for 512 dim
    ff_dim_multiplier: int = 4,
    num_transformer_blocks: int = 2
) -> LSeTNet:
    """
    Factory function to create the LSeTNet model.
    pretrained argument is kept for compatibility with model_factory but not used here.

    Note: For RTX 4060 (8GB), 224x224 input is safe. MultiHeadAttention memory is O(N^2) with N tokens. Avoid higher resolutions for stability.
    """
    # Ablation option: allow 2 or 4 transformer blocks
    if num_transformer_blocks not in [2, 4]:
        print(f"Warning: num_transformer_blocks={num_transformer_blocks} is non-standard. Recommended: 2 or 4.")
    return LSeTNet(
        num_classes=num_classes,
        img_size=img_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim_multiplier=ff_dim_multiplier,
        num_transformer_blocks=num_transformer_blocks,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
        layer_scale_init_values=layer_scale_init_values
    )

# Example usage (for testing purposes, not part of FL pipeline)
if __name__ == '__main__':
    # Assume IMG_SIZE and NUM_CLASSES are defined elsewhere or passed
    test_img_size = 224 # Corrected to standard 224
    test_num_classes = 3
    print("[INFO] For RTX 4060 (8GB), 224x224 input is safe. MultiHeadAttention memory is O(N^2) with N tokens. Avoid higher resolutions for stability.")

    # Test with 3 channels (RGB)
    model_rgb = LSeTNet_model(num_classes=test_num_classes, img_size=test_img_size, in_channels=3, num_transformer_blocks=2, num_heads=8)
    print("RGB Model:", model_rgb)
    dummy_input_rgb = torch.randn(2, 3, test_img_size, test_img_size) # Batch, Channels, H, W
    output_rgb = model_rgb(dummy_input_rgb)
    print("RGB Output shape:", output_rgb.shape)
    assert output_rgb.shape == (2, test_num_classes)
    features_rgb = model_rgb.extract_features(dummy_input_rgb)
    print("RGB Features shape:", features_rgb.shape)

    # Test with 1 channel (Grayscale)
    model_gray = LSeTNet_model(num_classes=test_num_classes, img_size=test_img_size, in_channels=1)
    print("\nGrayscale Model:", model_gray)
    dummy_input_gray = torch.randn(2, 1, test_img_size, test_img_size) # Batch, Channels, H, W
    output_gray = model_gray(dummy_input_gray)
    print("Grayscale Output shape:", output_gray.shape)
    assert output_gray.shape == (2, test_num_classes)
    features_gray = model_gray.extract_features(dummy_input_gray)
    print("Grayscale Features shape:", features_gray.shape)

    # Test freeze/unfreeze backbone
    model_freeze_test = LSeTNet_model(num_classes=test_num_classes, img_size=test_img_size, in_channels=3, freeze_backbone=True)
    for name, param in model_freeze_test.named_parameters():
        if "classifier" not in name:
            assert param.requires_grad == False, f"Param {name} not frozen!"
        else:
            assert param.requires_grad == True, f"Classifier param {name} is frozen!"
    print("\nFreeze backbone test successful: Backbone params are frozen, classifier is not.")

    model_freeze_test.unfreeze_backbone()
    for name, param in model_freeze_test.named_parameters():
        assert param.requires_grad == True, f"Param {name} not unfrozen!"
    print("Unfreeze backbone test successful: All backbone params are unfrozen.")

    # Check number of parameters
    num_params = sum(p.numel() for p in model_rgb.parameters() if p.requires_grad)
    print(f"\nNumber of trainable parameters (RGB): {num_params/1e6:.2f}M")