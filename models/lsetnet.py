
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from utils.common_utils import validate_tensor

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
        self.se_weights = None # For XAI visualization

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        self.se_weights = y # Store the SE attention weights
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + shortcut # Add()
        out = self.relu(out) # Activation after addition as in Keras example (relu after Add)
        out = self.maxpool(out)
        return out

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
    def __init__(self, num_classes=3, img_size=224, in_channels=3, embed_dim=512, num_heads=4, ff_dim_multiplier=4, num_transformer_blocks=1, dropout_rate=0.5, freeze_backbone: bool = False, drop_path_rate: float = 0.1, layer_scale_init_values: float = 1e-5):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.in_channels = in_channels # Store in_channels
        self.embed_dim = embed_dim  # Store embed_dim

        self.feature_maps = None  # For XAI visualization hooks
        self._feature_hook_handle = None  # Track hook for cleanup

        # Linear projection to match transformer embed dim
        self.cnn_to_transformer = nn.Linear(512, embed_dim)
        # Positional Encoding for Transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, (self.img_size // 32)**2, embed_dim) * 0.02)

        # Initial CNN layer
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ResidualSEBlocks
        self.res_blocks = nn.Sequential(
            ResidualSEBlock(64, 128),
            ResidualSEBlock(128, 256),
            ResidualSEBlock(256, 512)
        )

        # Final CNN layer before Transformer
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Spatial Attention (CBAM)
        self.cbam = CBAM(512)

        # Register hook for XAI on the last CNN layer before transformer
        self._feature_hook_handle = self.cbam.register_forward_hook(self._save_features_hook)

        # Normalization before Transformer
        self.norm = nn.BatchNorm2d(512)

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim_multiplier=ff_dim_multiplier,
                dropout=dropout_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_values=layer_scale_init_values
            ) for _ in range(num_transformer_blocks)
        ])


        # Classification Head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes),
        )

        # Initialize classifier weights for better stability
        if isinstance(self.classifier[0], nn.Linear):
            nn.init.xavier_uniform_(self.classifier[0].weight)
            if self.classifier[0].bias is not None:
                nn.init.zeros_(self.classifier[0].bias)
        if isinstance(self.classifier[-1], nn.Linear):
            if self.classifier[-1].bias is not None:
                nn.init.zeros_(self.classifier[-1].bias)

        # Freeze backbone if required (must be after all layers are initialized)
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        # Freeze initial conv, residual blocks, final conv, and transformer
        for param in self.conv_block1.parameters():
            param.requires_grad = False
        for param in self.res_blocks.parameters():
            param.requires_grad = False
        for param in self.conv_block5.parameters():
            param.requires_grad = False
        for param in self.cbam.parameters(): # Freeze CBAM
            param.requires_grad = False
        for param in self.norm.parameters(): # Freeze norm
            param.requires_grad = False
        for param in self.cnn_to_transformer.parameters(): # Freeze projection
            param.requires_grad = False
        self.pos_embedding.requires_grad = False # Freeze positional embedding
        for block in self.transformer_blocks: # Iterate through transformer blocks
            for param in block.parameters():
                param.requires_grad = False
        print("LSeTNet backbone frozen.")

    def unfreeze_backbone(self):
        # Unfreeze initial conv, residual blocks, final conv, and transformer
        for param in self.conv_block1.parameters():
            param.requires_grad = True
        for param in self.res_blocks.parameters():
            param.requires_grad = True
        for param in self.conv_block5.parameters():
            param.requires_grad = True
        for param in self.cbam.parameters(): # Unfreeze CBAM
            param.requires_grad = True
        for param in self.norm.parameters(): # Unfreeze norm
            param.requires_grad = True
        for param in self.cnn_to_transformer.parameters(): # Unfreeze projection
            param.requires_grad = True
        self.pos_embedding.requires_grad = True # Unfreeze positional embedding
        for block in self.transformer_blocks: # Iterate through transformer blocks
            for param in block.parameters():
                param.requires_grad = True
        print("LSeTNet backbone UN-FROZEN.")

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() # Set BatchNorm to evaluation mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        print("LSeTNet BatchNorm layers frozen.")

    def unfreeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train() # Set BatchNorm to training mode
                m.weight.requires_grad = True
                m.bias.requires_grad = True
        print("LSeTNet BatchNorm layers UN-FROZEN.")

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

        # Block 1
        x = self.conv_block1(x)

        # Blocks 2-4
        x = self.res_blocks(x)

        # Block 5
        x = self.conv_block5(x)

        # Apply CBAM
        x = self.cbam(x)

        # Apply Normalization before Transformer
        x = self.norm(x)

        # Transformer
        # Reshape to (B, H*W, C) for positional embedding
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2) # (B, H*W, C)
        x = self.cnn_to_transformer(x) # Project to (B, H*W, ff_dim)
        x = x + self.pos_embedding[:, :x.size(1), :] # Add positional embedding

        for block in self.transformer_blocks:
            x = block.forward_flat(x) if hasattr(block, 'forward_flat') else self._apply_block_flat(block, x)

        # Classification Head
        # x is (B, L, ff_dim)
        x = x.transpose(1, 2).view(b, self.embed_dim, h, w)
        x = self.global_avg_pool(x).view(x.size(0), -1) # Flatten after GlobalAveragePooling
        x = self.classifier(x)
        x = validate_tensor(x, "LSeTNet output")
        return x

    def _apply_block_flat(self, block, x_flat):
        # Utility to apply block to flat sequence and keep it flat
        # Pre-LN logic already integrated in TransformerBlock.forward above
        # But we need to avoid the internal reshape to 4D and back if possible
        # For now, we will update TransformerBlock to have forward_flat or just handle it here
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

        x = self.conv_block1(x)
        x = self.res_blocks(x)
        x = self.conv_block5(x)

        # Apply CBAM
        x = self.cbam(x)

        # Apply Normalization before Transformer
        x = self.norm(x)

        # Transformer
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2) # (B, H*W, C)
        x = self.cnn_to_transformer(x) # Project to (B, H*W, ff_dim)
        x = x + self.pos_embedding[:, :x.size(1), :] # Add positional embedding

        for block in self.transformer_blocks:
            x = block.forward_flat(x) if hasattr(block, 'forward_flat') else self._apply_block_flat(block, x)

        # Features before final dense layers
        # x is (B, L, C)
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.global_avg_pool(x).view(x.size(0), -1) 
        return x

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
    num_transformer_blocks: int = 1
) -> LSeTNet:
    """
    Factory function to create the LSeTNet model.
    pretrained argument is kept for compatibility with model_factory but not used here.
    """
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