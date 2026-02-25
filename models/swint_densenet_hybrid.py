import torch
import torch.nn as nn
import torch.nn.functional as F # Import F for normalization
from timm import create_model
from torchvision import models


class HybridSwinDenseNetMLP(nn.Module):
    """
    Swin-T (RGB) + DenseNet121 (1-ch) fused classifier for grayscale CT slices.
    - Replicates grayscale to RGB for Swin.
    - Converts DenseNet first conv to 1-channel by averaging pretrained weights.
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        swin_name: str = "swin_tiny_patch4_window7_224",
        dropout: float = 0.3,
        freeze_backbones: bool = False,
    ):
        super().__init__()

        self.feature_maps = {} # For XAI visualization hooks
        self._hook_handles = []  # Track hooks for cleanup

        # Swin backbone (pooled features)
        self.swin = create_model(swin_name, pretrained=pretrained, num_classes=0)
        swin_dim = getattr(self.swin, "num_features", None) or self.swin.embed_dim

        # Register hook for Swin
        self._hook_handles.append(self.swin.norm.register_forward_hook(self.save_swin_features))

        # DenseNet121 backbone (1-channel, features only)
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        # Convert first conv to 1-ch
        with torch.no_grad():
            w = self.densenet.features.conv0.weight  # [64, 3, 7, 7]
            self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.densenet.features.conv0.weight.copy_(w.mean(dim=1, keepdim=True))
        dn_dim = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # features only

        # Register hook for DenseNet
        self._hook_handles.append(self.densenet.features.norm5.register_forward_hook(self.save_densenet_features))

        # Fusion head
        self.classifier = nn.Sequential(
            nn.Linear(swin_dim + dn_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        if freeze_backbones:
            for p in self.swin.parameters(): p.requires_grad = False
            for p in self.densenet.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the Swin and DenseNet backbones, making all their parameters trainable.
        """
        for p in self.swin.parameters(): p.requires_grad = True
        for p in self.densenet.parameters(): p.requires_grad = True
        print("HybridSwinDenseNetMLP backbones UN-FROZEN.")

    def save_swin_features(self, module, input, output):
        self.feature_maps['swin_features'] = output

    def save_densenet_features(self, module, input, output):
        self.feature_maps['densenet_features'] = output

    def remove_feature_hook(self):
        """Remove all forward hooks to save memory during training when XAI is not needed."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] or [B,3,H,W]
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        swin_feat = self.swin(x_rgb)  # [B, swin_dim]
        swin_feat = F.normalize(swin_feat, dim=1) # Normalize Swin features

        x_gray = x if x.size(1) == 1 else x.mean(1, keepdim=True)
        dn_feat = self.densenet(x_gray)  # [B, dn_dim]
        dn_feat = F.normalize(dn_feat, dim=1) # Normalize DenseNet features

        fused = torch.cat([swin_feat, dn_feat], dim=1)
        output = self.classifier(fused)
        from utils.common_utils import validate_tensor
        output = validate_tensor(output, "HybridSwinDenseNet output")
        return output

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        swin_feat = self.swin(x_rgb)
        swin_feat = F.normalize(swin_feat, dim=1)
        x_gray = x if x.size(1) == 1 else x.mean(1, keepdim=True)
        dn_feat = self.densenet(x_gray)
        dn_feat = F.normalize(dn_feat, dim=1)
        return torch.cat([swin_feat, dn_feat], dim=1)

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (features) from the model for tasks like SHAP or t-SNE.
        """
        return self.extract_features(x)
