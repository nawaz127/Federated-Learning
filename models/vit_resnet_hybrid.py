import torch
import torch.nn as nn
import torch.nn.functional as F # Import F for normalization
from timm import create_model
from torchvision import models
from torchvision.models import ResNet18_Weights


class HybridViTCNNMLP(nn.Module):
    """
    Hybrid Vision Transformer (ViT) + ResNet18 (CNN) + MLP head
    - ViT takes 3-channel RGB (we replicate grayscale to RGB)
    - ResNet18 is adapted to 1-channel inputs (keeps ImageNet weights by mean init)
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        vit_name: str = "vit_base_patch16_224",
        dropout_rate: float = 0.3,
        freeze_backbones: bool = False,
    ):
        super().__init__()

        self.feature_maps = {} # For XAI visualization hooks, stores {'vit_features': ..., 'cnn_features': ...}
        self._hook_handles = []  # Track hooks for cleanup

        # ---- ViT backbone (features only)
        self.vit = create_model(vit_name, pretrained=pretrained, num_classes=0)  # feature extractor
        vit_dim = getattr(self.vit, "num_features", None) or self.vit.embed_dim

        # Register hook for ViT
        self._hook_handles.append(self.vit.norm.register_forward_hook(self.save_vit_features))

        # ---- ResNet18 backbone (1-channel friendly, features only)
        # Fix deprecated 'pretrained' parameter - use 'weights' instead
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.cnn = models.resnet18(weights=weights)
        # convert first conv to 1-channel by averaging pretrained RGB weights
        with torch.no_grad():
            w = self.cnn.conv1.weight  # [64, 3, 7, 7]
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.conv1.weight.copy_(w.mean(dim=1, keepdim=True))
        cnn_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # Register hook for CNN
        self._hook_handles.append(self.cnn.layer4.register_forward_hook(self.save_cnn_features))

        # ---- Fusion + classifier head
        self.classifier = nn.Sequential(
            nn.Linear(vit_dim + cnn_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        if freeze_backbones:
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.cnn.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the ViT and CNN backbones, making all their parameters trainable.
        """
        for p in self.vit.parameters():
            p.requires_grad = True
        for p in self.cnn.parameters():
            p.requires_grad = True
        print("HybridViTCNNMLP backbones UN-FROZEN.")

    def save_vit_features(self, module, input, output):
        self.feature_maps['vit_features'] = output

    def save_cnn_features(self, module, input, output):
        self.feature_maps['cnn_features'] = output

    def remove_feature_hook(self):
        """Remove all forward hooks to save memory during training when XAI is not needed."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W] (grayscale) or [B,3,H,W] (already RGB)
        """
        # ViT expects 3-channel
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        vit_feat = self.vit(x_rgb)            # [B, vit_dim]
        vit_feat = F.normalize(vit_feat, dim=1) # Normalize ViT features

        # ResNet path uses true grayscale (1-ch) if available
        cnn_in = x if x.size(1) == 1 else x.mean(1, keepdim=True)
        cnn_feat = self.cnn(cnn_in)           # [B, cnn_dim]
        cnn_feat = F.normalize(cnn_feat, dim=1) # Normalize CNN features

        fused = torch.cat([vit_feat, cnn_feat], dim=1)
        output = self.classifier(fused)
        from utils.common_utils import validate_tensor
        output = validate_tensor(output, "HybridViTCNN output")
        return output

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        vit_feat = self.vit(x_rgb)
        vit_feat = F.normalize(vit_feat, dim=1)
        cnn_in = x if x.size(1) == 1 else x.mean(1, keepdim=True)
        cnn_feat = self.cnn(cnn_in)
        cnn_feat = F.normalize(cnn_feat, dim=1)
        return torch.cat([vit_feat, cnn_feat], dim=1)

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (features) from the model for tasks like SHAP or t-SNE.
        """
        return self.extract_features(x)
