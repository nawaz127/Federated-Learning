import torch
import torch.nn as nn
from torchvision import models


class DenseNet121Medical(nn.Module):
    """
    DenseNet121 model for medical image classification using ImageNet pretrained weights.
    Converts first convolution to accept 1-channel input by averaging pretrained weights.
    """
    def __init__(self,
                 num_classes: int = 3,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 freeze_backbone: bool = False):  # <--- NEW ARGUMENT
        super().__init__()

        self.feature_maps = None # For XAI visualization hooks
        self._feature_hook_handle = None  # Track hook for cleanup

        # 1. Load the Backbone
        if pretrained:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.densenet = models.densenet121(weights=None)

        # NOTE: Dataloader outputs 3-channel RGB (grayscale→RGB conversion in CTScanDataset).
        # Keep original 3-channel conv0 weights from ImageNet pretraining.
        # Handle grayscale input in forward() if needed.

        # Register hook for XAI
        self._feature_hook_handle = self.densenet.features.norm5.register_forward_hook(self.save_features)

        # 3. Freeze Backbone (Optional)
        if freeze_backbone:
            for param in self.densenet.parameters():
                param.requires_grad = False
            print("DenseNet121 backbone frozen.")

        # 4. Create Custom Classifier
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()

        # The classifier weights are NEW, so they always have requires_grad=True by default
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def save_features(self, module, input, output):
        self.feature_maps = output

    def remove_feature_hook(self):
        """Remove the forward hook to save memory during training when XAI is not needed."""
        if self._feature_hook_handle is not None:
            self._feature_hook_handle.remove()
            self._feature_hook_handle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DenseNet121 model.
        x: [B, 3, H, W] (RGB) or [B, 1, H, W] (grayscale).
        """
        # Handle grayscale input by repeating to 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.densenet(x)
        output = self.classifier(features)
        from utils.common_utils import validate_tensor
        output = validate_tensor(output, "DenseNet121 output")
        return output

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the DenseNet backbone without classification.
        """
        return self.densenet(x)

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (features) from the model for tasks like SHAP or t-SNE.
        """
        return self.extract_features(x)

    def unfreeze_backbone(self):
        """
        Call this method later in training (e.g., Round 10) to fine-tune the whole model.
        """
        for param in self.densenet.parameters():
            param.requires_grad = True
        print("DenseNet121 backbone UN-FROZEN.")
