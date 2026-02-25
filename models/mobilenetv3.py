import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3(nn.Module):
    """
    MobileNetV3 model for Federated Learning, useful for resource-constrained devices
    like mobile phones and IoT devices.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5, freeze_backbone: bool = False):
        super().__init__()

        self.feature_maps = None # For XAI visualization hooks
        self._feature_hook_handle = None  # Track hook for cleanup
        if pretrained:
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v3_large(weights=None)

        # NOTE: Dataloader outputs 3-channel RGB (grayscale→RGB conversion in CTScanDataset).
        # Keep original 3-channel conv weights from ImageNet pretraining.
        # Handle grayscale input in forward() if needed.

        # Register hook for XAI
        self._feature_hook_handle = self.backbone.features[-1].register_forward_hook(self.save_features)

        num_features = self.backbone.classifier[0].in_features

        self.backbone.classifier = nn.Sequential()

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("MobileNetV3 backbone frozen.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("MobileNetV3 backbone UN-FROZEN.")

    def _initialize_weights(self):
        """Initialize only the custom classifier head weights, not the pretrained backbone."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def save_features(self, module, input, output):
        self.feature_maps = output

    def remove_feature_hook(self):
        """Remove the forward hook to save memory during training when XAI is not needed."""
        if self._feature_hook_handle is not None:
            self._feature_hook_handle.remove()
            self._feature_hook_handle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle grayscale input by repeating to 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.backbone(x)
        output = self.classifier(features)
        from utils.common_utils import validate_tensor
        output = validate_tensor(output, "MobileNetV3 output")
        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (features) from the model for tasks like SHAP or t-SNE.
        """
        return self.extract_features(x)
