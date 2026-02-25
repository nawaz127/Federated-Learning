
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet50(nn.Module):
    """
    ResNet50 model for more complex medical image analysis
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5, freeze_backbone: bool = False):
        super().__init__()

        self.feature_maps = None # For XAI visualization hooks
        self._feature_hook_handle = None  # Track hook for cleanup
        
        # Fix deprecated 'pretrained' parameter - use 'weights' instead
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet50(weights=weights)
        # Keep 3-channel input (grayscale images are converted to RGB in dataloader)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Register hook for XAI (store handle so it can be removed)
        self._feature_hook_handle = self.backbone.layer4.register_forward_hook(self.save_features)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

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
        print("ResNet50 backbone frozen.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("ResNet50 backbone UN-FROZEN.")

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
        # Keep forward path in float32 for numerical stability in federated aggregation.
        x = x.float()
        features = self.backbone(x).float()
        # Clamp features to prevent extreme values from blowing up the classifier
        features = torch.clamp(features, min=-1e6, max=1e6)
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        output = self.classifier(features).float()
        from utils.common_utils import validate_tensor
        output = validate_tensor(output, "ResNet50 output")  # warns + cleans NaN, does not raise
        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (features) from the model for tasks like SHAP or t-SNE.
        """
        return self.extract_features(x)


# def get_model(model_name: str, num_classes: int, pretrained: bool = True, dropout_rate: float = 0.5):
#     if model_name == 'resnet50':
#         return MedicalResNet50(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
#     elif model_name == 'customcnn':
#         return CustomCNN(num_classes=num_classes)
#     else:
#         raise ValueError(f"Model {model_name} is not supported.")



# class FocalLoss(nn.Module):
#     """
#     Focal Loss for addressing class imbalance in medical image classification.
#     """
#     def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

#         if self.reduction == 'mean':
#             return F_loss.mean()
#         elif self.reduction == 'sum':
#             return F_loss.sum()
#         else:
#             return F_loss


# class LabelSmoothingLoss(nn.Module):
#     """
#     Label smoothing loss for medical image classification
#     """
#     def __init__(self, num_classes: int, smoothing: float = 0.1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.num_classes = num_classes
#         self.smoothing = smoothing
#         self.confidence = 1.0 - smoothing

#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         log_probs = F.log_softmax(inputs, dim=1)
#         targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
#         targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
#         loss = (-targets_smooth * log_probs).sum(dim=1).mean()
#         return loss
