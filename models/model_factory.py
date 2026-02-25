import torch
import torch.nn as nn
import torch.nn.functional as F


from models.densenet121 import DenseNet121Medical
from models.lsetnet import LSeTNet_model
from models.mobilenetv3 import MobileNetV3
from models.resnet_model import ResNet50
from models.swin_tiny import swin_tiny_model
from models.swint_densenet_hybrid import HybridSwinDenseNetMLP  # Corrected import
from models.vit import vit_model
from models.vit_resnet_hybrid import HybridViTCNNMLP  # Corrected import


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
    use_fedbn: bool = False, # Placeholder for now, to be implemented
    drop_path_rate: float = 0.1, # For transformer models
    lsetnet_num_transformer_blocks: int | None = None,
    lsetnet_num_heads: int | None = None,
    lsetnet_ff_dim_multiplier: int | None = None, # New parameter
):

    if model_name == 'resnet50':
        return ResNet50(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone)

    elif model_name == 'hybridmodel':
        return HybridViTCNNMLP(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate, freeze_backbones=freeze_backbone)
    elif model_name == "mobilenetv3":
        return MobileNetV3(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone)
    elif model_name == "hybridswin":
        return HybridSwinDenseNetMLP(num_classes=num_classes, pretrained=pretrained, dropout=dropout_rate, freeze_backbones=freeze_backbone)
    elif model_name == "densenet121":
        return DenseNet121Medical(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone)
    elif model_name == "LSeTNet":
        return LSeTNet_model(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=0.4,  # slightly lower dropout for stability
            freeze_backbone=freeze_backbone,
            drop_path_rate=0.2,  # more regularization
            layer_scale_init_values=1e-4,  # stronger layer scaling
            num_transformer_blocks=lsetnet_num_transformer_blocks if lsetnet_num_transformer_blocks is not None else 3,
            num_heads=lsetnet_num_heads if lsetnet_num_heads is not None else 8,
            ff_dim_multiplier=lsetnet_ff_dim_multiplier if lsetnet_ff_dim_multiplier is not None else 4, # Pass new parameter
        )
    elif model_name == "swin_tiny":
        return swin_tiny_model(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate, drop_path_rate=drop_path_rate, freeze_backbone=freeze_backbone)
    elif model_name == "vit":
        return vit_model(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate, drop_path_rate=drop_path_rate, freeze_backbone=freeze_backbone)
    elif model_name == "vit_tiny":
        return vit_model(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate, drop_path_rate=drop_path_rate, freeze_backbone=freeze_backbone, vit_name="vit_tiny_patch16_224")
    else:
        raise ValueError(f"Model {model_name} is not supported.")

def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical image classification.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for medical image classification
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + (1 - targets_one_hot) * self.smoothing / (self.num_classes - 1)
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss
