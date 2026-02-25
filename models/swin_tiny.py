import torch
import torch.nn as nn
from timm import create_model


class SwinTiny(nn.Module):
    """
    Swin-Tiny Transformer model adapted for medical image classification.
    - Handles 1-channel (grayscale) input by repeating to 3 channels.
    - Uses a custom classification head.
    - Supports freezing and unfreezing the backbone for FL strategies.
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        swin_name: str = "swin_tiny_patch4_window7_224",
        dropout_rate: float = 0.3,
        drop_path_rate: float = 0.1, # NEW ARGUMENT
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.feature_maps = None # For XAI visualization hooks
        self._feature_hook_handle = None  # Track hook for cleanup
        self.swin = create_model(swin_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)  # feature extractor
        swin_dim = getattr(self.swin, "num_features", None) or self.swin.embed_dim

        # Register hook for XAI
        self._feature_hook_handle = self.swin.norm.register_forward_hook(self.save_features)

        self.classifier = nn.Sequential(
            nn.Linear(swin_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.swin.parameters():
            p.requires_grad = False
        print("Swin-Tiny backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.swin.parameters():
            p.requires_grad = True
        print("Swin-Tiny backbone UN-FROZEN.")

    def save_features(self, module, input, output):
        self.feature_maps = output

    def remove_feature_hook(self):
        """Remove the forward hook to save memory during training when XAI is not needed."""
        if self._feature_hook_handle is not None:
            self._feature_hook_handle.remove()
            self._feature_hook_handle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swin Transformer expects 3-channel input
        # Convert 1-channel (grayscale) to 3-channel (RGB) by repeating
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.swin(x)
        output = self.classifier(features)
        from utils.common_utils import validate_tensor
        output = validate_tensor(output, "SwinTiny output")
        return output

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Convert 1-channel (grayscale) to 3-channel (RGB) by repeating
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        return self.swin(x)

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (features) from the model for tasks like SHAP or t-SNE.
        """
        return self.extract_features(x)

def swin_tiny_model(num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.3, drop_path_rate: float = 0.1, freeze_backbone: bool = False):
    """
    Factory function to create the SwinTiny model.
    """
    return SwinTiny(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone
    )

if __name__ == '__main__':
    test_num_classes = 3

    # Test with 3 channels (RGB input assumed by default in model factory)
    model_rgb = swin_tiny_model(num_classes=test_num_classes, pretrained=False)
    print("RGB Model:", model_rgb)
    dummy_input_rgb = torch.randn(1, 3, 224, 224) # Batch, Channels, H, W
    output_rgb = model_rgb(dummy_input_rgb)
    print("RGB Output shape:", output_rgb.shape)
    assert output_rgb.shape == (1, test_num_classes)
    features_rgb = model_rgb.extract_features(dummy_input_rgb)
    print("RGB Features shape:", features_rgb.shape)

    # Test with 1 channel (Grayscale input)
    model_gray = swin_tiny_model(num_classes=test_num_classes, pretrained=False)
    print("\nGrayscale Model:", model_gray)
    dummy_input_gray = torch.randn(1, 1, 224, 224) # Batch, Channels, H, W
    output_gray = model_gray(dummy_input_gray)
    print("Grayscale Output shape:", output_gray.shape)
    assert output_gray.shape == (1, test_num_classes)
    features_gray = model_gray.extract_features(dummy_input_gray)
    print("Grayscale Features shape:", features_gray.shape)

    # Test freeze/unfreeze backbone
    model_freeze_test = swin_tiny_model(num_classes=test_num_classes, pretrained=False, freeze_backbone=True)
    for name, param in model_freeze_test.named_parameters():
        if not name.startswith("classifier"):
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
    print(f"\nNumber of trainable parameters: {num_params}")
