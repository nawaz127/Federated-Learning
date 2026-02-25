import torch
import torch.nn as nn # Import for isinstance checks
import torch.nn.functional as F
import numpy as np
import cv2

# Import for LIME
import lime
from lime import lime_image

# Import for SHAP


class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.model.eval()
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer = self._get_target_layer()
        self._hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def _get_target_layer(self):
        # Dynamically find the target layer in the model
        # This assumes a hierarchical model structure, e.g., model.features.layerX
        # or model.backbone.layerX. It can now handle integer indices for Sequential modules.
        sub_layers = self.target_layer_name.split('.')
        current_module = self.model
        for s in sub_layers:
            if s.isdigit(): # Handle integer indexing for Sequential modules
                current_module = current_module[int(s)]
            elif hasattr(current_module, s):
                current_module = getattr(current_module, s)
            else:
                raise AttributeError(f"Layer '{s}' not found in '{self.target_layer_name}'")
        return current_module

    def forward(self, input_image, target_category):
        self.model.zero_grad()
        output = self.model(input_image)
        
        if target_category is None:
            target_category = torch.argmax(output, dim=1)

        one_hot = F.one_hot(target_category, num_classes=output.shape[1]).float()
        one_hot_output = torch.sum(one_hot * output)

        one_hot_output.backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        return self._generate_cam(activations, gradients, input_image.shape[2:])

    def _generate_cam(self, activations, gradients, image_size):
        # activations shape: [batch, channels, H, W]
        # gradients shape:   [batch, channels, H, W]
        # Use first sample in batch
        weights = np.mean(gradients[0], axis=(1, 2))  # [channels]
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, image_size)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)  # Prevent division by zero
        return cam

    def show_cam_on_image(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_image = heatmap + np.float32(img)
        cam_image = cam_image / np.max(cam_image)
        return np.uint8(255 * cam_image)

    def __call__(self, input_image, target_category=None):
        return self.forward(input_image, target_category)

    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __del__(self):
        self.remove_hooks()


class LimeImageExplainer:
    def __init__(self, model, preprocess_fn, device='cuda'):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.explainer = lime_image.LimeImageExplainer()
        self.device = device
        self.model.eval()

    def _predict_fn(self, images):
        # LIME provides images as numpy arrays, we need to convert them to tensors
        # and apply the same preprocessing as the model expects.
        # Images usually come in (N, H, W, C) format, convert to (N, C, H, W)
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2)
        images = images.to(self.device)
        
        # Apply model-specific preprocessing (e.g., normalization)
        images = self.preprocess_fn(images)

        with torch.no_grad():
            outputs = self.model(images)
            if isinstance(outputs, tuple): # Handle models that return (output, attention_weights)
                outputs = outputs[0]
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
        return probabilities

    def explain_instance(self, input_image_np, top_labels=5, hide_color=0, num_samples=1000):
        # input_image_np is expected to be a numpy array (H, W, C)
        explanation = self.explainer.explain_instance(
            input_image_np,
            self._predict_fn,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=num_samples
        )
        return explanation








class XAI_Factory:
    @staticmethod
    def create_explainer(method: str, model, preprocess_fn=None, background_data_np=None, target_layer_name=None, device='cuda', **kwargs):
        # We need to import xai_config here to avoid circular imports if xai_config imports something from xai_utils
        from xai_config import GRAD_CAM_TARGET_LAYERS, LIME_DEFAULT_NUM_SAMPLES, SHAP_DEFAULT_BACKGROUND_SAMPLES

        if method.lower() == "gradcam":
            if target_layer_name is None:
                # Dynamically get target layer if not provided
                if model.__class__.__name__.lower() in GRAD_CAM_TARGET_LAYERS:
                    target_layer_name = GRAD_CAM_TARGET_LAYERS[model.__class__.__name__.lower()]
                else:
                    raise ValueError("target_layer_name must be provided for GradCAM or model type not in GRAD_CAM_TARGET_LAYERS.")
            return GradCAM(model, target_layer_name)
        elif method.lower() == "lime":
            if preprocess_fn is None:
                raise ValueError("preprocess_fn must be provided for LIME.")
            num_samples = kwargs.get("num_samples", LIME_DEFAULT_NUM_SAMPLES)
            return LimeImageExplainer(model, preprocess_fn, device=device)
        # SHAP and Attention Rollout removed
        else:
            raise ValueError(f"Unknown XAI method: {method}")
