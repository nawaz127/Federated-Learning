import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from lime import lime_image


def compute_lime_explanation(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    preprocess_fn,
    num_samples: int = 1000,
    top_labels: int = 1,
    save_path: str = None,
):
    """
    SECTION 7 — LIME INTEGRATION
    Implement LIME for image explanation.
    """
    model.eval()
    explainer = lime_image.LimeImageExplainer()

    def _predict(images: np.ndarray) -> np.ndarray:
        # LIME provides images in RGB [0, 255] or [0, 1] usually.
        # Convert to tensor and apply project-specific preprocessing
        tensor = torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device)
        tensor = preprocess_fn(tensor)
        with torch.no_grad():
            outputs = model(tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # No masking, using softmax safely
            probs = F.softmax(outputs, dim=1).cpu().numpy()
        return probs

    explanation = explainer.explain_instance(
        image,
        _predict,
        top_labels=top_labels,
        num_samples=num_samples,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Get the explanation for the top label
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
        )
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        # Use mark_boundaries to show explanation
        from skimage.segmentation import mark_boundaries
        plt.imshow(mark_boundaries(temp / 255.0 if temp.max() > 1 else temp, mask))
        plt.title(f"LIME Explanation (Label {explanation.top_labels[0]})")
        
        plt.savefig(save_path)
        plt.close()

    return explanation
