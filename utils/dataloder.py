import logging
import os
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set deterministic seed for reproducibility.
    Required for publication-grade experiments.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For fully deterministic operations on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"✅ Random seed set to {seed} for reproducibility")


class CTScanDataset(Dataset):

    def __init__(self, data_dir: str, transform=None, subset: str = 'train'):
        """
        Args:
            data_dir: Root directory containing class folders
            transform: Albumentations transform pipeline
            subset: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset

        # # Class mapping
        # self.class_to_idx = {'Bengin cases': 0, 'Malignant cases': 1, 'Normal cases': 2,}
        # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        # Dynamically generate class mapping from folder names
        class_names = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}


        # Load all image paths and labels
        self.samples = self._load_samples()

        # Calculate class weights for balanced training
        self.class_weights = self._calculate_class_weights()

        logger.info(f"Loaded {len(self.samples)} samples for {subset} set")
        self._print_class_distribution()


    def _load_samples(self) -> list[tuple[str, int]]:
        """Load all image paths and their corresponding labels"""
        samples = []

        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path) or class_name not in self.class_to_idx:
                continue

            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    samples.append((img_path, class_idx))

        return samples

    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets"""
        class_counts = [0] * len(self.class_to_idx)
        for _, label in self.samples:
            class_counts[label] += 1

        total_samples = len(self.samples)
        weights = []
        for count in class_counts:
            if count == 0:
                weights.append(0.0)
                logger.warning("⚠️ One or more classes have 0 samples — check your dataset.")
            else:
                weights.append(total_samples / (len(self.class_to_idx) * count))

        return torch.FloatTensor(weights)

    def _print_class_distribution(self):
        """Print class distribution for debugging"""
        class_counts = [0] * len(self.class_to_idx)
        for _, label in self.samples:
            class_counts[label] += 1

        logger.info("Class distribution:")
        for idx, count in enumerate(class_counts):
            class_name = self.idx_to_class[idx]
            logger.info(f"  {class_name}: {count} samples")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load and preprocess image
        image = self._load_and_preprocess_image(img_path)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, label

    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess medical CT scan image"""
        try:
            # Load image as grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError(f"Could not load image: {img_path}")

            # Medical image preprocessing
            image = self._apply_medical_preprocessing(image)

            # ✅ CRITICAL FIX: Convert grayscale to RGB for Transformer models (ViT, Swin, Hybrid CNN-Transformer)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            return image

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a blank 3-channel image as fallback (matches RGB output of normal path)
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def _apply_medical_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply medical-specific preprocessing to CT scans"""
        # Removed redundant Resize; Albumentations handles resizing

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Normalize to [0, 255] range
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to float32 for albumentations
        #image = image.astype(np.float32) / 255.0
        image = image.astype(np.uint8)

        return image

def get_medical_transforms(image_size: tuple[int, int] = (224, 224),
                          subset: str = 'train') -> A.Compose:
    """
    Get medical image augmentation transforms using Albumentations.
    
    ✅ FIXED: Uses ImageNet normalization (required for pretrained CNN, ViT, Swin)
    ✅ FIXED: Removed RandomResizedCrop (distorts anatomical structures in CT)
    ✅ FIXED: Uses Resize instead for CT scans
    
    Args:
        image_size: Target image size (height, width)
        subset: 'train', 'val', or 'test'
    
    Returns:
        Albumentations compose object
    """

    if subset == 'train':
        # Training augmentations - medical image specific
        # ✅ Uses ImageNet normalization for better convergence with pretrained models
        # ✅ Uses Resize instead of RandomResizedCrop to preserve anatomical structures
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.02, 0.02),
                rotate=(-10, 10),
                p=0.5
            ),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.Blur(blur_limit=3, p=0.15),
            A.CoarseDropout(max_holes=2, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, fill_value=0, p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    else:
        # Validation/Test transforms - minimal processing
        # ✅ Uses ImageNet normalization for consistency
        transform = A.Compose([
            A.Resize(224, 224),

            # ✅ FIXED: ImageNet normalization
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            ToTensorV2()
        ])

    return transform


def create_federated_client_loader(client_dir: str,
                                   batch_size: int = 16,
                                   num_workers: int = 2,
                                   pin_memory: bool = True) -> DataLoader:
    """
    Create a data loader for federated learning client.
    
    ✅ REQUIRED: Each client loads ONLY its own folder (no internal splitting)
    
    Args:
        client_dir: Path to client folder (e.g., "Federated_Dataset/Clients_NonIID/Client_1")
        batch_size: Batch size for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader for the federated client
    """
    logger.info(f"Creating federated client loader for: {client_dir}")

    # Allow passing custom transform for augmentation
    def_loader_transform = get_medical_transforms((224, 224), 'train')
    dataset = CTScanDataset(
        data_dir=client_dir,
        transform=def_loader_transform,
        subset='train'
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    logger.info(f"✅ Federated client loader created: {len(dataset)} samples, batch_size={batch_size}")
    return loader


def create_global_loader(data_dir: str,
                        batch_size: int = 16,
                        num_workers: int = 2,
                        pin_memory: bool = True,
                        subset: str = 'test') -> DataLoader:
    """
    Create a global data loader for validation or test set.
    
    ✅ REQUIRED: For server-side validation and testing in federated learning
    
    Args:
        data_dir: Path to validation or test folder (e.g., "Federated_Dataset/Validation")
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        subset: 'val' or 'test'
    
    Returns:
        DataLoader for global validation/test
    """
    logger.info(f"Creating global {subset} loader for: {data_dir}")

    # Allow passing custom transform for val/test
    def_loader_transform = get_medical_transforms((224, 224), subset)
    dataset = CTScanDataset(
        data_dir=data_dir,
        transform=def_loader_transform,
        subset=subset
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    logger.info(f"✅ Global {subset} loader created: {len(dataset)} samples")
    return loader


def create_data_loaders(data_dir: str,
                       batch_size: int = 16,
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       test_split: float = 0.1,
                       image_size: tuple[int, int] = (224, 224),
                       num_workers: int = 1,
                       pin_memory: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    ⚠️ DEPRECATED: This function is NOT suitable for federated learning!
    
    For federated learning, use:
        - create_federated_client_loader() for client training
        - create_global_loader() for validation/testing
    
    This function is kept for non-federated training scenarios only.
    
    Create train, validation, and test data loaders with proper medical image handling
    
    Args:
        data_dir: Root directory containing class folders
        batch_size: Batch size for training
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        image_size: Target image size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    logger.warning("⚠️ DEPRECATED: create_data_loaders() is NOT suitable for federated learning!")
    logger.warning("Use create_federated_client_loader() and create_global_loader() instead.")

    # Verify splits sum to 1
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    # Get transforms
    train_transform = get_medical_transforms(image_size, 'train')
    val_transform = get_medical_transforms(image_size, 'val')
    test_transform = get_medical_transforms(image_size, 'test')

    # Create full dataset to get all samples
    full_dataset = CTScanDataset(data_dir, transform=None, subset='full')

    # Split data indices
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Create random split
    train_indices, temp_indices = train_test_split(
        range(total_size),
        train_size=train_size,
        stratify=[full_dataset.samples[i][1] for i in range(total_size)],
        random_state=42
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        stratify=[full_dataset.samples[i][1] for i in temp_indices],
        random_state=42
    )

    # Create subset datasets
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    test_samples = [full_dataset.samples[i] for i in test_indices]

    # Create datasets with appropriate transforms
    train_dataset = CTScanDataset(data_dir, transform=train_transform, subset='train')
    train_dataset.samples = train_samples

    val_dataset = CTScanDataset(data_dir, transform=val_transform, subset='val')
    val_dataset.samples = val_samples

    test_dataset = CTScanDataset(data_dir, transform=test_transform, subset='test')
    test_dataset.samples = test_samples

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def visualize_batch(data_loader: DataLoader, num_samples: int = 8):
    """Visualize a batch of CT scan images"""
    # Get a batch
    batch_images, batch_labels = next(iter(data_loader))

    # Setup the plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    # Class names from the dataset
    class_names = ['Benign', 'Malignant', 'Normal']

    for i in range(min(num_samples, len(batch_images))):
        # Convert tensor to numpy and denormalize
        img = batch_images[i].cpu().numpy()

        # Handle different tensor shapes
        if len(img.shape) == 3:
            # Shape is [3, 224, 224] for RGB images (Transformer models)
            # Or [1, 224, 224] for grayscale
            if img.shape[0] == 3:
                # RGB image - transpose to [224, 224, 3]
                img = img.transpose(1, 2, 0)
            elif img.shape[0] == 1:
                # Grayscale - squeeze to [224, 224]
                img = img.squeeze(0)

        # Denormalize using ImageNet statistics
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Display
        if img.shape[-1] == 3:
            axes[i].imshow(img)
        else:
            axes[i].imshow(img, cmap='gray')

        axes[i].set_title(f'Label: {class_names[batch_labels[i]]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================================
# RTX 4060 8GB Memory Optimization Recommendations
# ============================================================================
"""
📌 Memory Optimization Guidelines for RTX 4060 (8GB VRAM):

| Model Type       | Recommended Batch Size |
|------------------|------------------------|
| CNN (ResNet,     |                        |
|  DenseNet, etc.) | batch_size = 16       |
|------------------|------------------------|
| Hybrid CNN-      |                        |
| Transformer      | batch_size = 8         |
|------------------|------------------------|
| ViT / Swin       | batch_size = 4         |
|------------------|------------------------|

Usage:
    # For CNN models:
    client_loader = create_federated_client_loader(client_dir, batch_size=16)
    
    # For Hybrid models:
    client_loader = create_federated_client_loader(client_dir, batch_size=8)
    
    # For ViT/Swin:
    client_loader = create_federated_client_loader(client_dir, batch_size=4)
"""


def get_class_weights(data_loader: DataLoader) -> torch.Tensor:
    """Calculate class weights from data loader"""
    class_counts = torch.zeros(3)
    total_samples = 0

    for _, labels in data_loader:
        for label in labels:
            class_counts[label] += 1
            total_samples += 1

    # Calculate weights
    weights = total_samples / (3 * class_counts)
    return weights


if __name__ == "__main__":
    DATA_DIR = r"E:\Python\Research\LungCancerFL\Federated_Learning_lung_cancer\DataSet\Lung-CT Scan"  # Replace with your path

    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=DATA_DIR,
        batch_size=16,
        image_size=(224, 224),
        num_workers=0,     # Set 0 for Windows if multiprocess errors
        pin_memory=False   # You can turn on if using GPU
    )

    visualize_batch(train_loader)
    weights = get_class_weights(train_loader)
    print("Class Weights:", weights)
