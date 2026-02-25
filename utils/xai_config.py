# utils/xai_config.py
# CANONICAL XAI configuration — used by both xai_utils.XAI_Factory and client.py
# If you need to change target layers, update them HERE (single source of truth).

GRAD_CAM_TARGET_LAYERS = {
    'densenet121': 'densenet.features.norm5',
    'mobilenetv3': 'backbone.features.16', # Assuming 17 layers in backbone.features, 0-indexed
    'resnet50': 'backbone.layer4',
    'vit': 'vit.norm',
    'swin_tiny': 'swin.norm',
    'hybridmodel': 'cnn.layer4', # For HybridViTCNNMLP, targeting the CNN branch
    'hybridswin': 'densenet.features.norm5', # For HybridSwinDenseNetMLP, targeting the DenseNet branch
    'LSeTNet': 'norm', # Target the batch norm layer before transformer blocks
    'vit_tiny': 'vit.norm', # For ViT-Tiny, similar to regular ViT
}

# LIME Configuration
LIME_DEFAULT_NUM_SAMPLES = 1000

# SHAP Configuration
# Note: For SHAP DeepExplainer, a representative background dataset is often used.
# The size/nature of this background dataset can significantly impact results.
# For simplicity, we'll assume a placeholder for now.
# Actual implementation might require passing a subset of training data.
SHAP_DEFAULT_BACKGROUND_SAMPLES = 100 # Number of samples to use for background when creating SHAP explainer

# Attention Rollout will need attention maps from TransformerBlock.