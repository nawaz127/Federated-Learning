import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
from models.lsetnet import LSeTNet_model
import time

# Model setup
model = LSeTNet_model(num_classes=3, img_size=224, in_channels=3, num_transformer_blocks=3, num_heads=8)
model.eval()

# Dummy input
batch_size = 4
input_tensor = torch.randn(batch_size, 3, 224, 224)

# Test output shape
with torch.no_grad():
    output = model(input_tensor)
    print('Output shape:', output.shape)
    assert output.shape == (batch_size, 3), 'Output shape mismatch!'

# Benchmark speed
start = time.time()
for _ in range(10):
    _ = model(input_tensor)
end = time.time()
print(f'Average inference time per batch: {(end-start)/10:.4f} seconds')

# Benchmark memory
print('Model parameters:', sum(p.numel() for p in model.parameters()))
print('Model trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# If CUDA is available, test on GPU
if torch.cuda.is_available():
    model.cuda()
    input_tensor = input_tensor.cuda()
    with torch.no_grad():
        output = model(input_tensor)
    print('GPU Output shape:', output.shape)
    torch.cuda.empty_cache()
