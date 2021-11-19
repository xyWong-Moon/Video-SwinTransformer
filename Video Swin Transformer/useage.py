import torch
from model import swin_tiny_patch4_window7_224

model = swin_tiny_patch4_window7_224(num_classes=1000)

input = torch.rand(10, 3, 32, 224, 224)  # [batch size, channl, frames, H, W]
output = model(input)
print(f'output_shape: {output.shape}')