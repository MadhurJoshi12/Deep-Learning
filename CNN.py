import torch
import torch.nn as nn

image = torch.randn(1,1,28,28)

conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1
)

output = conv(image)
# print("Input image", image.shape)
# print("Output image", output.shape)

# POOLING 
pool = nn.MaxPool2d(kernel_size=2)

y = pool(image)

print("Before Pooling", image.shape)
print("After Pooling", y.shape)
