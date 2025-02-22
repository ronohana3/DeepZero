import torch
import torch.nn as nn
from models.cnn import AdjustableCNN  # Import the author's CNN model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Test different depths to match Figure 2's x-axis values
depth_values = list(range(1, 10))  # Adjust based on actual values
param_counts = {}

for depth in depth_values:
    model = AdjustableCNN(depth=depth, channel_num=18)  # Adjust depth
    param_counts[depth] = count_parameters(model)

# Print the mapping of depth → number of parameters
for depth, params in param_counts.items():
    print(f"Depth: {depth}, Parameters: {params}")