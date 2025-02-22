import torch.nn as nn

class AdjustableCNN(nn.Module):
    def __init__(self, depth, channel_num=24):
        super(AdjustableCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc = nn.Linear(channel_num, 10)  # One FC layer with 32 input features and 10 output classes

        # Define the convolutional layers with adjustable depth
        for i in range(depth):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(3, channel_num, 3, padding=1))
            else:
                self.conv_layers.append(nn.Conv2d(channel_num, channel_num, 3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(channel_num))
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.MaxPool2d(2, 2))

        # Adaptive pooling to calculate the final input size for the FC layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    # def forward(self, x):
    #     for layer in self.conv_layers:
    #         x = layer(x)
    #     x = self.adaptive_pool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     return x

    def get_layer_index_from_name(self, param_name):
        layer_idx = 0  # Layer counter

        for name, layer in self.named_children():
            if isinstance(layer, nn.ModuleList):
                for sub_idx, sub_layer in enumerate(layer):
                    full_layer_name = f"{name}.{sub_idx}"
                    if param_name.startswith(full_layer_name):
                        return layer_idx
                    layer_idx += 1  # Increment for each sub-layer

            else:
                if param_name.startswith(name):  # Match prefix with layer name
                    return layer_idx
                layer_idx += 1  # Move to next layer

        return -1  # Return -1 if not found (shouldn't happen for valid params)

    def forward(self, x, return_intermediates=False, start_layer_idx=0):
        intermediates = []

        # Iterate through layers but start from `start_layer_idx`
        for i, layer in enumerate(self.conv_layers):
            if i >= start_layer_idx:  # Start from specified layer index
                x = layer(x)
            if return_intermediates and i >= start_layer_idx:
                intermediates.append(x.clone().detach())  # Store intermediate activations

        # Adaptive pooling & Flatten
        if start_layer_idx <= len(self.conv_layers):  # Only do this if we haven't skipped it
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)

        # Fully connected layers (if applicable)
        if start_layer_idx <= len(self.conv_layers) + 1:  # Only compute if FC layers aren't skipped
            x = self.fc(x)
            if return_intermediates:
                intermediates.append(x.clone().detach())

        return (x, intermediates) if return_intermediates else x

