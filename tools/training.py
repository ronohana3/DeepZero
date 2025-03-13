from torch.nn import functional as F
import torch

def warmup_lr(optimizer, current_epoch, current_step, steps_per_epoch, warmup_epoch, base_lr):
    overall_steps = warmup_epoch * steps_per_epoch
    current_steps = current_epoch * steps_per_epoch + current_step
    lr = base_lr * current_steps/overall_steps
    for p in optimizer.param_groups:
        p['lr']=lr

def mean_squared_loss(x, y):
    y = F.one_hot(y) - 0.1
    return ( ( x - y )**2 ).mean()

def quantize_params(params, bit_width=8):
    """Quantizes parameters using uniform quantization."""
    qmin = 0
    qmax = 2**bit_width - 1
    Rmin = params.min()
    Rmax = params.max()
    scale = (Rmax - Rmin) / (qmax - qmin)
    quantized = torch.round((params - Rmin) / scale) * scale + Rmin
    return torch.clamp(quantized, Rmin, Rmax)

def initialize_quantization(network, num_bits=8):
    quant_params = {}
    layer_quant_params = {}

    for name, param in network.named_parameters():
        if not isinstance(param, torch.Tensor) or not param.dtype.is_floating_point:
            continue  # Skip non-trainable parameters

        # Extract layer identifier and type (weight/bias)
        layer_name = ".".join(name.split(".")[:-1])
        param_type = name.split(".")[-1]

        # Check if the layer is a BatchNorm layer
        is_batchnorm = isinstance(dict(network.named_modules())[layer_name], torch.nn.BatchNorm2d)

        if param_type == "weight":
            if is_batchnorm:
                # BatchNorm weight (gamma) typically ranges from [0.5, 2.0]
                R_max, R_min = 1.5, 0.5
            else:
                # R_max = 5*param.max().item()
                # R_min = 5*param.min().item()
                R_max = 2*param.max().item()
                R_min = 2*param.min().item()
            # Compute quantization scale
            s = (R_max - R_min) / (2 ** num_bits - 1)

            R_max = round(R_max / s) * s
            R_min = round(R_min / s) * s

            # Store quantization parameters
            layer_quant_params[layer_name] = (R_max, R_min, s)

            # Apply quantization
            param.data = torch.round(param / s) * s

            # Save in quant_params dictionary
            quant_params[name] = (R_max, R_min, s)

        elif param_type == "bias":
            if is_batchnorm:
                # BatchNorm bias (beta) typically ranges from [-0.1, 0.1]
                R_max, R_min = 0.1, -0.1
            else:
                R_max, R_min, s = layer_quant_params[layer_name]  # Use weight quantization params

            # Compute quantization scale
            s = (R_max - R_min) / (2 ** num_bits - 1)

            R_max = round(R_max / s) * s
            R_min = round(R_min / s) * s

            # Apply quantization
            param.data = torch.round(param / s) * s

            # Save in quant_params dictionary
            quant_params[name] = (R_max, R_min, s)

    return quant_params


def update_quant_params(network, quant_params, num_bits=8):
    levels = 2 ** num_bits  # number of quantization levels

    for name, param in network.named_parameters():
        if name not in quant_params:
            continue
        if not isinstance(param, torch.Tensor) or not param.dtype.is_floating_point:
            continue

        # Get current quantization parameters for this weight
        R_max, R_min, s = quant_params[name]
        R_max_old = R_max
        R_min_old = R_min

        # Compute distribution statistics of current weights
        weight_data = param.data
        actual_max = weight_data.max().item()
        actual_min = weight_data.min().item()
        mean_val = weight_data.mean().item()
        total_elems = weight_data.numel()

        # Compute saturation percentages at the ends of the range
        range_width = R_max_old - R_min_old
        high_threshold = R_max_old - 0.05 * range_width
        low_threshold = R_min_old + 0.05 * range_width
        # Count weights near the extremes
        count_high = int((weight_data >= high_threshold).sum().item())
        count_low = int((weight_data <= low_threshold).sum().item())
        pct_high = count_high / total_elems
        pct_low = count_low / total_elems

        # Determine new R_max and R_min
        R_max_new, R_min_new = R_max_old, R_min_old  # default to old range
        #Adjust if weights saturate either or both ends
        if pct_high >= 0.15 and pct_low >= 0.15:
            # Both sides saturated: expand range by 20%
            new_width = range_width * 1.20
            center = (R_max_old + R_min_old) / 2.0
            R_max_new = center + new_width / 2.0
            R_min_new = center - new_width / 2.0
        elif pct_high >= 0.40 or pct_low >= 0.40:
            # One side saturated: shift range to center around the mean of weights
            # Calculate shift needed to center the range on the mean
            current_center = (R_max_old + R_min_old) / 2.0
            shift = mean_val - current_center
            # Limit the shift so we don’t exclude any weight:
            if shift > 0:  # shifting range up (increasing R_min and R_max)
                max_up_shift = actual_min - R_min_old
                if max_up_shift < 0:
                    max_up_shift = 0.0
                shift = min(shift, max_up_shift)
            elif shift < 0:
                max_down_shift = R_max_old - actual_max  # how much we can lower R_max without excluding the highest weight
                if max_down_shift < 0:
                    max_down_shift = 0.0
                shift = max(shift, -max_down_shift)
            # Apply the allowed shift
            R_max_new = R_max_old + shift
            R_min_new = R_min_old + shift
        elif (actual_max - actual_min) <= 0.70 * range_width:
            # Contract range by 15%, without excluding any weights
            new_width = range_width * 0.85
            # Calculate slack on each end
            slack_high = R_max_old - actual_max
            slack_low = actual_min - R_min_old
            # Distribute the range reduction proportionally to slack
            reduction = range_width - new_width  # total amount to cut from the range
            slack_total = slack_high + slack_low if (slack_high + slack_low) != 0 else reduction
            cut_high = reduction * (slack_high / slack_total)
            cut_low = reduction * (slack_low / slack_total)
            R_max_new = R_max_old - cut_high
            R_min_new = R_min_old + cut_low
        # Ensure the adjusted range still contains all weights (safety check)
        R_max_new = max(R_max_new, actual_max)
        R_min_new = min(R_min_new, actual_min)

        # Recompute the quantization scale for the new range
        new_s = (R_max_new - R_min_new) / (levels - 1)
        if new_s == 0:
            # In case all weights are the same and range_width became 0
            new_s = 1e-8
        R_max_new = round(R_max_new / new_s) * new_s
        R_min_new = round(R_min_new / new_s) * new_s

        # Requantize the weight values in place using the new scale
        param.data.clamp_(R_min_new, R_max_new)  # clamp to the new range boundaries
        # Quantize: map to nearest quantized value
        param.data = torch.round(param / new_s) * new_s

        # Update quantization params for this weight
        quant_params[name] = (R_max_new, R_min_new, new_s)

    return quant_params


def validate_quantization(network, quant_params, tolerance=1e-5):
    for name, param in network.named_parameters():
        if name in quant_params:
            R_max, R_min, s = quant_params[name]

            quantized_values = torch.round(param / s) * s
            assert torch.allclose(param, quantized_values, atol=tolerance)
            # Correcting quantization to avoid accumulating numerical errors
            param.data = quantized_values


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
