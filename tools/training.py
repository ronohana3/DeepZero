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
