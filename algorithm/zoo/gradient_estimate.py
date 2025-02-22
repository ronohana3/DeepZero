import torch


@torch.no_grad()
def calculate_loss(params_dict, network, x, y, loss_func):
    state_dict_backup = network.state_dict()
    network.load_state_dict(params_dict, strict=False)
    loss = loss_func(network(x), y).detach().item()
    network.load_state_dict(state_dict_backup)
    return loss


@torch.no_grad()
def rge(func, params_dict, sample_size, step_size, base=None):
    if base == None:
        base = func(params_dict)
    grads_dict = {}
    for _ in range(sample_size):
        perturbs_dict, perturbed_params_dict = {}, {}
        for key, param in params_dict.items():
            perturb = torch.randn_like(param)
            perturb /= (torch.norm(perturb) + 1e-8)
            perturb *= step_size
            perturbs_dict[key] = perturb
            perturbed_params_dict[key] = perturb + param
        directional_derivative = (func(perturbed_params_dict) - base) / step_size
        if len(grads_dict.keys()) == len(params_dict.keys()):
            for key, perturb in perturbs_dict.items():
                grads_dict[key] += perturb * directional_derivative / sample_size
        else:
            for key, perturb in perturbs_dict.items():
                grads_dict[key] = perturb * directional_derivative / sample_size
    return grads_dict


@torch.no_grad()
def cge(network, x, y, loss_func, step_size):
    output, intermediates = network(x, return_intermediates=True)
    base_loss = loss_func(output, y)
    grads_dict = {}
    for name, param in network.named_parameters():
        # TODO: check if this condition holds for all networks or just works for cnn model
        if not isinstance(param, torch.Tensor) or not param.dtype.is_floating_point:
            continue  # Skip non-trainable parameters (like num_batches_tracked)

        # Determine layer index of the parameter
        layer_idx = network.get_layer_index_from_name(name)
        if layer_idx == -1:
            raise ValueError(f"Could not find layer index for parameter: {name}")

        x_start = intermediates[layer_idx - 1] if layer_idx > 0 else x  # Use stored activations

        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()
        params_flat = param.view(-1)

        for idx in range(params_flat.numel()):
            params_flat[idx] += step_size
            perturbed_output = network(x_start, start_layer_idx=layer_idx)
            perturbed_loss = loss_func(perturbed_output, y)
            params_flat[idx] -= step_size
            directional_derivative_flat[idx] = (perturbed_loss - base_loss) / step_size

        grads_dict[name] = directional_derivative.to(param.device)
    return grads_dict


@torch.no_grad()
def zo_rge_qo(network, x, y, loss_func, step_size, quant_params):

    grads_dict = {}

    # Backup original parameters
    original_params = {name: param.clone() for name, param in network.named_parameters()}

    # Generate quantized noise for all layers
    noise_dict = {}
    for name, param in network.named_parameters():
        if not isinstance(param, torch.Tensor) or not param.dtype.is_floating_point:
            continue  # Skip non-trainable parameters

        # Get layer-specific quantization parameters
        if name not in quant_params:
            raise ValueError(f"Missing quantization parameters for layer: {name}")

        R_max, R_min, s = quant_params[name]

        # Generate quantized noise
        m = max(int(step_size / s), 1)  # Discretized step multiplier
        noise = torch.randint(-m, m + 1, param.shape, device=param.device) * s

        # Store noise for later use
        noise_dict[name] = noise

        # Perturb parameters with noise
        param.add_(noise).clamp_(R_min, R_max)

    # Compute perturbed loss after applying noise to all parameters
    loss_pos = loss_func(network(x), y)

    # Restore original parameters and apply negative noise
    for name, param in network.named_parameters():
        R_max, R_min, s = quant_params[name]
        if name in noise_dict:
            param.copy_(original_params[name] - noise_dict[name]).clamp_(R_min, R_max)

    # Compute perturbed loss with negative noise
    loss_neg = loss_func(network(x), y)

    # Restore original parameters
    for name, param in network.named_parameters():
        if name in original_params:
            param.copy_(original_params[name])

    # Compute gradient sign for each parameter
    for name in noise_dict:
        grads_dict[name] = torch.sign(loss_pos - loss_neg) * torch.sign(noise_dict[name])

    return grads_dict

@torch.no_grad()
def zoqo_per_layer_noise(network, x, y, loss_func, step_size, quant_params):

    grads_dict = {}

    for name, param in network.named_parameters():
        if not isinstance(param, torch.Tensor) or not param.dtype.is_floating_point:
            continue  # Skip non-trainable parameters

        # Get layer-specific quantization parameters
        if name not in quant_params:
            raise ValueError(f"Missing quantization parameters for layer: {name}")

        R_max, R_min, s = quant_params[name]

        # Generate quantized noise
        m = max(int(step_size / s), 1)  # Discretized step multiplier
        noise = torch.randint(-m, m + 1, param.shape, device=param.device) * s

        # Perturb parameters with noise positive
        param.add_(noise).clamp_(R_min, R_max)
        # Compute perturbed loss positive
        loss_pos = loss_func(network(x), y)
        # Perturb parameters with noise negative
        param.add_(-2*noise).clamp_(R_min, R_max)
        loss_neg = loss_func(network(x), y)

        grads_dict[name] = torch.sign(loss_pos - loss_neg) * torch.sign(noise)

        # Restore original parameters
        param.add_(noise).clamp_(R_min, R_max)

    return grads_dict


@torch.no_grad()
def zo_cge_qo(network, x, y, loss_func, quant_params):
    output, intermediates = network(x, return_intermediates=True)
    base_loss = loss_func(output, y)
    grads_dict = {}
    for name, param in network.named_parameters():
        if not isinstance(param, torch.Tensor) or not param.dtype.is_floating_point:
            continue  # Skip non-trainable parameters (like num_batches_tracked)

        # Get layer-specific quantization parameters
        if name not in quant_params:
            raise ValueError(f"Missing quantization parameters for layer: {name}")

        R_max, R_min, s = quant_params[name]

        # Determine layer index of the parameter
        layer_idx = network.get_layer_index_from_name(name)
        if layer_idx == -1:
            raise ValueError(f"Could not find layer index for parameter: {name}")

        x_start = intermediates[layer_idx - 1] if layer_idx > 0 else x  # Use stored activations

        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()
        params_flat = param.view(-1)

        for idx in range(params_flat.numel()):
            if params_flat[idx] != (params_flat[idx] + s).clamp_(R_min, R_max):
                params_flat[idx] += s
                perturbed_output = network(x_start, start_layer_idx=layer_idx)
                perturbed_loss = loss_func(perturbed_output, y)
                directional_derivative_flat[idx] = torch.sign(perturbed_loss - base_loss)
                params_flat[idx] -= s
            else:
                directional_derivative_flat[idx] = 0

        grads_dict[name] = directional_derivative.to(param.device)
    return grads_dict
