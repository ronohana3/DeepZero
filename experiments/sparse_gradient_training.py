import math
import os
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import sys
from models.cnn import AdjustableCNN
from cfg import results_path
from tools import *
from algorithm.zoo import cge, zoqo_per_layer_noise, zo_cge_qo, zo_rge_qo
from data import prepare_dataset


def main(args):
    # Misc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    save_path = os.path.join(results_path, gen_folder_name(args, ignore=['log', 'dataset', 'seed', 'network', 'cnn_channel_num', 'nesterov','scheduler']))

    # Data
    loaders, class_num = prepare_dataset(args.dataset, args.batch_size)

    # Network
    if args.network == "resnet20":
        from models.resnet_s import resnet20, param_name_to_module_id_rn20
        param_name_to_module_id = param_name_to_module_id_rn20
        network_init_func = resnet20
        network_kwargs = {
            'num_classes': class_num
        }
    elif args.network == "cnn":
        network_init_func = AdjustableCNN
        network_kwargs = {
            'depth': args.cnn_depth,
            'channel_num': args.cnn_channel_num
        }
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    network = network_init_func(**network_kwargs).to(device)

    print(f"Training CNN with {count_parameters(network)} parameters using {args.method}")

    # Optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum, nesterov=args.nesterov)
    global_length = (args.epoch - args.warmup_epochs) * len(loaders['train'])
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_length)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * global_length),
                                                                                int(0.75 * global_length)], gamma=0.1)
    else:
        raise NotImplementedError(f'scheduler {args.scheduler} not implemented')

    # Makedir or Resume
    from_scratch = False
    if args.log:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=False)
            best_acc = 0.
            epoch = 0
            from_scratch = True
        elif os.path.exists(os.path.join(save_path, "ckpt.pth")):
            state_dict = torch.load(os.path.join(save_path, "ckpt.pth"), map_location=device)
            for key, val in state_dict["state_dicts"].items():
                if key == 'quant_params':
                    continue
                eval(f"{key}.load_state_dict(val)")
            if args.method in ['ZO_CGE_Q', 'ZO_RGE_Q']:
                quant_params = state_dict["state_dicts"]["quant_params"]
            best_acc = state_dict["best_acc"]
            epoch = state_dict["epoch"]
        else:
            best_acc = 0.
            epoch = 0
            from_scratch = True

        tensorboard_path = os.path.join(save_path, 'tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        logger = SummaryWriter(tensorboard_path)
        save_args_to_file(args, save_path)
    else:
        epoch = 0

    if args.method in ['ZO_CGE_Q', 'ZO_RGE_Q']:
        if from_scratch:
            quant_params = initialize_quantization(network, num_bits=args.num_bits)
    # Training Loop
    while epoch < args.epoch:
        epoch += 1

        if args.method in ['ZO_CGE_Q', 'ZO_RGE_Q'] and epoch in [62]:
            print("requan")
            quant_params = update_quant_params(network, quant_params, args.num_bits)

        ###############
        #### Train ####
        ###############

        network.train()
        acc = AverageMeter()
        loss = AverageMeter()

        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                    desc=f"Epoch {epoch} Training", ncols=160)

        for i, (x, y) in enumerate(pbar):

            if epoch <= args.warmup_epochs:
                warmup_lr(optimizer, epoch - 1, i + 1, len(loaders['train']), args.warmup_epochs, args.lr)

            x_cuda, y_cuda = x.to(device), y.to(device)

            if args.method == 'FO':

                optimizer.zero_grad()
                fx = network(x_cuda)
                loss_batch = F.cross_entropy(fx, y_cuda)
                loss_batch.backward()
                optimizer.step()

            elif args.method == 'ZO_CGE':

                with torch.no_grad():
                    fx = network(x_cuda)
                    loss_batch = F.cross_entropy(fx, y_cuda).cpu()
                # Compute CGE gradient estimate
                loss_func = torch.nn.CrossEntropyLoss()
                grads_dict = cge(network=network, x=x_cuda, y=y_cuda, loss_func=loss_func, step_size=args.zoo_step_size)

                optimizer.zero_grad()

                # Apply CGE gradient updates
                for name, param in network.named_parameters():
                    if name in grads_dict:
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        param.grad.copy_(grads_dict[name])

                optimizer.step()

            elif args.method in ['ZO_CGE_Q', 'ZO_RGE_Q']:
                optimizer.zero_grad()
                with torch.no_grad():
                    fx = network(x_cuda)
                    loss_batch = F.cross_entropy(fx, y_cuda).cpu()

                    loss_func = torch.nn.CrossEntropyLoss()
                    # Compute gradient sign using ZOQO
                    if args.method == 'ZO_CGE_Q':
                        grads_dict = zo_cge_qo(network, x_cuda, y_cuda, loss_func, quant_params)
                    else:
                        grads_dict = zo_rge_qo(network, x_cuda, y_cuda, loss_func, args.zoo_step_size, quant_params)

                    # Apply updates with quantized learning rate
                    for name, param in network.named_parameters():
                            if name in grads_dict:
                                R_max, R_min, s = quant_params[name]  # Get layer-specific quantization params

                                # Compute quantized learning rate
                                eta_q = max(round(args.lr / s), 1) * s
                                # Update parameter with quantized step
                                param.data = torch.clamp(param - eta_q * grads_dict[name], R_min, R_max)

                validate_quantization(network, quant_params)

            else:
                raise NotImplementedError(f'method {args.method} not implemented')

            acc.update(torch.argmax(fx, 1).eq(y_cuda).float().mean().item(), y.size(0))
            loss.update(loss_batch.item(), y.size(0))

            if epoch > args.warmup_epochs and args.method not in ['ZO_CGE_Q', 'ZO_RGE_Q']:
                scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(f"Lr {lr:.2e} Acc {100 * acc.avg:.2f}% Loss {loss.avg}")
        if args.log:
            logger.add_scalar("train/acc", acc.avg, epoch)
            logger.add_scalar("train/loss", loss.avg, epoch)

        ##############
        #### Test ####
        ##############
        network.eval()
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epoch {epoch} Testing:", ncols=120)
        acc = AverageMeter()
        loss = AverageMeter()
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(x)
                loss_batch = F.cross_entropy(fx, y).cpu()
            acc.update(torch.argmax(fx, 1).eq(y).float().mean(), y.size(0))
            loss.update(loss_batch.item(), y.size(0))

            pbar.set_postfix_str(f"Acc {100 * acc.avg:.2f}% Loss {loss.avg}")
        if args.log:
            logger.add_scalar("test/acc", acc.avg, epoch)
            logger.add_scalar("test/loss", loss.avg, epoch)

        # Save CKPT
        if args.log:
            state_dict = {
                "state_dicts": {
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                },
                "epoch": epoch,
                "best_acc": best_acc,
            }

            if args.method in ['ZO_CGE_Q', 'ZO_RGE_Q']:
                state_dict["state_dicts"]['quant_params'] = quant_params
            if acc.avg > best_acc:
                best_acc = acc.avg
                state_dict['best_acc'] = best_acc
                torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))


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
    """Update quantization parameters (R_max, R_min, s) for each layer and requantize weights in place."""
    levels = 2 ** num_bits  # number of quantization levels

    for name, param in network.named_parameters():
        # Only process float tensors that have quantization params (skip BN running stats, etc.)
        if name not in quant_params:
            continue
        if not isinstance(param, torch.Tensor) or not param.dtype.is_floating_point:
            continue

        # Get current quantization parameters for this weight
        R_max, R_min, s = quant_params[name]
        R_max_old = R_max
        R_min_old = R_min  # for clarity

        # Compute distribution statistics of current weights
        weight_data = param.data  # tensor of weights
        actual_max = weight_data.max().item()
        actual_min = weight_data.min().item()
        mean_val = weight_data.mean().item()
        total_elems = weight_data.numel()

        # Compute saturation percentages at the ends of the range
        # Define "near" R_max/R_min as within 5% of the current range
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
        # Condition 2: Adjust if weights saturate either or both ends
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
                max_up_shift = actual_min - R_min_old  # how much we can raise R_min without excluding the lowest weight
                if max_up_shift < 0:
                    max_up_shift = 0.0
                shift = min(shift, max_up_shift)
            elif shift < 0:  # shifting range down (decreasing R_min and R_max)
                max_down_shift = R_max_old - actual_max  # how much we can lower R_max without excluding the highest weight
                if max_down_shift < 0:
                    max_down_shift = 0.0
                shift = max(shift, -max_down_shift)
            # Apply the allowed shift
            R_max_new = R_max_old + shift
            R_min_new = R_min_old + shift
        # Condition 3: If not saturated but under-utilized range
        elif (actual_max - actual_min) <= 0.70 * range_width:
            # Contract range by 15%, without excluding any weights
            new_width = range_width * 0.85
            # Calculate slack on each end
            slack_high = R_max_old - actual_max
            slack_low = actual_min - R_min_old
            # Distribute the range reduction proportionally to slack
            reduction = range_width - new_width  # total amount to cut from the range
            # Avoid division by zero (if no slack, though in this branch slack_total should be >= reduction)
            slack_total = slack_high + slack_low if (slack_high + slack_low) != 0 else reduction
            # Portion to remove from each end
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
            new_s = 1e-8  # tiny non-zero to avoid division by zero
        # Align R_max_new and R_min_new to the quantization grid (optional for stability)
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


if __name__ == "__main__":
    sys.path.append(".")
    p = argparse.ArgumentParser()
    p = process_cli(p)
    args = p.parse_args()
    args.log = True
    # args.method = 'FO'
    # args.method = 'ZO_CGE_Q'
    args.method = 'ZO_RGE_Q'
    args.cnn_depth = 1
    args.num_bits = 4
    args.lr = 0.01
    args.epoch = 50
    main(args)