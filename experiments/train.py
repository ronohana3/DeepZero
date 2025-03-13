import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from algorithm.zoo import cge, zo_cge_qo, zo_rge_qo
from models.cnn import AdjustableCNN
from cfg import results_path
from tools import *
from data.prepare_data import prepare_dataset


def main(args):
    # Misc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    save_path = os.path.join(results_path, gen_folder_name(args, ignore=['log', 'dataset', 'seed', 'network', 'cnn_channel_num', 'nesterov','scheduler']))

    # Data
    loaders, class_num = prepare_dataset(args.dataset, args.batch_size)

    # Network
    if args.network == "cnn":
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
    from_scratch = True
    if args.log:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=False)
            best_acc = 0.
            epoch = 0
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
            from_scratch = False
        else:
            best_acc = 0.
            epoch = 0

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

        # if args.method in ['ZO_CGE_Q', 'ZO_RGE_Q'] and epoch in [50]:
        #     quant_params = update_quant_params(network, quant_params, args.num_bits)

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p = process_cli(p)
    args = p.parse_args()
    main(args)