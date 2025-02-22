import numpy as np
import torch
import random
import os


def process_cli(parser):
    parser.add_argument('--dry-run', action='store_false', dest='log')
    parser.add_argument('--seed', type=int, default=324823217)
    parser.add_argument('--network', choices=['resnet20', 'cnn'], default='cnn')
    parser.add_argument('--dataset', choices=['cifar10'], default='cifar10')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--cnn-depth', type=int, default=1)
    parser.add_argument('--cnn-channel-num', type=int, default=18)
    parser.add_argument('--method', type=str, choices=['FO', 'ZO_CGE', 'ZO_CGE_Q', 'ZO_RGE_Q'], default='ZO_CGE')
    parser.add_argument('--num-bits', type=int, default=8)
    parser.add_argument('--zoo-step-size', type=float, default=5e-3)

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'step'], default='cosine')

    return parser

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def gen_folder_name(args, ignore=('log')):
    def get_attr(inst, arg):
        value = getattr(inst, arg)
        if isinstance(value, float):
            return f"{value:.8f}"
        else:
            return value
    folder_name = ''
    for arg in vars(args):
        if arg in ignore:
            continue
        folder_name += f'{arg}_{get_attr(args, arg)}__'
    return folder_name[:-1]


def save_args_to_file(args, results_folder):
    args_path = os.path.join(results_folder, "args.txt")
    with open(args_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def override_func(inst, func, func_name):
    bound_method = func.__get__(inst, inst.__class__)
    setattr(inst, func_name, bound_method)
