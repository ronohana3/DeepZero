import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from models.cnn import AdjustableCNN
from data import prepare_dataset
from cfg import results_path
from tqdm import tqdm
from tools import *

def list_experiments(results_path):
    """List all experiment folders inside results_path and let the user select one."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results directory does not exist: {results_path}")

    experiments = [f for f in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, f))]

    if not experiments:
        raise FileNotFoundError("No experiment folders found in results path.")

    print("\nAvailable Experiments:")
    for idx, exp in enumerate(experiments):
        print(f"{idx + 1}. {exp}")

    while True:
        try:
            selection = int(input("\nSelect an experiment by number: ")) - 1
            if 0 <= selection < len(experiments):
                return os.path.join(results_path, experiments[selection])
            else:
                print("Invalid selection. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def load_tensorboard_logs(log_dir):
    """Load training and testing accuracy logs from TensorBoard files."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    train_acc, test_acc, epochs = [], [], []
    # Extract training accuracy
    if 'train/acc' in event_acc.Tags()['scalars']:
        for event in event_acc.Scalars('train/acc'):
            epochs.append(event.step)
            train_acc.append(event.value)

    # Extract testing accuracy
    if 'test/acc' in event_acc.Tags()['scalars']:
        for event in event_acc.Scalars('test/acc'):
            test_acc.append(event.value)

    return np.array(epochs), np.array(train_acc), np.array(test_acc)


def plot_accuracy(epochs, train_acc, test_acc, save_path):
    """Plot training and testing accuracy over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, test_acc, label="Test Accuracy", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs. Testing Accuracy")
    plt.grid()
    plt.savefig(os.path.join(save_path, "accuracy_plot.png"))
    plt.show()


def evaluate_best_model(results_folder, dataset="cifar10"):
    """Load the best model and evaluate it on the test dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model arguments (depth, channels)
    args_file = os.path.join(results_folder, "args.txt")
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Arguments file not found: {args_file}")

    args_dict = {}
    with open(args_file, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            args_dict[key] = int(value) if value.isdigit() else value

    depth = int(args_dict.get("cnn_depth", 1))
    channel_num = int(args_dict.get("cnn_channel_num", 24))

    # Load dataset
    loaders, class_num = prepare_dataset(dataset, batch_size=args_dict.get("batch_size", 256))

    # Initialize model
    model = AdjustableCNN(depth=depth, channel_num=channel_num).to(device)

    best_model_path = os.path.join(results_folder, "best.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model file not found: {best_model_path}")

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict["state_dicts"]["network"])

    model.eval()
    pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing:", ncols=120)
    acc = AverageMeter()
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            fx = model(x)
        acc.update(torch.argmax(fx, 1).eq(y).float().mean(), y.size(0))
        pbar.set_postfix_str(f"Acc {100 * acc.avg:.2f}%")


def plot_weights(results_folder):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model arguments (depth, channels)
    args_file = os.path.join(results_folder, "args.txt")
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Arguments file not found: {args_file}")

    args_dict = {}
    with open(args_file, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            args_dict[key] = int(value) if value.isdigit() else value

    depth = int(args_dict.get("cnn_depth", 1))
    channel_num = int(args_dict.get("cnn_channel_num", 24))

    # Initialize model
    model = AdjustableCNN(depth=depth, channel_num=channel_num).to(device)

    last_model_path = os.path.join(results_folder, "ckpt.pth")
    if not os.path.exists(last_model_path):
        raise FileNotFoundError(f"Last model file not found: {last_model_path}")

    state_dict = torch.load(last_model_path, map_location=device)
    model.load_state_dict(state_dict["state_dicts"]["network"])

    print_weight_frequencies(model)


def get_weight_frequencies(network):
    freq_dict = {}

    for name, param in network.named_parameters():
        # Count occurrences of each quantized value
        unique_values, counts = torch.unique(param.cpu().round(decimals=4), return_counts=True)
        freq_dict[name] = dict(zip(unique_values.tolist(), counts.tolist()))

    return freq_dict


def print_weight_frequencies(network):
    freq_dict = get_weight_frequencies(network)

    print("\n==== Weight Value Frequencies per Layer ====")
    for layer, freqs in freq_dict.items():
        freq_str = " | ".join(f"{value:.3f}: {count}" for value, count in sorted(freqs.items()))
        print(f"{layer}: {freq_str}")

    for layer, freqs in freq_dict.items():
        value, count = zip(*sorted(freqs.items()))
        plt.title(layer)
        plt.stem(value, count)
        plt.show()



if __name__ == "__main__":
    results_folder = None  # Set to None so user selects experiment

    # If no folder is provided, ask user to select one
    if not results_folder:
        results_folder = list_experiments(results_path)

    # plot_weights(results_folder)

    log_dir = os.path.join(results_folder, "tensorboard")

    # Load TensorBoard logs
    # epochs, train_acc, test_acc = load_tensorboard_logs(log_dir)

    # Plot accuracy
    # plot_accuracy(epochs, train_acc, test_acc, results_folder)
    #
    # Evaluate best model
    evaluate_best_model(results_folder)
