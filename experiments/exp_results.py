import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from models.cnn import AdjustableCNN
from data import prepare_dataset
from cfg import results_path
from tqdm import tqdm
from tools import *
import re

def list_experiments(results_path):
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


def extract_metrics(log_file_path):
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    # Define regex patterns to extract accuracy and loss
    train_pattern = re.compile(r"Epoch \d+ Training: .*? Acc ([\d\.]+)% Loss ([\d\.]+)")
    test_pattern = re.compile(r"Epoch \d+ Testing:: .*? Acc ([\d\.]+)% Loss ([\d\.]+)")

    # Read log file and extract metrics
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            train_match = train_pattern.search(line)
            test_match = test_pattern.search(line)

            if train_match:
                train_acc.append(float(train_match.group(1)))
                train_loss.append(float(train_match.group(2)))

            if test_match:
                test_acc.append(float(test_match.group(1)))
                test_loss.append(float(test_match.group(2)))

    return train_acc, test_acc, train_loss, test_loss
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

def plot_results(train_acc, test_acc, train_loss, test_loss):
    # Generate epoch numbers
    epochs = list(range(1, len(train_acc) + 1))

    # Create plots
    plt.figure(figsize=(12, 6))

    # Train and Test Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, test_acc, label="Test Accuracy", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Test Accuracy vs Epoch")
    plt.legend()
    plt.grid()

    # Train and Test Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label="Train Loss", marker='o')
    plt.plot(epochs, test_loss, label="Test Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss vs Epoch")
    plt.legend()
    plt.grid()

    # Show plot
    plt.tight_layout()
    plt.show()
def evaluate_best_model(results_folder, dataset="cifar10"):
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


if __name__ == "__main__":
    results_folder = None

    if not results_folder:
        results_folder = list_experiments(results_path)

    log_dir = os.path.join(results_folder, "tensorboard")

    log_file_path = os.path.join(results_folder, "log.txt")

    train_acc, test_acc, train_loss, test_loss = extract_metrics(log_file_path)
    plot_results(train_acc, test_acc, train_loss, test_loss)

    # Load TensorBoard logs
    # epochs, train_acc, test_acc = load_tensorboard_logs(log_dir)

    # Plot accuracy
    # plot_accuracy(epochs, train_acc, test_acc, results_folder)
    #
    # Evaluate best model
    # evaluate_best_model(results_folder)
