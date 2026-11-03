import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

try:
    from torchinfo import summary
except ImportError:
    def summary(model):
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

def main(args, device):
    xtrain, xtest, ytrain, y_test = load_data()
    class_names = [
        "Actinic keratosis",
        "Basal cell carcinoma",
        "Benign keratosis",
        "Dermatofibroma",
        "Melanoma",
        "Melanocytic nevus",
        "Vascular lesion"
    ]
    plot_dermamnist_samples(xtrain, ytrain, class_names, samples_per_class=5)

    if not args.test:
        n_samples = xtrain.shape[0]
        n_val = int(0.2 * n_samples)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        xval = xtrain[val_indices]
        yval = ytrain[val_indices]
        xtrain = xtrain[train_indices]
        ytrain = ytrain[train_indices]
        print(f"Training set: {xtrain.shape[0]} samples")
        print(f"Validation set: {xval.shape[0]} samples")
    else:
        print(f"Using full training set: {xtrain.shape[0]} samples")
        print(f"Test set: {xtest.shape[0]} samples")
        xval, yval = xtest, y_test

    n_classes = get_n_classes(ytrain)

    if args.nn_type == "mlp":
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xval = xval.reshape(xval.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
        mean = np.mean(xtrain, axis=0)
        std = np.std(xtrain, axis=0)
        std[std == 0] = 1e-8
        xtrain = normalize_fn(xtrain, mean, std)
        xval = normalize_fn(xval, mean, std)
        xtest = normalize_fn(xtest, mean, std)
        input_size = xtrain.shape[1]
        model = MLP(input_size=input_size, n_classes=n_classes)
    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape(-1, 28, 28, 3).transpose((0, 3, 1, 2))
        xval = xval.reshape(-1, 28, 28, 3).transpose((0, 3, 1, 2))
        xtest = xtest.reshape(-1, 28, 28, 3).transpose((0, 3, 1, 2))
        mean = xtrain.mean(axis=(0, 2, 3), keepdims=True)
        std = xtrain.std(axis=(0, 2, 3), keepdims=True)
        xtrain = (xtrain - mean) / (std + 1e-8)
        xval = (xval - mean) / (std + 1e-8)
        xtest = (xtest - mean) / (std + 1e-8)
        model = CNN(input_channels=3, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown network type: {args.nn_type}")
    
    model = model.to(device)
    print("Using device:", device)
    print("Model is on:", next(model.parameters()).device)

    print(f"Using {args.nn_type.upper()} model with {n_classes} output classes")
    summary(model)

    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=device)

    preds_train = method_obj.fit(xtrain, ytrain)

    if not args.test:
        preds = method_obj.predict(xval)
        eval_target = yval
        eval_name = "Validation"
    else:
        preds = method_obj.predict(xtest)
        eval_target = y_test
        eval_name = "Test"

    train_acc = accuracy_fn(preds_train, ytrain)
    train_macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {train_acc:.3f}% - F1-score = {train_macrof1:.6f}")

    eval_acc = accuracy_fn(preds, eval_target)
    eval_macrof1 = macrof1_fn(preds, eval_target)
    print(f"{eval_name} set: accuracy = {eval_acc:.3f}% - F1-score = {eval_macrof1:.6f}")

    if hasattr(method_obj, 'plot_training_progress') and hasattr(method_obj, 'train_losses'):
        print("\nPlotting training progress...")
        method_obj.plot_training_progress(method_obj.train_losses,
                                          method_obj.val_accuracies if hasattr(method_obj, 'val_accuracies') else None)

def plot_dermamnist_samples(x, y, class_names, samples_per_class=5):
    num_classes = len(class_names)
    fig, axes = plt.subplots(samples_per_class, num_classes, figsize=(num_classes * 2.5, samples_per_class * 2.5))
    for col in range(num_classes):
        class_indices = np.where(y == col)[0]
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        for row, img_idx in enumerate(selected_indices):
            ax = axes[row, col] if samples_per_class > 1 else axes[col]
            ax.imshow(x[img_idx])
            ax.axis("off")
            if row == 0:
                ax.set_title(class_names[col], fontsize=10, pad=6)
    plt.suptitle("DermaMNIST Sample Images by Class", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def timed_main(args, device):
    start = time.time()
    main(args, device)
    end = time.time()
    print(f"\nTotal runtime: {end - start:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp", help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    timed_main(args, device)
