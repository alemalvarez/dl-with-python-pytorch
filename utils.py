import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch

def tensor_info(tensor, name):
    print(f"==================== {name} tensor info ====================")
    print(f"Type: {type(tensor)}")
    # For np arrays:
    if(isinstance(tensor, np.ndarray)):
        print(f"Shape: {tensor.shape}")
        if tensor.ndim == 1 or tensor.shape[0] == 1:
            print(f"Content: {tensor}")
        elif tensor.ndim == 2:
            print(f"{name}[0]: {tensor[0]}")
    # For DataLoaders:
    if(isinstance(tensor, DataLoader)):
        print(f"Batch size: {tensor.batch_size}")
        for X, y in tensor:
            print(f"Shape of features in first batch: {X.shape}")
            print(f"Shape of labels in first batch: {y.shape} {y.dtype}")
            break
    # For torch tensors:
    if(isinstance(tensor, torch.Tensor)):
        print(f"Shape: {tensor.shape}")
        if tensor.ndim == 1 or tensor.shape[0] == 1:
            print(f"Content: {tensor}")
    print("")

def draw_image(img, label=None):
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def draw_grid(images, labels=None):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        if labels is not None:
            plt.title(str(labels[i]))
        plt.axis("off")
        plt.imshow(images[i].squeeze(), cmap="gray")
    plt.show()

def batch_accuracy(y_hat, y):
    predicted = torch.argmax(y_hat, 1)
    correct = (predicted == y).sum().item()
    return correct / y.size(0)
