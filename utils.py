import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import tensor

def tensor_info(tensor, name):
    print(f"Type of {name}: {type(tensor)}")
    # For np arrays:
    if(isinstance(tensor, np.ndarray)):
        print(f"Shape of {name}: {tensor.shape}")
        if tensor.ndim == 1:
            print(f"{name}: {tensor}")
        elif tensor.ndim == 2:
            print(f"{name}[0]: {tensor[0]}")
    # For DataLoaders:
    if(isinstance(tensor, DataLoader)):
        print(f"Batch size: {tensor.batch_size}")
        for X, y in tensor:
            print(f"Shape of features in first batch of {name}: {X.shape}")
            print(f"Shape of labels in first batch of {name}: {y.shape} {y.dtype}")
            break
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