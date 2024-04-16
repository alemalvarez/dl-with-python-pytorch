import numpy as np
from typing import Union
from torch.utils.data import DataLoader
import torch
import tensorflow as tf

import matplotlib.pyplot as plt

def tensor_info(tensor: object, name: str) -> None:
    """
    Print information about the tensor.

    Args:
        tensor (Union[np.ndarray, DataLoader, torch.Tensor, tf.Tensor]): The tensor to inspect.
        name (str): The name of the tensor.
    """
    print(f"==================== {name} | tensor_info ====================")
    print(f"Type: {type(tensor)}")

    # For DataLoaders:
    if isinstance(tensor, DataLoader):
        print(f"Batch size: {tensor.batch_size}")
        for X, y in tensor:
            print(f"Shape of features in first batch: {X.shape}")
            print(f"Shape of labels in first batch: {y.shape} {y.dtype}")
            break

    # For np arrays, torch tensors, and tensorflow tensors:
    if isinstance(tensor, (np.ndarray, torch.Tensor, tf.Tensor)):
        print(f"Shape: {tensor.shape} | nDims: {tensor.ndim} | dtype: {tensor.dtype}")
        if tensor.ndim == 0:  # Scalar
            print(f"Content: {tensor}")
        elif tensor.ndim == 1 or (tensor.ndim == 2 and tensor.shape[0] == 1):  # 1D or 1xN 2D
            print(f"Content: {tensor}")
        elif tensor.ndim == 2: # NxM 2D
            print(f"{name}[0]: {tensor[0]}")

    print("")

def draw_image(img: np.ndarray, label: str = None) -> None:
    """
    Display an image.

    Args:
        img (np.ndarray): The image to display.
        label (str, optional): The label for the image. Defaults to None.
    """
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def draw_grid(images: np.ndarray, labels: list = None) -> None:
    """
    Display a grid of images.

    Args:
        images (np.ndarray): The images to display in a grid.
        labels (list, optional): The labels for the images. Defaults to None.
    """
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        if labels is not None:
            plt.title(str(labels[i]))
        plt.axis("off")
        plt.imshow(images[i].squeeze(), cmap="gray")
    plt.show()

def batch_accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculate the accuracy of predicted batch.

    Args:
        y_hat (torch.Tensor): The predicted tensor.
        y (torch.Tensor): The target tensor.

    Returns:
        float: The accuracy of the predicted batch.
    """
    predicted = torch.argmax(y_hat, 1)
    correct = (predicted == y).sum().item()
    return correct / y.size(0)

def model_eval(model, test_tuple=None, test_dataloader=None, metrics=['accuracy']):
    """
    Evaluate the model on test data.
    
    Args:
        model: The model to evaluate.
        test_tuple (tuple): A tuple of (test_samples, test_labels) numpy arrays.
        test_dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader with test data.
        metrics (list): A list of metric names to evaluate. Default is ['accuracy'].
    """
    # Validate input
    if (test_tuple is None and test_dataloader is None):
        raise ValueError("You must provide either test_data or test_dataloader")

    # Keras model
    if isinstance(model, tf.keras.Model):
        print(f"==================== Evaluating Keras model '{model.name}' ====================")
        if test_tuple is None:
            raise ValueError("You must provide test_tuple")
        test_samples, test_labels = test_tuple
        test_metrics = model.evaluate(test_samples, test_labels, verbose=0)
        for metric, value in zip(model.metrics_names, test_metrics):
            print(f"{metric}: {value}")

    # PyTorch model
    # This code is AWFUL. Please somebody help.
    elif isinstance(model, torch.nn.Module):
        print(f"==================== Evaluating PyTorch model '{model._get_name()}' =================")
        model.eval()  # Set the model to evaluation mode

        if test_dataloader is not None:
            test_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for X, y in test_dataloader:
                    y_hat = model(X)
                    loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    test_loss += loss.item()
                    predicted = torch.argmax(y_hat, 1)
                    total_predictions += y.size(0)
                    correct_predictions += (predicted == y).sum().item()

            test_loss /= len(test_dataloader.dataset)
            test_accuracy = correct_predictions / total_predictions
            print(f"Test loss: {test_loss} | Test accuracy: {test_accuracy}")
        else:
            test_samples, test_labels = test_tuple
            test_samples = torch.from_numpy(test_samples)
            test_labels = torch.from_numpy(test_labels)
            y_hat = model(test_samples)
            loss = torch.nn.CrossEntropyLoss()(y_hat, test_labels)
            predicted = torch.argmax(y_hat, 1)
            test_accuracy = (predicted == test_labels).sum().item() / len(test_labels)
            print(f"Test loss: {loss.item()} | Test accuracy: {test_accuracy}")
    else:
        raise ValueError("model must be a Keras or PyTorch model")