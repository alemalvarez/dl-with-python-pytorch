import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch
import tensorflow as tf

def tensor_info(tensor, name):
    print(f"==================== {name} tensor info ====================")
    print(f"Type: {type(tensor)}")
    # For np arrays:
    if(isinstance(tensor, np.ndarray)):
        print(f"Shape: {tensor.shape} | nDim: {tensor.ndim}")
        if tensor.ndim == 0: # Scalar
            print(f"Content: {tensor}")
        elif tensor.ndim == 1 or (tensor.ndim == 2 and tensor.shape[0] == 1): # 1D or 1xN 2D
            print(f"Content: {tensor}")
        elif tensor.ndim == 2: # NxM 2D
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
        if tensor.ndim == 0: # Scalar
            print(f"Content: {tensor}")
        elif tensor.ndim == 1 or tensor.shape[0] == 1: # 1D or 1xN 2D
            print(f"Content: {tensor}")
    # For Tensorflow tensors:
    if(isinstance(tensor, tf.Tensor)):
        print(f"Shape: {tensor.shape}")
        if tensor.ndim == 0: # Scalar
            print(f"Content: {tensor}")
        elif tensor.ndim == 1 or tensor.shape[0] == 1: # 1D or 1xN 2D
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

def model_eval (model, test_samples, test_labels):
    # Keras model:
    if isinstance(model, tf.keras.Model):
        print(f"==================== Evaluating Keras model '{model.name}' ====================")
        test_loss, test_acc = model.evaluate(test_samples, test_labels, verbose=0)
        print(f"Test loss: {test_loss} | Test accuracy: {test_acc}")
        
    # Pytorch model:
    if isinstance(model, torch.nn.Module):
        print(f"==================== Evaluating Pytorch model '{model._get_name()}' =================")
        model.eval()  # Set the model to evaluation mode

        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for X, y in test_samples:
                y_hat = model(X)
                loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                test_loss += loss.item()
                predicted = torch.argmax(y_hat, 1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()

        test_loss /= len(test_samples.dataset)
        test_accuracy = correct_predictions / total_predictions
        print(f"Test loss: {test_loss} | Test accuracy: {test_accuracy}")