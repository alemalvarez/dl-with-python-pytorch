import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import tensorflow.keras as keras
import tensorflow as tf


from keras.src.callbacks import History

# Utils file with some useful helper functions when working with tensors and models.
# They are nice to have, specially if you're starting.

def make_dataloader(X_list: list, y_list: list, batch_size: int = 32, shuffle: bool = True, X_type="float32", y_type="float32") -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader from a list of samples and labels.

    Args:
        X (list): A list of samples.
        y (list): A list of labels.
        batch_size (int, optional): The batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        X_type (str, optional): The type of the samples. Defaults to "float32".
        y_type (str, optional): The type of the labels. Defaults to "float32".

    Returns:
        torch.utils.data.DataLoader: A PyTorch DataLoader.
    """
    for X, y in zip(X_list, y_list):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        dataset = []
        for i in range(len(X)):
            dataset.append((X[i].astype(X_type), y[i].astype(y_type)))
        
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def tensor_info(tensor: object, name: str, decorator=True, image=True) -> None:
    """
    Print information about the tensor.

    Args:
        tensor (Union[np.ndarray, DataLoader, torch.Tensor, tf.Tensor]): The tensor to inspect.
        name (str): The name of the tensor.
    """
    if decorator:
        print(f"==================== {name} | tensor_info ====================")
    print(f"Type: {type(tensor)}")

    # For DataLoaders:
    if isinstance(tensor, torch.utils.data.DataLoader):
        print(f"Batch size: {tensor.batch_size} | N batches: {len(tensor)} ")
        X, y = next(iter(tensor))
        print("-- First batch of features")
        tensor_info(X, "features", decorator=False, image=image)
        print("-- First batch of labels")
        tensor_info(y, "labels", decorator=False, image=image)

    # For np arrays, torch tensors, and tensorflow tensors:
    if isinstance(tensor, (np.ndarray, torch.Tensor, tf.Tensor)):
        print(f"Shape: {tensor.shape} | nDims: {tensor.ndim} | dtype: {tensor.dtype}")
        if tensor.ndim == 0:  # Scalar
            print(f"Content: {tensor}")
        elif tensor.ndim == 1 or (tensor.ndim == 2 and (tensor.shape[0] == 1 or tensor.shape[1] == 1)):  # 1D or 1xN 2D
            if isinstance(tensor, torch.Tensor):
                print(f"Content: {torch.flatten(tensor[:20])}")
            else:
                print(f"Content: {np.ndarray.flatten(tensor[:20])}")
            if(len(tensor) > 20):
                print("... (showing only the first 20 elements)")
        elif tensor.ndim == 2: # NxM 2D
            print(f"{name}[0]: {tensor[0][:20]}")
            if(len(tensor[0]) > 20):
                print("... (showing only the first 20 elements)")
            if image:
                draw_image(np.array(tensor), label=name, size=1)
        elif tensor.ndim == 3: # NxMxP 3D
            if tensor.shape[1] == tensor.shape[2]:  # Square! image?
                if image:
                    draw_image(tensor[0], label=f"{name}[0]", size=1)
        elif tensor.ndim == 4: # NxMxPxQ 4D
            if tensor.shape[2] == tensor.shape[3]:  # Square! image batch?
                if image: 
                    draw_image(tensor[0][0], label=f"{name}[0]", size=1)

    # For torch Tensors:
    if isinstance(tensor, torch.Tensor):
        if(tensor.requires_grad and tensor.grad is not None):
            print("-- Gradient:")
            tensor_info(tensor.grad, "gradient", decorator=False, image=False)

    # For tensorflow Variables:
    if isinstance(tensor, tf.Variable):
        print(f"Shape: {tensor.shape} | dtype: {tensor.dtype}")
        print(f"Content: {tensor.numpy()}")
    print("")

def draw_image(img: np.ndarray, label: str = None, size: int = None) -> None:
    """
    Display an image.

    Args:
        img (np.ndarray): The image to display.
        label (str, optional): The label for the image. Defaults to None.
    """
    if size is not None:
        plt.figure(figsize=(size, size))
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)
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
            plt.title(str(labels[i-1]))
        plt.axis("off")
        plt.imshow(images[i-1].squeeze(), cmap=plt.cm.binary)
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
    
def function_race(functions: list, inputs: list):
    """
    Executes a race between multiple functions, measuring their execution time.

    Args:
        functions (list): A list of functions to race.
        inputs (list): A list of inputs to pass to each function.

    Returns:
        None: This function does not return any value.

    Raises:
        Exception: If any of the functions raise an exception during execution.

    """
    for function in functions:
        try:
            t0 = time.time()
            print(f"Trying {function.__name__}")
            output = function(*inputs)
            print(f"{function.__name__} succeeded")
            print("Took: {0:.5f} s\n".format(time.time() - t0))
        except Exception as e:
            print(f"{function.__name__} failed: {e}")

def draw_training_history(history: History):
    """
    Draw the training history of a Keras model.

    Args:
        history (keras.src.callbacks.History): The history object of a Keras model.
    """
    fig, axes = plt.subplots(1, len(history.history), figsize=(20, 5))
    fig.suptitle(history.model.name, fontsize=16)
    for i, key in enumerate(history.history.keys()):
        axes[i].plot(history.history[key])
        axes[i].set_title(key)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(key)
    plt.show()

def keras_comparer (models: list, X_train, y_train, X_test, y_test, val_data=None, epochs=10, batch_size=32, ):
    """
    Compare multiple Keras models.

    Args:
        models (list): A list of Keras models to compare.
    """
    print(f"==================== Comparing {len(models)} Keras models ====================")
    print(f"Epochs: {epochs} | Batch size: {batch_size}")
    for model in models:
        # Check if a model has been compiled
        if not model.optimizer:
            raise ValueError(f"Model '{model.name}' has not been compiled")
        t0 = time.time()
        model.build(X_train.shape)
        print(f"\nTraining model '{model.name}' with {model.count_params()} parameters")
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            verbose=0
        )
        print(f"Training took: {time.time() - t0:.5f} s")
        draw_training_history(model.history)
        results = model.evaluate(X_test, y_test, verbose=0)
        print(f"Testing evaluation for model '{model.name}':")
        for metric, value in zip(model.metrics_names, results):
            print(f"{metric}: {value}")