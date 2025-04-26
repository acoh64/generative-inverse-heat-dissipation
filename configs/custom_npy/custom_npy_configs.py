import ml_collections
import numpy as np
import torch

from configs.mnist import default_mnist_configs


def get_config(npy_file_path, test_npy_file_path=None):
    """
    Returns a configuration for training on a custom .npy dataset
    using the same parameters as the default MNIST training.

    Args:
        npy_file_path (str): Path to the .npy file containing training data
        test_npy_file_path (str, optional): Path to the .npy file containing test data

    Returns:
        config: Configuration object
    """
    # Get the default MNIST config
    config = default_mnist_configs.get_default_configs()

    # Override with custom dataset information
    config.data.dataset = "CUSTOM_NPY"
    config.data.npy_file_path = npy_file_path
    config.data.test_npy_file_path = test_npy_file_path

    # Update image size to match the 32x32 dimensions
    config.data.image_size = 32

    config.model.model_channels = 64
    config.model.num_res_blocks = 1

    # Return the config
    return config
