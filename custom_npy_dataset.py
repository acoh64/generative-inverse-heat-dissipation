import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class NpyDataset(Dataset):
    """
    Dataset class for loading data from .npy files
    """

    def __init__(self, npy_file, transform=None):
        """
        Args:
            npy_file (str): Path to the .npy file containing the data
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Convert to float and normalize to [0,1] if needed
        if sample.dtype == np.uint8:
            sample = sample.astype(np.float32) / 255.0

        # Convert to torch tensor
        if self.transform:
            sample = self.transform(sample)
        else:
            # Handle different input formats
            if len(sample.shape) == 2:
                # Add channel dimension for grayscale images
                sample = np.expand_dims(sample, 0)
            elif len(sample.shape) == 3:
                # Check if already in CHW format
                if sample.shape[0] == 1 or sample.shape[0] == 3:
                    # Already in CHW format, keep as is
                    pass
                elif sample.shape[2] == 1 or sample.shape[2] == 3:
                    # In HWC format, transpose to CHW
                    sample = np.transpose(sample, (2, 0, 1))

            sample = torch.from_numpy(sample).float()

        # Return sample with empty dict to match the format of the original dataset
        return sample, {}


def get_npy_dataloaders(
    npy_train_file,
    npy_test_file=None,
    config=None,
    transform=None,
    train_batch_size=None,
    eval_batch_size=None,
):
    """
    Creates dataloaders from .npy files

    Args:
        npy_train_file (str): Path to the .npy file containing training data
        npy_test_file (str, optional): Path to the .npy file containing test data
                                       If None, uses the training data for testing too
        config: Configuration object (for batch sizes)
        transform: Optional transforms to apply
        train_batch_size (int, optional): Override config batch size
        eval_batch_size (int, optional): Override config eval batch size

    Returns:
        train_dataloader, test_dataloader
    """
    if config is not None:
        if not train_batch_size:
            train_batch_size = config.training.batch_size
        if not eval_batch_size:
            eval_batch_size = config.eval.batch_size
    else:
        if not train_batch_size:
            train_batch_size = 128
        if not eval_batch_size:
            eval_batch_size = 256

    # Create training dataset
    train_dataset = NpyDataset(npy_train_file, transform)

    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Create test dataset (either from separate file or reuse training data)
    if npy_test_file:
        test_dataset = NpyDataset(npy_test_file, transform)
    else:
        test_dataset = train_dataset

    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
