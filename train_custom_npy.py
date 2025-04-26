import argparse
import logging
import os
from pathlib import Path

import torch

from configs.custom_npy import custom_npy_configs
from train import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model with a custom NPY dataset"
    )
    parser.add_argument(
        "--npy_file",
        type=str,
        required=True,
        help="Path to the .npy file containing your custom dataset",
    )
    parser.add_argument(
        "--test_npy_file",
        type=str,
        default=None,
        help="Path to the .npy file containing your test dataset (optional)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        help="Working directory for checkpoints and logs",
    )
    return parser.parse_args()


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    args = parse_args()

    # Check if NPY file exists
    if not os.path.exists(args.npy_file):
        logger.error(f"NPY file {args.npy_file} does not exist!")
        return

    # Check if test NPY file exists if provided
    if args.test_npy_file and not os.path.exists(args.test_npy_file):
        logger.error(f"Test NPY file {args.test_npy_file} does not exist!")
        return

    # Create working directory
    workdir = args.workdir
    Path(workdir).mkdir(parents=True, exist_ok=True)

    # Get configuration
    config = custom_npy_configs.get_config(args.npy_file, args.test_npy_file)

    # Set device
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Log configuration
    logger.info(f"Training with configuration: {config}")
    logger.info(f"Using device: {config.device}")

    # Start training
    logger.info("Starting training...")
    train(config, workdir)
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
