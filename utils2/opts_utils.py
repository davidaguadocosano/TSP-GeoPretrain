import torch
import random
import argparse
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set the seed for various random number generators for reproducibility.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v: str) -> bool:
    """
    Transform string inputs into boolean.

    Args:
        v (str): The string input.

    Returns:
        bool: The string input transformed to boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    