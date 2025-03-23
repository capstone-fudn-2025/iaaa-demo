import random
import numpy as np
import torch
import warnings


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')