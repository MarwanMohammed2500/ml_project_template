import numpy as np
import random
import torch

def set_seed(seed:int=42):
    """
    sets the seed in all needed libraries (torch, torch.cuda, numpy, and random for general python code)

    Arguments:
    seed:int=42, the seed
    """
    np.random.seed(seed) # Sets NumPy random seed
    torch.manual_seed(seed) # type:ignore - Sets PyTorch's random seed
    torch.cuda.manual_seed(seed) # Sets PyTorch's random seed on CUDA
    torch.cuda.manual_seed_all(seed) # Sets PyTorch's random seed on all objects on the GPU
    random.seed(seed) # Sets the random seed on all python objects