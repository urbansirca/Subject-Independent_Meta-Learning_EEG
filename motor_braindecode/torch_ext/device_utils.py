import torch
import logging

def get_device(gpu_id=0):
    """
    Get the best available device (CUDA if available, otherwise CPU).
    
    Parameters
    ----------
    gpu_id : int
        GPU device ID to use if CUDA is available
        
    Returns
    -------
    device : torch.device
        The selected device
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        logging.info(f"Using CUDA device: {device}")
        return device
    else:
        device = torch.device('cpu')
        logging.info("CUDA not available, using CPU")
        return device

def to_device(tensor_or_module, device):
    """
    Move tensor or module to the specified device.
    
    Parameters
    ----------
    tensor_or_module : torch.Tensor or torch.nn.Module
        The tensor or module to move
    device : torch.device
        The target device
        
    Returns
    -------
    torch.Tensor or torch.nn.Module
        The tensor or module moved to the target device
    """
    return tensor_or_module.to(device)

def set_cuda_device_safely(gpu_id=0):
    """
    Safely set CUDA device if available, otherwise do nothing.
    
    Parameters
    ----------
    gpu_id : int
        GPU device ID to use if CUDA is available
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        logging.info(f"Set CUDA device to {gpu_id}")
    else:
        logging.info("CUDA not available, skipping device setting")

def set_random_seeds_safe(seed, gpu_id=0):
    """
    Set random seeds safely, handling both CUDA and CPU cases.
    
    Parameters
    ----------
    seed : int
        Random seed to set
    gpu_id : int
        GPU device ID to use if CUDA is available
    """
    import random
    import numpy as np
    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        logging.info(f"Set CUDA seeds for device {gpu_id}")
    else:
        logging.info("CUDA not available, skipping CUDA seed setting")
    
    # Set deterministic behavior if CUDA is available
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
