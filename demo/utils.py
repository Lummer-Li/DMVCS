import os
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F

def get_logger(file_name, data_name):
    """
    Create and configure a logger with both file and console outputs
    
    Args:
        file_name (str): Name identifier for the logger (typically __name__)
        data_name (str): Base name for the log file (appended with .log)
        
    Returns:
        logging.Logger: Configured logger instance with:
            - File handler writing to ./logs/{data_name}.log
            - Console stream handler
            - Standard log format
    """
    # Initialize logger with specified name
    logger = logging.getLogger(file_name)
    # Set logging threshold to INFO level
    logger.setLevel(logging.INFO)
    
    # Configure log file path and handler
    filename = "./logs/" + data_name + ".log"  # Log directory: ./logs/
    # filename = data_name + ".log"  # Alternate location (commented out)
    
    # Create file handler with INFO level
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    
    # Define log message format:
    # Timestamp - Logger Name - Log Level - Message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Create console output handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)  # Use same format as file
    
    # Attach handlers to logger
    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger

def set_seed(seed=42):
    """Initialize all random number generators with specified seed
    
    Ensures reproducibility across:
    - NumPy computations
    - PyTorch CPU/CUDA operations
    - Python built-in random module
    - Hash-based operations
    
    Args:
        seed (int): Random seed value (default=42)
    
    Note:
        Setting `cudnn.deterministic=True` may impact performance
        but is necessary for reproducible GPU results
    """
    # NumPy random number generation
    np.random.seed(seed)
    
    # PyTorch CPU random states
    torch.manual_seed(seed)
    
    # PyTorch CUDA random states
    torch.cuda.manual_seed(seed)          # Current GPU
    torch.cuda.manual_seed_all(seed)      # All GPUs (multi-GPU setups)
    
    # CUDA convolution optimization settings
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner
    torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
    
    # Python built-in random module
    random.seed(seed)
    
    # Environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)  # Prevent hash randomization

def save_model(state, dataset_name):
    """
    Save model state dictionary to specified path with dataset-based naming
    
    Parameters
    ----------
    state : dict
        Model state dictionary containing:
        - 'model_state_dict': Model parameters
        - 'optimizer_state_dict': Optimizer state (optional)
        - 'epoch': Training epoch (optional)
    dataset_name : str
        Identifier for model versioning, used in filename
    
    Raises
    ------
    PermissionError
        If lacking write permissions for model directory
    IOError
        If disk space insufficient or path invalid
    
    Notes
    -----
    - Automatically creates './models' directory if non-existent
    - Uses PyTorch's serialization format (.pth)
    - Recommended to include training metadata in state
    """
    # Create model directory if not exists
    if not os.path.exists('./models'):
        os.makedirs('./models')  # Recursive directory creation
    
    # Construct filesystem path
    save_path = os.path.join('./models', f'{dataset_name}.pth')
    
    # Serialize model state
    torch.save(state, save_path)  # Uses pickle protocol
    
    # User feedback
    print(f'Model checkpoint saved at: {save_path}')



def centroids_func(class_num, x, cluster_matrix, ratio=1.0):
    """Calculate cluster centroids using top confidence samples.
    
    Args:
        class_num: Number of target classes
        x: Input feature matrix [num_samples, feature_dim]
        cluster_matrix: Cluster confidence scores [num_samples, class_num]
        ratio: Percentage of top samples to use (default: 1.0)
    
    Returns:
        Centroids matrix [class_num, feature_dim]
    """
    # Select top-k most confident samples per class
    topk = int(cluster_matrix.shape[0] / class_num * ratio)
    topk_values, _ = torch.topk(cluster_matrix, k=topk, dim=0, largest=True)

    # Create threshold condition using the last value in topk results
    condition = topk_values[-1, :] if len(topk_values) > 0 else cluster_matrix[0]

    # Create mask for high-confidence samples (others zeroed out)
    high_confidence = torch.where(
        cluster_matrix >= condition,
        cluster_matrix,
        torch.tensor(0.0, device=cluster_matrix.device, dtype=cluster_matrix.dtype)
    )

    # Convert confidence scores to probability distribution per class
    indicate_matrix = high_confidence / high_confidence.sum(dim=0)

    # Calculate weighted centroids using confidence scores
    centroids = torch.matmul(indicate_matrix.T, x)
    return centroids

def get_centroids(args, class_num, x, cluster_matrix):
    """Combine centroids calculated with different sample ratios.
    
    Args:
        args: Configuration object containing ratios list
        class_num: Number of target classes
        x: Input feature matrix
        cluster_matrix: Cluster confidence scores
    
    Returns:
        Averaged centroids matrix [class_num, feature_dim]
    """
    # Start with base centroids (ratio=1.0)
    centroids = centroids_func(class_num, x, cluster_matrix)
    
    # Add centroids from different sample ratios
    for ratio in args.ratios:
        centroids += centroids_func(class_num, x, cluster_matrix, ratio=ratio)
    
    # Average all centroids for final result
    centroids = centroids / (len(args.ratios) + 1)
    return centroids

def get_sim(centroids):
    """Calculate similarity structure between centroids.
    
    Args:
        centroids: Input centroid matrix [class_num, feature_dim]
    
    Returns:
        Softmax-normalized similarity matrix [class_num, class_num]
    """
    # Calculate pairwise similarity scores
    structure = torch.mm(centroids, centroids.T)
    
    # Convert to probability distribution per row
    return F.softmax(structure, dim=1)
