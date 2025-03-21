import scipy
import torch 
import sklearn
import numpy as np
from typing import Tuple, List
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

def get_mask(view_num, data_len, missing_rate):
    """
    Generates a binary mask matrix for multi-view data with controlled missing rates.

    Ensures each sample has at least one view present while approximating the specified
    overall missing rate across all views.

    Parameters
    ----------
    view_num : int
        Number of views/modalities
    data_len : int
        Number of data samples
    missing_rate : float
        Target proportion of missing data entries (0 ≤ missing_rate < 1). Higher values
        indicate more missing data.

    Returns
    -------
    numpy.ndarray
        Binary mask matrix where 1 indicates present data and 0 indicates missing

    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    # Validate inputs
    if not isinstance(view_num, int) or view_num < 1:
        raise ValueError("view_num must be a positive integer")
    if not isinstance(data_len, int) or data_len < 1:
        raise ValueError("data_len must be a positive integer")
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1)")
    
    # Calculate maximum achievable missing rate
    max_missing = (view_num - 1) / view_num
    if missing_rate > max_missing:
        missing_rate = max_missing  # Clamp to maximum possible

    # Set random seeds for reproducibility
    np.random.seed(0)
    one_rate = 1.0 - missing_rate  # Target presence rate

    # Handle edge case where maximum missing is required
    if one_rate <= 1/view_num:
        enc = OneHotEncoder(categories=[range(view_num)], sparse_output=False)
        return enc.fit_transform(np.random.randint(view_num, size=(data_len, 1))).astype(np.int64)

    # Handle complete data case
    if missing_rate == 0:
        return np.ones((data_len, view_num), dtype=np.int64)

    # Iterative optimization to achieve target missing rate
    error = 1
    tolerance = 0.005  # Allowed error in presence rate
    enc = OneHotEncoder(categories=[range(view_num)], sparse_output=False)
    
    while error > tolerance:
        # Ensure at least one view per sample
        view_preserve = enc.fit_transform(np.random.randint(view_num, size=(data_len, 1)))
        
        # Calculate required additional present entries
        target_present = int(view_num * data_len * one_rate)
        current_present = data_len  # From view_preserve
        needed_presents = target_present - current_present

        # Calculate probability for additional entries (accounting for overlap)
        possible_slots = view_num * data_len - data_len  # Available slots
        prob = needed_presents / possible_slots if possible_slots > 0 else 0

        # Generate additional presence indicators
        additional = np.random.rand(data_len, view_num) < prob
        additional = additional.astype(np.int64) * (1 - view_preserve)  # Prevent overlap

        # Combine matrices and clip values
        matrix = np.clip(view_preserve + additional, 0, 1)
        
        # Calculate actual presence rate
        actual_present = matrix.sum()
        if target_present == 0:
            error = 0
        else:
            error = abs(actual_present - target_present) / target_present

    return matrix.astype(np.int64)

path = '../../../datasets/'
def loadData(data_name):
    """
    Load multi-view dataset from .mat file with consistent data structure
    
    Parameters:
        data_name (str): Path to .mat file containing dataset
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - features: NumPy object array of shape (1, n_views) containing view data
            - ground_truth: Flattened integer array of shape (n_samples,)
            
    Raises:
        ValueError: If dataset format is not recognized or required fields are missing
    """
    dataset_config = {
        'ALOI': {
            'feature_base': ('X', 4),
            'n_views': 4,
            'label_key': 'Y',
            'label_process': ['squeeze']
        },
        'Caltech-5V': {
            'feature_keys': ['X1', 'X2', 'X3', 'X4', 'X5'],
            'n_views': 5,
            'label_key': 'Y',
            'label_process': ['squeeze']
        },
        'uci-digit': {
            'feature_keys': ['mfeat_fac', 'mfeat_fou', 'mfeat_kar'],
            'n_views': 3,
            'label_key': 'truth',
            'label_process': ['squeeze']
        },
        'handwritten': {
            'feature_base': ('X', 6),
            'n_views': 6,
            'label_key': 'Y',
            'transpose': True,
            'label_process': ['squeeze']
        },
        'Mfeat': {
            'feature_base': ('X', 6),
            'n_views': 6,
            'label_key': 'Y',
            'transpose': True,
            'label_process': ['squeeze']
        },
        '100leaves': {
            'feature_base': ('X', 3),
            'n_views': 3,
            'label_key': 'Y',
            'label_process': ['squeeze']
        },
        'ALOI100': {
            'feature_base': ('X', 4),
            'n_views': 4,
            'label_key': 'Y',
            'label_process': ['squeeze']
        },
    }

    # Identify dataset type
    dataset_type = next((k for k in dataset_config if k in data_name), None)
    if not dataset_type:
        raise ValueError(f"Unsupported dataset format: {data_name}")

    config = dataset_config[dataset_type]
    data = scipy.io.loadmat(data_name)
    
    # Feature extraction
    features = np.empty((1, config['n_views']), dtype=object)
    
    if 'feature_base' in config:
        base_key, n_views = config['feature_base']
        for i in range(n_views):
            if config.get('transpose', False):
                features[0][i] = data[base_key][i][0].astype(np.float32)
            else:
                features[0][i] = data[base_key][0][i].astype(np.float32)
    elif 'feature_keys' in config:
        for i, key in enumerate(config['feature_keys']):
            features[0][i] = data[key].astype(np.float32)
    else:
        features[0][0] = data[config['features_key']].astype(np.float32)
        if config.get('transpose', False):
            features[0][0] = features[0][0].T

    # Ground truth processing
    if config['label_key'] not in data:
        raise ValueError(f"Missing ground truth key: {config['label_key']}")

    gnd = data[config['label_key']].astype(np.int32)
    
    for operation in config['label_process']:
        if operation[0] == 'squeeze':
            gnd = np.squeeze(gnd)
        elif operation[0] == 'reshape':
            gnd = gnd.reshape(operation[1])

    return features, gnd.flatten()

class MultiViewDataset(Dataset):
    def __init__(self, dataname: str, missing_rate: float):
        """
        Multi-view dataset loader with missing view handling
        
        Args:
            dataname: Name/path of the dataset (without .mat extension)
            missing_rate: Proportion of missing views (0-1)
        """
        # Load and preprocess data
        features, gnd = loadData(path + dataname + '.mat')
        self.num_views = features.shape[1]
        self.num_samples = len(gnd)
        
        # Convert features to tensors and normalize
        self.features = [
            torch.from_numpy(
                sklearn.preprocessing.MinMaxScaler().fit_transform(view)
            ).float()
            for view in features[0]
        ]
        
        # Store ground truth and indices
        self.gnd = torch.from_numpy(gnd).long()
        self.indices = torch.arange(self.num_samples)
        
        # Generate missing view mask
        self.mask = torch.from_numpy(
            get_mask(self.num_views, self.num_samples, missing_rate)
        ).float()
        
        # Validate dimensions
        self._validate_shapes()

    def _validate_shapes(self):
        """Ensure all components have consistent dimensions"""
        assert all(v.shape[0] == self.num_samples for v in self.features), \
            "Feature dimension mismatch"
        assert self.mask.shape == (self.num_samples, self.num_views), \
            "Mask shape mismatch"
        assert self.gnd.shape == (self.num_samples,), \
            "Ground truth shape mismatch"

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tuple: Contains
                - views: List of tensors (one per view)
                - label: Ground truth label
                - idx: Sample index
                - mask: View availability mask
        """
        return (
            [view[idx] for view in self.features],  # Views
            self.gnd[idx],                          # Label
            torch.from_numpy(np.array(idx)),        # Idx
            self.indices[idx],                      # Index
            self.mask[idx]                          # Availability mask
        )

    @property
    def view_dims(self) -> list:
        """Get dimensionality of each view"""
        return [v.shape[1] for v in self.features]

def dataset_with_info(dataname: str, missing_rate: float = 0.0) -> Tuple[
    MultiViewDataset, int, int, int, List[int], np.ndarray
]:
    """
    Loads a multi-view dataset and provides comprehensive metadata
    
    Args:
        dataname: The name of Dataset file
        missing_rate: Proportion of missing views (0-1)
    
    Returns:
        Tuple containing:
        - Initialized MultiViewDataset
        - Sample count
        - Number of views
        - Cluster count
        - Feature dimensions per view
        - Ground truth labels
    
    Raises:
        ValueError: For invalid inputs or data loading failures
    """
    # Load and validate data
    try:
        features, gnd = loadData(path + dataname + '.mat')
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}") from e

    if features.size == 0:
        raise ValueError("Loaded features array is empty")
    if len(gnd) == 0:
        raise ValueError("No ground truth labels found")

    # Extract dataset characteristics
    num_views = features.shape[1]
    sample_count = features[0][0].shape[0]
    cluster_count = len(np.unique(gnd))
    feature_dims = [features[0][v].shape[1] for v in range(num_views)]

    # Initialize dataset
    dataset = MultiViewDataset(dataname, missing_rate=missing_rate)

    # Display dataset summary
    summary = (
        f"Dataset: {dataname}\n"
        f"Samples: {sample_count:,}\n"
        f"Views: {num_views}\n"
        f"Clusters: {cluster_count}\n"
        f"Feature Dimensions: {feature_dims}"
    )
    print(summary)

    return dataset, sample_count, num_views, cluster_count, feature_dims, gnd
