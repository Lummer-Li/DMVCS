import numpy as np
from typing import Tuple
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def bestMap(y_pred, y_true):
    """
    Find optimal label mapping between predicted and true labels using Hungarian algorithm
    
    Args:
        y_pred: 1D array of predicted labels
        y_true: 1D array of ground truth labels
    
    Returns:
        Adjusted predicted labels optimally mapped to true labels
    """
    # Validate inputs
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    assert y_pred.ndim == y_true.ndim == 1, "Inputs must be 1D arrays"
    assert len(y_pred) == len(y_true), "Inputs must have same length"
    
    # Handle empty case
    if len(y_pred) == 0:
        return np.array([], dtype=np.int64)
    
    # Calculate label range
    D = max(y_pred.max(), y_true.max()) + 1
    
    # Create co-occurrence matrix using vectorized operations
    w = np.zeros((D, D), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    
    # Find optimal mapping using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # Create label mapping array
    label_map = np.zeros(D, dtype=np.int64)
    label_map[row_ind] = col_ind
    
    # Apply optimal mapping to predictions
    return label_map[y_pred]

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm for optimal label alignment
    
    Args:
        y_true (array-like): True labels array of shape (n_samples,)
        y_pred (array-like): Predicted labels array of shape (n_samples,)
        
    Returns:
        float: Clustering accuracy between 0.0 and 1.0
        
    Raises:
        AssertionError: If input constraints are violated
    """
    # Convert inputs to flattened integer arrays
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    
    # Validate input dimensions
    assert y_pred.size == y_true.size, "Input arrays must have the same length"
    assert y_pred.size > 0, "Input arrays cannot be empty"
    
    # Determine label space size
    max_label = max(y_pred.max(), y_true.max())
    n_classes = max_label + 1
    
    # Build co-occurrence matrix using vectorized operations
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion_matrix, (y_pred, y_true), 1)
    
    # Find optimal label mapping using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    
    # Calculate total correctly matched samples
    correct_matches = confusion_matrix[row_ind, col_ind].sum()
    
    # Compute clustering accuracy
    accuracy = correct_matches / y_pred.size
    
    return accuracy

def purity(y_true, y_pred):
    """
    Calculate clustering purity score between true labels and predicted clusters
    
    Args:
        y_true (array-like): True class labels, shape (n_samples,)
        y_pred (array-like): Cluster assignments, shape (n_samples,)
    
    Returns:
        float: Purity score between 0.0 (worst) and 1.0 (perfect)
    """
    # Convert inputs to numpy arrays and flatten
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Validate input dimensions
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")
    if y_true.size == 0:
        return 0.0  # Handle empty input case
    
    # Get unique cluster IDs and remap to contiguous indices
    clusters, cluster_ids = np.unique(y_pred, return_inverse=True)
    classes, class_ids = np.unique(y_true, return_inverse=True)
    
    # Create contingency matrix using vectorized operations
    contingency = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    np.add.at(contingency, (cluster_ids, class_ids), 1)
    
    # Calculate purity score (sum of max matches / total samples)
    return contingency.max(axis=1).sum() / y_true.size

def clusteringMetrics(trueLabel: np.ndarray, predictiveLabel: np.ndarray) -> Tuple[float, ...]:
    """
    Compute comprehensive clustering evaluation metrics
    
    Args:
        trueLabel: Ground truth class labels (1D array)
        predictiveLabel: Cluster assignments (1D array)
        
    Returns:
        Tuple containing:
        - ACC: Clustering Accuracy
        - NMI: Normalized Mutual Information
        - Purity: Cluster Purity
        - ARI: Adjusted Rand Index
        - Fscore: Fowlkes-Mallows Score
        - Precision: Macro-averaged Precision (requires label alignment)
        - Recall: Macro-averaged Recall (requires label alignment)
        
    Note: For meaningful Precision/Recall, ensure labels are aligned using bestMap
    """
    # Convert inputs to 1D numpy arrays
    trueLabel = np.asarray(trueLabel).ravel()
    predictiveLabel = np.asarray(predictiveLabel).ravel()
    
    # Validate input shapes
    if trueLabel.shape != predictiveLabel.shape:
        raise ValueError(f"Shape mismatch: trueLabel {trueLabel.shape}, predictiveLabel {predictiveLabel.shape}")
    if trueLabel.size == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Calculate metrics in optimal order (complexity-based)
    ACC = cluster_acc(trueLabel, predictiveLabel)
    NMI = metrics.normalized_mutual_info_score(trueLabel, predictiveLabel)
    ARI = metrics.adjusted_rand_score(trueLabel, predictiveLabel)
    Fscore = metrics.fowlkes_mallows_score(trueLabel, predictiveLabel)
    
    # Alignment-sensitive metrics (optional: add bestMap here if needed)
    aligned_labels = predictiveLabel  # Replace with bestMap(predictiveLabel, trueLabel) for alignment
    Purity = purity(trueLabel, aligned_labels)
    Precision = metrics.precision_score(trueLabel, aligned_labels, average='macro', zero_division=0)
    Recall = metrics.recall_score(trueLabel, aligned_labels, average='macro', zero_division=0)
    
    return ACC, NMI, Purity, ARI, Fscore, Precision, Recall