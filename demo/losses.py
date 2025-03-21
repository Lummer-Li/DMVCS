import torch
import torch.nn.functional as F
from utils import get_sim

def contrastive_loss(z_i, z_j, temperature=0.5, tau=0.01):
    """Compute contrastive loss between positive pairs and negative samples"""
    batch_size = z_i.shape[0]   
    N = 2 * batch_size  # Total number of samples (original + augmented)
    mask = torch.ones((N, N))  
    mask = mask.fill_diagonal_(0)  # Mask self-comparisons
    # Remove positive pair connections
    for i in range(batch_size):
        mask[i, batch_size+i] = 0
        mask[batch_size+i, i] = 0
    mask = mask.bool()  # Convert to boolean mask

    # Concatenate original and augmented samples
    z = torch.cat([z_i, z_j], dim=0)    

    # Compute similarity matrix between all samples
    sim_matrix = torch.matmul(z, z.T) / temperature     
    # Get positive pair similarities (diagonal elements)
    sim_i_j = torch.diag(sim_matrix, batch_size)  # Original vs augmented
    sim_j_i = torch.diag(sim_matrix, -batch_size)  # Augmented vs original

    # Combine positive similarities
    pos_sim = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)    
    # Extract negative similarities using mask
    neg_sim = sim_matrix[mask].reshape(N, -1)                      

    # Cross-entropy loss calculation
    labels = torch.zeros(N).to(pos_sim.device).long()  # All positives are at index 0
    logits = torch.cat((pos_sim, neg_sim), dim=1)  # Combine positive and negatives                   
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)

    return loss / N * tau  # Normalized loss with temperature scaling

def orthogonal_loss(shared, specific, tau=0.01):
    """Penalize correlation between shared and view-specific features"""
    feature_dim = specific.shape[1]
    # Center shared features
    _shared = shared - shared.mean(dim=0)
    # Compute correlation matrix
    correlation_matrix = _shared.t().matmul(specific)
    # Measure deviation from orthogonal (diagonal should match feature dimension)
    trace_diff = torch.abs(correlation_matrix.trace() - feature_dim) * tau
    return trace_diff 

def cas_loss(cons_centroids, spec_centroids):
    """Cross-view structure alignment loss"""
    # Get similarity structures for both spaces
    cons_structure_sim = get_sim(cons_centroids)  # Common space similarity
    spec_structure_sim = get_sim(spec_centroids)  # Specific space similarity
    # Compare similarity structures using contrastive loss
    loss_csa = contrastive_loss(cons_structure_sim, spec_structure_sim)     
    return loss_csa

def mns_loss(args, cons_infos, spec_infos):
    """Multi-neighborhood similarity preservation loss"""
    # Combine all specific features from different views
    cons_mean = torch.mean(torch.stack(cons_infos), dim=0)
    specific_concat = torch.concat(spec_infos, dim=1)
    learned_feature = torch.concat([cons_mean, specific_concat], dim=1)

    cons_mean = torch.mean(torch.stack(cons_infos), dim=0)
    specific_concat = torch.concat(spec_infos, dim=1)
    learned_feature = torch.concat([cons_mean, specific_concat], dim=1)

    # Calculate multi-order neighborhood relations
    multi_order = get_multi_order(learned_feature, args.topk)
    # Apply neighborhood-aware contrastive loss
    loss_mnc = multi_samples_contrastive_loss(learned_feature, learned_feature, multi_order) 
    return loss_mnc

def multi_samples_contrastive_loss(x1, x2, mask, temperature=0.5, base_temperature=0.5):
    """Mask-guided contrastive loss with multiple positive samples"""
    device = x1.device
    # Compute similarity scores
    anchor_dot_contrast = torch.div(torch.matmul(x1, x2.T), temperature)
    # Stabilize numerical values
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  
    logits = anchor_dot_contrast - logits_max.detach() 
    
    # Apply neighborhood mask
    mask = mask.float().to(device)  
    
    # Compute softmax probabilities
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  
    
    # Calculate weighted positive probabilities
    mean_log_prob_pos = (mask * log_prob).sum(1) / torch.abs(mask).sum(1)      
    
    # Final temperature-scaled loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos  
    return loss.mean()

def compute_pure_k_hop_neighbors(knn_matrix, k_max):
    """Decompose KNN graph into pure k-hop connections"""
    pure_neighbors_list = []
    # 1-hop neighbors (direct connections)
    pure_neighbors_list.append(torch.clamp(knn_matrix, max=1))

    for k in range(2, k_max + 1):
        # Compute k-hop connectivity matrix
        current_k_hop = knn_matrix
        for _ in range(k - 1):
            current_k_hop = current_k_hop @ knn_matrix  # Matrix multiplication for hops
        current_k_hop = torch.clamp(current_k_hop, max=1)

        # Subtract lower-order connections to get pure k-hop
        lower_hops_sum = sum(pure_neighbors_list)
        pure_k_hop = current_k_hop - lower_hops_sum
        # Ensure non-negative values
        pure_k_hop = torch.clamp(pure_k_hop, min=0, max=1)
        pure_neighbors_list.append(pure_k_hop)

    return pure_neighbors_list

def get_multi_order(x, n_neighbors=10, epoch=None, k_list=None):
    """Build multi-scale neighborhood relationship matrix"""
    # Normalize input features
    x = F.normalize(x, dim=1, p=2)
    # Compute similarity matrix
    sim_matrix = torch.matmul(x, x.T)
    
    # Build KNN adjacency matrix
    if k_list is None:
        values, indices = torch.topk(sim_matrix, n_neighbors, dim=1)
        knn_matrix = torch.zeros_like(sim_matrix)
        knn_matrix.scatter_(1, indices, values)
    else:
        knn_matrix = None

    knn_matrix = knn_matrix.float()

    # Decompose into pure hop matrices (up to 3-hops)
    order_max = 3
    order_list = compute_pure_k_hop_neighbors(knn_matrix, order_max)
    
    # Apply exponentially decaying weights
    W = 1.0  # Weight for 1-hop
    multi_order = W * order_list[0]
    for i in range(2, len(order_list)):
        W /= 2.0  # Halve weight for each additional hop
        multi_order += W * order_list[i]

    return multi_order