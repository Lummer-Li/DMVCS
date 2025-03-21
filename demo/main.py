import torch
import configs
import argparse
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from models import MyNet
from dataloader import dataset_with_info
from utils import set_seed, get_logger, get_centroids
from metrics import clusteringMetrics
from losses import  orthogonal_loss, cas_loss, mns_loss

def main(args, logger, datasetforuse, data_size, view_num, nc, input_dims, gnd):
    """Main training pipeline for multi-view clustering model
    
    Args:
        args: Configuration parameters
        logger: Configured logging handler
        datasetforuse: Dataset object containing multi-view data
        data_size: Number of samples
        view_num: Number of data views/modalities
        nc: Number of clusters
        input_dims: List of input dimensions per view
        gnd: Ground truth labels
    
    Returns:
        Tuple of evaluation metrics (ACC, NMI, Purity, ARI, Fscore, Precision, Recall)
    """
    
    # Initialize data loaders with full dataset utilization
    train_loader = DataLoader(
        datasetforuse, 
        batch_size=args.batch_size, 
        shuffle=True,      # Enable random sampling
        drop_last=False    # Use all samples including partial batches
    ) 
    test_loader = DataLoader(
        datasetforuse, 
        batch_size=args.batch_size, 
        shuffle=False,     # Maintain original order for evaluation
        drop_last=False
    )

    # Display configuration header
    print("="*120)
    logger.info(str(args))  # Log all hyperparameters
    print("="*120)

    # Model initialization with multi-view architecture
    model = MyNet(
        args=args,          # Configuration object
        input_dims=input_dims,  # Feature dimensions per view
        view_num=view_num,  # Number of data modalities
        class_num=nc        # Target cluster count
    ).to(args.device)       # Device placement (GPU/CPU)

    # Adam optimizer with L2 regularization
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr,              # Learning rate
        weight_decay=args.weight_decay  # Weight decay coefficient
    )

    # Mean Squared Error loss for reconstruction
    mse_loss_fn = nn.MSELoss()

    # Training metrics storage
    losses = []

    # Main training loop
    for epoch in range(args.train_epochs):
        total_loss = 0.  # Epoch loss accumulator
        
        # Process each training batch
        for x, y, idx, inpu, mask in train_loader:
            # Initialize loss components
            loss_dis = 0.    # Multi-view decomposition loss (reconstruction + orthogonality)
            loss_csa = 0.    # Cross-view structure alignment loss
            loss_mnc = 0.    # Modality-neutral consistency loss
                            
            # Set model to training mode
            model.train()
            
            # Process each view's data
            for v in range(view_num):
                # Apply data masking for missing view handling
                # Transfer data to GPU/CPU device
                x[v] = x[v].to(args.device)
                y = y.to(args.device)

            # Forward pass through network
            xrs, cons_infos, spec_infos, cons_clusters, spec_clusters = model(x, clustering=False)
            
            # Aggregate common information across views
            cons_info = torch.mean(torch.stack(cons_infos), dim=0)  # Average common features
            cons_cluster = torch.mean(torch.stack(cons_clusters), dim=0)  # Average cluster assignments
            cons_centroids = get_centroids(args, nc, cons_info, cons_cluster)  # Common centroids

            # Reset gradients before backward pass
            optimizer.zero_grad()
            
            # Calculate view-specific losses
            for v in range(view_num):
                # 1. Reconstruction loss (mask-aware data reconstruction)
                loss_dis += mse_loss_fn(xrs[v], x[v])
                
                # 2. Cross-view structure alignment
                spec_centroids = get_centroids(args, nc, spec_infos[v], spec_clusters[v])
                loss_csa += cas_loss(cons_centroids, spec_centroids)  # Align common-specific structures
                
                # 3. Orthogonal constraint between view-specific components
                for w in range(v+1, view_num):
                    loss_dis += orthogonal_loss(spec_infos[v], spec_infos[w])  # Prevent redundancy
                    
            # 4. Multi-hop Neighbor Contrastive Learning
            loss_mnc = mns_loss(args, cons_infos, spec_infos)

            # 5. Cluster consistency across views
            for v in range(view_num):
                for w in range(v+1, view_num):
                    loss_dis += mse_loss_fn(cons_clusters[v], cons_clusters[w])  # Enforce cluster agreement

            # Combine all loss components
            loss = loss_dis + loss_csa + loss_mnc
            
            # Backpropagation and optimization
            total_loss += loss.item() 
            loss.backward(retain_graph=True)  # Maintain computation graph for potential multi-task learning
            optimizer.step()  # Update model parameters
            
        # Epoch logging    
        losses.append(total_loss)
        # print(f'epoch: {epoch+1}, total loss: {loss.item():.4f}, dis loss: {loss_dis:.4f}, csa loss: {loss_csa:.4f}, mnc loss: {loss_mnc:.4f}')
        
        # Validation phase
        if (epoch + 1) % args.valid_epochs == 0:
            # Switch to evaluation mode
            model.eval()  # Disables dropout/batchnorm
            with torch.no_grad():  # No gradient tracking
                # Initialize feature storage
                learned_features = []  # Combined representations
                consist_features = []  # Shared components
                specific_features = []  # View-specific components
                
                # Process evaluation batches
                for x, y, idx, inpu, mask in test_loader:
                    # Apply view-specific masking
                    for v in range(view_num):
                        # Zero out missing views using mask
                        # mask[:, v].unsqueeze(1) adds feature dimension for broadcasting
                        x[v] = x[v] * mask[:, v].unsqueeze(1)  
                        x[v] = x[v].to(args.device)  # Device transfer
                    
                    # Decompose inputs into components
                    xrs, cons_infos, spec_infos, cons_clusters, spec_clusters = model(x)
                    
                    # Aggregate common features across views
                    consist = torch.mean(torch.stack(cons_infos), dim=0)  # Average shared features
                    # Concatenate view-specific features column-wise
                    specific = torch.concat(spec_infos, dim=1)  
                    # Combine shared + specific features
                    learned_feature = torch.concat([consist, specific], dim=1)  

                    # Store features for evaluation
                    learned_features.extend(learned_feature.detach().cpu().numpy())
                    consist_features.extend(consist.detach().cpu().numpy())
                    specific_features.extend(specific.detach().cpu().numpy())

                # Clustering evaluation
                learned_features = np.array(learned_features)
                kmeans = KMeans(n_clusters=nc, n_init=50)
                y_pred = kmeans.fit_predict(learned_features)      
                
                # Metric computation
                ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(gnd, y_pred) 
                
                # Result logging
                info = {
                    "epoch": epoch + 1, 
                    "acc": '%.4f'%ACC, 
                    "nmi": '%.4f'%NMI,
                    "ari": '%.4f'%ARI,
                    "Purity": '%.4f'%Purity,
                    "fscore": '%.4f'%Fscore,
                    "percision": '%.4f'%Precision,
                    "recall": '%.4f'%Recall
                }
                logger.info(str(info))

            # Final epoch return
            if (epoch + 1) == args.train_epochs:
                return ACC, NMI, Purity, ARI, Fscore, Precision, Recall

if __name__ == '__main__':
    """
    Deep Multi-view Clustering with Intra-view Similarity and Cross-view Correlation Learning (MISCC)
    Main execution pipeline for multi-view clustering training
    
    Steps:
    1. Parameter configuration
    2. Dataset preparation
    3. Model initialization
    4. Training model and optimize its parameters
    5. Final evaluation
    """
    # Argument parsing for training configuration
    parser = argparse.ArgumentParser()
    # Experiment setup
    parser.add_argument('--seed', type=int, default=10,
                      help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='ALOI',
                      help='Dataset name from available options')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='Number of samples per training batch')
    parser.add_argument('--shffule', type=bool, default=True,
                      help='Shffuling the choosed datasets')
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                      help='Momentum factor (not used in Adam)')
    parser.add_argument('--weight_decay', type=float, default=0,
                      help='Weight decay (L2 penalty)')
    
    # Model architecture
    parser.add_argument('--embedding_dim', type=int, default=64,
                      help='Dimension of encoder hidden layer')
    parser.add_argument('--hidden_dims', type=list, default=[1024, 1024],
                      help='Dimension of each hidden layer')
    parser.add_argument('--topk', type=int, default=10,
                      help='Number of neighbors in one-order graph')
    parser.add_argument('--ratios', type=list, default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                      help='Rate of Prototypes')
    
    # Training schedule
    parser.add_argument('--train_epochs', type=int, default=500,
                      help='Training epochs for reconstruction')
    parser.add_argument('--valid_epochs', type=int, default=100,
                      help='Cluster validation interval during training')
    
    # Advanced configurations
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Computation device: cuda:id or cpu')
    parser.add_argument('--save_flag', type=bool, default=False,
                      help='Flag for model checkpoint saving')
    
    args = parser.parse_args()
    args = configs.get_config(args)

    # Environment setup
    logger = get_logger(__file__, args.dataset)
    datasetforuse, data_size, view_num, nc, input_dims, gnd = dataset_with_info(args.dataset)
    
    acc_list, nmi_list, ari_list, pur_list, fscore_list = [], [], [], [], []
    for i in range(5):                  
        print('======================== current seed %d ======================='%(i+1))
        set_seed(i+1)
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = main(args, logger, datasetforuse, data_size, view_num, nc, input_dims, gnd)
        acc_list.append(ACC)
        nmi_list.append(NMI)
        ari_list.append(ARI)
        pur_list.append(Purity)
        fscore_list.append(Fscore)

    print('Final results (Standard and Variance): ')
    print('ACC: ave|{:04f} std|{:04f}'.format(np.mean(acc_list), np.std(acc_list, ddof=1)))
    print('NMI: ave|{:04f} std|{:04f}'.format(np.mean(nmi_list), np.std(nmi_list, ddof=1)))
    print('PUR: ave|{:04f} std|{:04f}'.format(np.mean(pur_list), np.std(pur_list, ddof=1)))
    print('ARI: ave|{:04f} std|{:04f}'.format(np.mean(ari_list), np.std(ari_list, ddof=1)))
    print('Fscore: ave|{:04f} std|{:04f}'.format(np.mean(fscore_list), np.std(fscore_list, ddof=1)))