import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Flexible multi-layer perceptron encoder with optional batch normalization
    
    Args:
        dims (list): Layer dimensions [input_dim, hidden1, ..., output_dim]
        bn (bool): Add batch norm after hidden layers
    """
    def __init__(self, dims, bn = False):
        super(Encoder, self).__init__()
        assert len(dims) >= 2
        models = []

        # Construct Hidden Layer
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))      # Add Linear Layer
            if i != len(dims) - 2:
                models.append(nn.ReLU(inplace=True))            # Add activate function

        self.models = nn.Sequential(*models)

    def forward(self, X):
        """Input shape: (batch_size, input_dim)"""
        return self.models(X)
    
class Decoder(nn.Module):
    """
    Multi-layer perceptron decoder for feature reconstruction
    
    Architecture: Series of linear layers with ReLU activation on final layer
    Typical use: Expanding latent representations to high-dimensional outputs
    """
    def __init__(self, dims):
        """
        Initialize decoder layers
        
        Args:
            dims (list): Layer dimension sequence. Example:
                [256, 512, 784] creates:
                - Input layer: 256 units
                - Hidden layer: 512 units 
                - Output layer: 784 units
        """
        super(Decoder, self).__init__()
        
        # Layer container initialization
        models = []
        
        # Layer construction loop (iterates for N-1 layers where N = len(dims))
        for i in range(len(dims) - 1):
            # Linear transformation: dims[i] → dims[i+1]
            models.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add ReLU only after final linear layer
            # Condition: i == len(dims)-2 indicates last layer index
            if i == len(dims) - 2:
                models.append(nn.ReLU())
        
        # Sequential container for layer execution
        self.models = nn.Sequential(*models)
    
    def forward(self, X):
        """
        Forward pass processing
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor: Reconstructed output of shape (batch_size, output_dim)
        """
        # Sequential processing through defined layers
        return self.models(X)
    
class Expert(nn.Module):
    """A neural network expert module with one hidden layer"""
    
    def __init__(self, input_size, output_size):
        """Initialize the expert network
        Args:
            input_size: Dimension of input features
            output_size: Dimension of output features
        """
        super(Expert, self).__init__()
        
        self.fc = nn.Sequential(
            # First fully connected layer
            nn.Linear(input_size, 1024),
            # Batch normalization for stability and faster training
            nn.BatchNorm1d(1024),
            # Non-linear activation function
            nn.ReLU(),
            # Second fully connected layer (output layer)
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        """Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.fc(x)

class MyNet(nn.Module):
    """
    Multi-view autoencoder network with feature decomposition
    Key Components:
        - View-specific encoders/decoders
        - Latent space projection
    """
    def __init__(self, args, input_dims, view_num, class_num):
        """
        Args:
            args: Configuration parameters
            input_dims (list): Input dimensions for each view
            view_num: Number of data views/modalities
            class_num: Number of target classes
        """
        super().__init__()
        # Initialize architecture parameters
        self.input_dims = input_dims  # List of input dimensions per view
        self.view = view_num          # Number of data views/modalities
        self.class_num = class_num    # Number of target classes
        self.embedding_dim = args.embedding_dim  # Latent space dimension
        self.h_dims = args.hidden_dims  # Encoder hidden layer dimensions
        self.device = args.device     # Computation device

        # Reverse hidden dims for decoder construction
        h_dims_reverse = list(reversed(args.hidden_dims))

        # View-specific components
        self.encoders = []  # Encoder networks for each view
        self.decoders = []  # Decoder networks for each view
        self.gating_networks = []
        self.consists = []
        self.specifics = []
        for v in range(self.view):
            # Encoder architecture: Input -> Hidden Layers -> Embedding
            self.encoders.append(Encoder([input_dims[v]] + self.h_dims + [self.embedding_dim], bn=True).to(self.device))
            # Decoder architecture: Embedding -> Reversed Hidden -> Input
            self.decoders.append(Decoder([self.embedding_dim * 2] + h_dims_reverse + [input_dims[v]]).to(args.device))
            self.consists.append(Expert(self.embedding_dim, self.embedding_dim).to(self.device))
            self.specifics.append(Expert(self.embedding_dim, self.embedding_dim).to(self.device))
            self.gating_networks.append(nn.Linear(input_dims[v], 2).to(self.device))

        # Register components as proper module lists
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.consists = nn.ModuleList(self.consists)
        self.specifics = nn.ModuleList(self.specifics)
        self.gating_networks = nn.ModuleList(self.gating_networks)

        # Shared projection network
        self.cluster_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, xs, clustering=False, target=None):
        """
        Multi-view feature decomposition and reconstruction pipeline
        
        Args:
            xs: List of input tensors (one per view) [batch_size, input_dim]
            clustering: Flag to enable cluster optimization (unused here)
            target: Optional supervision labels (unused here)
        
        Returns:
            xrs: Reconstructed views [view_num x (batch_size, input_dim)]
            cons_infos: Common features across views [view_num x (batch_size, latent_dim)]
            spec_infos: View-specific features [view_num x (batch_size, latent_dim)]
            cons_clusters: Common feature cluster assignments [view_num x (batch_size, clusters)]
            spec_clusters: Specific feature cluster assignments [view_num x (batch_size, clusters)]
        """
        
        # Initialize storage containers
        xrs = []          # Reconstructed input views
        cons_infos = []   # Cross-view common features
        spec_infos = []   # View-specific features
        cons_clusters = []  # Cluster assignments for common features
        spec_clusters = []  # Cluster assignments for specific features

        # Process each view independently
        for v in range(self.view):
            # 1. Input Processing
            x = xs[v]  # Current view's input [batch_size, input_dim]
            
            # 2. Gating Mechanism
            gating_score = F.softmax(self.gating_networks[v](x), dim=1)  # [batch_size, 2]
            
            # 3. Feature Encoding
            z = self.encoders[v](x)  # Intermediate features [batch_size, hidden_dim]
            
            # 4. Feature Decomposition
            cons_info = self.consists[v](z)  # Common features [batch_size, latent_dim]
            spec_info = self.specifics[v](z)  # Specific features [batch_size, latent_dim]
            
            # Store decomposed features
            cons_infos.append(cons_info)
            spec_infos.append(spec_info)
            
            # 5. Gated Feature Fusion
            recon_info = torch.cat([
                cons_info * gating_score[:, 0].unsqueeze(1),  # Weighted common component
                spec_info * gating_score[:, 1].unsqueeze(1)  # Weighted specific component
            ], dim=1)  # [batch_size, 2*latent_dim]
            
            # 6. Feature Reconstruction
            xr = self.decoders[v](recon_info)  # Reconstructed view [batch_size, input_dim]
            xrs.append(xr)
            
            # 7. Cluster Assignment
            cons_clusters.append(self.cluster_head(cons_info))  # [batch_size, clusters]
            spec_clusters.append(self.cluster_head(spec_info))  # [batch_size, clusters]

        return xrs, cons_infos, spec_infos, cons_clusters, spec_clusters