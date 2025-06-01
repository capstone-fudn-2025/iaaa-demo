from copy import deepcopy
import torch
import torch.nn as nn
from models.base import BaseModel

from copy import deepcopy
import torch
import torch.nn as nn
from models.base import BaseModel

class MultiHeadAttention(nn.Module, BaseModel):
    """
    Multi-Head Attention-based neural network for time series imputation.
    """
    @property
    def name(self):
        return "MultiHeadAttention"
    
    def __init__(self, window_size, hidden_channels=64, num_heads=8, dropout=0.1):
        """
        Initialize the MultiHeadAttention model.
        
        Args:
            window_size (int): Size of the input window.
            hidden_channels (int): Number of hidden channels.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.window_size = window_size
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        # Ensure that hidden_channels is divisible by num_heads
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        
        # Embedding layer
        self.embedding = nn.Linear(window_size, hidden_channels)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.fc_1 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(hidden_channels, 1)
        self.relu = nn.ReLU()
            
    def forward(self, x):
        """
        Forward pass for the MultiHeadAttention model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, window_size].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size].
        """
        # Embedding: [batch_size, window_size] -> [batch_size, hidden_channels]
        x = self.embedding(x)
        
        # Reshape for attention: [batch_size, hidden_channels] -> [batch_size, 1, hidden_channels]
        x = x.unsqueeze(1)
        
        # Self-attention: [batch_size, 1, hidden_channels] -> [batch_size, 1, hidden_channels]
        x, _ = self.attention(x, x, x)
        
        # Flatten: [batch_size, 1, hidden_channels] -> [batch_size, hidden_channels]
        x = x.squeeze(1)
        
        # Feed-forward network
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        
        return x.squeeze(-1)
    
    def get_model(self):
        """
        Get a copy of the model for training.
        
        Returns:
            MultiHeadAttention: A copy of the model.
        """
        return deepcopy(self)