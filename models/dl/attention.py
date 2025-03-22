from copy import deepcopy
import torch
import torch.nn as nn
from models.base import BaseModel

class Attention(nn.Module, BaseModel):
    """
    Attention-based neural network for time series imputation.
    """
    @property
    def name(self):
        return "Attention"
    
    def __init__(self, window_size, hidden_channels=64):
        """
        Initialize the Attention model.
        
        Args:
            window_size (int): Size of the input window.
            hidden_channels (int): Number of hidden channels.
        """
        super().__init__()
        self.window_size = window_size
        self.hidden_channels = hidden_channels
        
        # Embedding layer
        self.embedding = nn.Linear(window_size, hidden_channels)
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=1)
        self.fc_1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc_2 = nn.Linear(hidden_channels, 1)
            
    def forward(self, x):
        """
        Forward pass for the Attention model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, window_size].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size].
        """
        x = self.embedding(x)
        x, _ = self.attention(x, x, x)
        x = self.fc_1(x)
        x = nn.ReLU()(x)
        x = self.fc_2(x)
        return x.squeeze(-1)
    
    def get_model(self):
        """
        Get a copy of the model for training.
        
        Returns:
            Attention: A copy of the model.
        """
        return deepcopy(self)