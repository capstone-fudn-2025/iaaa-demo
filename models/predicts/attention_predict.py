import torch.nn as nn
from copy import deepcopy

class Attention(nn.Module):
    """
    Attention-based neural network for time series imputation.
    """
    @property
    def name(self):
        return "Attention"
    
    def __init__(self, input_size, hidden_channels=64):
        """
        Initialize the Attention model.
        
        Args:
            input_size (int): Size of the input features.
            hidden_channels (int): Number of hidden channels.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        
        # Embedding layer
        self.embedding = nn.Linear(input_size, hidden_channels)
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=1)
        self.fc_1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc_2 = nn.Linear(hidden_channels, 1)
            
    def forward(self, x):
        """
        Forward pass for the Attention model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size].
        """
        # [batch_size, input_size] -> [batch_size, hidden_channels]
        x = self.embedding(x)
        # Reshape for attention: [batch_size, 1, hidden_channels]
        x = x.unsqueeze(1)
        # Apply self-attention (requires shape [seq_len, batch_size, hidden_channels])
        x = x.transpose(0, 1)  # [1, batch_size, hidden_channels]
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)  # [batch_size, 1, hidden_channels]
        x = x.squeeze(1)  # [batch_size, hidden_channels]
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