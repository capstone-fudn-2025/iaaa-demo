from copy import deepcopy
import torch
import torch.nn as nn
from models.base import BaseModel

from copy import deepcopy
import torch
import torch.nn as nn
from models.base import BaseModel

class GRU(nn.Module, BaseModel):
    """
    GRU-based neural network for time series imputation.
    """
    @property
    def name(self):
        return "GRU"
    
    def __init__(self, window_size, hidden_channels=64, num_layers=2, dropout=0.1, bidirectional=False):
        """
        Initialize the GRU model.
        
        Args:
            window_size (int): Size of the input window.
            hidden_channels (int): Number of hidden channels in the GRU.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout rate.
            bidirectional (bool): Whether to use bidirectional GRU.
        """
        super().__init__()
        self.window_size = window_size
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input embedding
        self.input_embedding = nn.Linear(1, hidden_channels)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension adjustment for bidirectional GRU
        gru_output_size = hidden_channels * 2 if bidirectional else hidden_channels
        
        # Output layers
        self.fc_1 = nn.Linear(gru_output_size, hidden_channels)
        self.fc_2 = nn.Linear(hidden_channels, 1)
        
        # Activation and dropout
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
            
    def forward(self, x):
        """
        Forward pass for the GRU model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, window_size].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size].
        """
        batch_size = x.size(0)
        
        # Reshape input for GRU: [batch_size, window_size] -> [batch_size, window_size, 1]
        x = x.unsqueeze(-1)
        
        # Embed each time step: [batch_size, window_size, 1] -> [batch_size, window_size, hidden_channels]
        x = self.input_embedding(x)
        
        # Pass through GRU: [batch_size, window_size, hidden_channels] -> [batch_size, window_size, gru_output_size]
        x, _ = self.gru(x)
        
        # Take the last output from sequence
        # [batch_size, window_size, gru_output_size] -> [batch_size, gru_output_size]
        x = x[:, -1, :]
        
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
            GRU: A copy of the model.
        """
        return deepcopy(self)