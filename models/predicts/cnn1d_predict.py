import torch.nn as nn
from copy import deepcopy

class CNN1D(nn.Module):
    """
    One-dimensional CNN for time series imputation.
    """
    @property
    def name(self):
        return "CNN1D"
    
    def __init__(self, input_size, hidden_channels=128, kernel_size=3):
        """
        Initialize the CNN1D model.
        
        Args:
            input_size (int): Size of the input features.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_channels * input_size, 1)
        
    def forward(self, x):
        """
        Forward pass for the CNN1D model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size].
        """
        # Reshape for 1D convolution: [batch_size, 1, input_size]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x.squeeze(-1)
    
    def get_model(self):
        """
        Get a copy of the model for training.
        
        Returns:
            CNN1D: A copy of the model.
        """
        return deepcopy(self)