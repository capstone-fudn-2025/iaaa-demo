import numpy as np
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Base trainer class for time series imputation models.
    This class should be extended by specific trainers for ML or DL models.
    """
    def __init__(self, window_size=7):
        """
        Initialize the trainer with parameters.
        
        Args:
            window_size (int): Size of the sliding window for prediction models.
        """
        self.window_size = window_size
    
    def create_windows(self, data):
        """
        Transform univariate time series to multivariate data using sliding windows.
        
        Args:
            data (np.ndarray): The univariate time series data.
            
        Returns:
            tuple: X (feature windows) and y (target values)
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size])
        
        return np.array(X), np.array(y)
    
    @abstractmethod
    def train_forward(self, data, model):
        """
        Train a model in the forward direction.
        
        Args:
            data (np.ndarray): The training data.
            model: The model to train.
            
        Returns:
            model: The trained model.
        """
        pass
    
    @abstractmethod
    def train_backward(self, data, model):
        """
        Train a model in the backward direction.
        
        Args:
            data (np.ndarray): The training data.
            model: The model to train.
            
        Returns:
            model: The trained model.
        """
        pass
    
    @abstractmethod
    def predict_forward(self, data_before_gap, gap_size, model):
        """
        Predict the gap using a forward model.
        
        Args:
            data_before_gap (np.ndarray): Data before the gap.
            gap_size (int): Size of the gap to predict.
            model: The trained model.
            
        Returns:
            np.ndarray: Predicted values for the gap.
        """
        pass
    
    @abstractmethod
    def predict_backward(self, data_after_gap, gap_size, model):
        """
        Predict the gap using a backward model.
        
        Args:
            data_after_gap (np.ndarray): Data after the gap.
            gap_size (int): Size of the gap to predict.
            model: The trained model.
            
        Returns:
            np.ndarray: Predicted values for the gap (in correct order).
        """
        pass