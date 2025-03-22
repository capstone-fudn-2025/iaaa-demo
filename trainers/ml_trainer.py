import numpy as np
from copy import deepcopy
from trainers.base_trainer import BaseTrainer

class MLTrainer(BaseTrainer):
    """
    Trainer for machine learning models (scikit-learn compatible).
    """
    def train_forward(self, data, model):
        """
        Train a model in the forward direction.
        
        Args:
            data (np.ndarray): The training data.
            model: The scikit-learn compatible model to train.
            
        Returns:
            model: The trained model.
        """
        # Create windowed data
        X, y = self.create_windows(data)
        
        # Create a deep copy of the model to avoid modifying the original
        model_copy = deepcopy(model)
        
        # Fit the model
        model_copy.fit(X, y)
        
        return model_copy
    
    def train_backward(self, data, model):
        """
        Train a model in the backward direction.
        
        Args:
            data (np.ndarray): The training data.
            model: The scikit-learn compatible model to train.
            
        Returns:
            model: The trained model.
        """
        # Reverse the data for backward training
        reversed_data = data[::-1].copy()
        
        # Create windowed data
        X, y = self.create_windows(reversed_data)
        
        # Create a deep copy of the model to avoid modifying the original
        model_copy = deepcopy(model)
        
        # Fit the model
        model_copy.fit(X, y)
        
        return model_copy
    
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
        predictions = []
        current_window = data_before_gap[-self.window_size:]
        
        for _ in range(gap_size):
            # Reshape for prediction
            X_pred = current_window.reshape(1, -1)
            
            # Make prediction
            pred = model.predict(X_pred)[0]
            predictions.append(pred)
            
            # Update window for next prediction
            current_window = np.append(current_window[1:], pred)
        
        return np.array(predictions)
    
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
        # Reverse the data for backward prediction
        reversed_data = data_after_gap[::-1].copy()
        
        predictions = []
        current_window = reversed_data[:self.window_size]
        
        for _ in range(gap_size):
            # Reshape for prediction
            X_pred = current_window.reshape(1, -1)
            
            # Make prediction
            pred = model.predict(X_pred)[0]
            predictions.append(pred)
            
            # Update window for next prediction
            current_window = np.append(current_window[1:], pred)
        
        # Reverse the predictions to get the correct order
        return np.array(predictions[::-1])