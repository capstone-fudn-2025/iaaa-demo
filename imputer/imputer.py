import numpy as np
import pandas as pd
from copy import deepcopy

class Imputer:
    def __init__(self, window_size=7, min_gap_size=5, combination_method='wdbi'):
        """
        Initialize the imputer with parameters.
        
        Args:
            window_size (int): Size of the sliding window for prediction models.
            min_gap_size (int): Minimum gap size to use ML/DL models instead of linear interpolation.
            combination_method (str): Method to combine forward and backward predictions ('wdbi' or 'mean').
        """
        self.window_size = window_size
        self.min_gap_size = min_gap_size
        self.combination_method = combination_method
    
    def find_nan_gap(self, data):
        """
        Find nan gap from data and return the start and end index of the gap.
        
        Args:
            data (np.ndarray or pd.Series): The data to be processed.
        
        Returns:
            list: The list of (start, end) index pairs of the gaps.
        """
        # Convert data to numpy array if it's a pandas Series
        if isinstance(data, pd.Series):
            data = data.values
        
        # Convert the data to a boolean mask where True indicates NaN
        is_nan = np.isnan(data)
        
        # Find where the NaN status changes (from non-NaN to NaN, or NaN to non-NaN)
        change_points = np.diff(is_nan.astype(int))
        
        # Get indices where changes occur
        start_indices = np.nonzero(change_points == 1)[0] + 1  # NaN starts
        end_indices = np.nonzero(change_points == -1)[0] + 1   # NaN ends
        
        # Handle edge cases: if data starts with NaN
        if is_nan[0]:
            start_indices = np.insert(start_indices, 0, 0)
        
        # Handle edge cases: if data ends with NaN
        if is_nan[-1]:
            end_indices = np.append(end_indices, len(data))
        
        # Combine start and end indices into pairs
        gaps = list(zip(start_indices, end_indices))
        
        return gaps
    
    def combine_predictions_wdbi(self, forward_pred, backward_pred, data_before_gap, gap_size, total_length):
        """
        Combine forward and backward predictions using the WDBI (Weighted Distance-Based Imputation) method.
        
        Args:
            forward_pred (np.ndarray): Predictions from forward model.
            backward_pred (np.ndarray): Predictions from backward model.
            data_before_gap (np.ndarray): Data before the gap.
            gap_size (int): Size of the gap.
            total_length (int): Total length of the time series.
            
        Returns:
            np.ndarray: Combined predictions.
        """
        # Calculate weights based on relative position of missing data
        weight_forward = (len(data_before_gap) + gap_size) / total_length
        weight_backward = 1 - weight_forward
        
        print(f"Weight forward: {weight_forward:.4f}, Weight backward: {weight_backward:.4f}")
        
        # Combine predictions using weights
        combined_pred = weight_forward * forward_pred + weight_backward * backward_pred
        
        return combined_pred
    
    def impute(self, data, forward_model, backward_model, trainer):
        """
        Impute missing gaps in time series data using ML/DL models.
        
        Args:
            data (np.ndarray or pd.Series): The time series data with gaps.
            forward_model: The model for forward prediction.
            backward_model: The model for backward prediction.
            trainer: Trainer object to train models.
            
        Returns:
            np.ndarray: Data with imputed values.
        """
        # Make a copy to avoid modifying the original data
        imputed_data = deepcopy(data)
        
        # Convert data to numpy array if it's a pandas Series
        if isinstance(imputed_data, pd.Series):
            imputed_data = imputed_data.values
        
        # Find gaps of NaN values
        gaps = self.find_nan_gap(imputed_data)
        
        # Time index
        time_index = np.arange(len(imputed_data))
        
        # First pass: fill all gaps with interpolation
        non_nan_indices = np.nonzero(~np.isnan(imputed_data))[0]
        if len(non_nan_indices) > 0:
            # Use linear interpolation for all gaps initially
            x_obs = time_index[non_nan_indices]
            y_obs = imputed_data[non_nan_indices]
            
            # Interpolate all missing values
            imputed_data = np.interp(time_index, x_obs, y_obs)
        
        # Second pass: for larger gaps, use ML/DL models
        for start, end in gaps:
            gap_size = end - start
            
            # Skip small gaps (already handled by interpolation)
            if gap_size < self.min_gap_size:
                continue
            
            # Get data before and after gap
            data_before_gap = imputed_data[:start]
            data_after_gap = imputed_data[end:]
            
            # Check if enough data to train forward models
            has_enough_forward_data = len(data_before_gap) >= self.window_size + 1
            
            # Check if enough data to train backward models
            has_enough_backward_data = len(data_after_gap) >= self.window_size + 1
            
            # Skip if not enough data in either direction
            if not has_enough_forward_data and not has_enough_backward_data:
                continue
            
            # Initialize predictions as None
            forward_model_pred = None
            backward_model_pred = None
            
            # Train and predict with forward model if enough data
            if has_enough_forward_data:
                print(f"Training forward model for gap of size {gap_size}")
                # Train forward model
                forward_model_trained = trainer.train_forward(data_before_gap, forward_model)
                
                # Predict using forward model
                forward_model_pred = trainer.predict_forward(data_before_gap, gap_size, forward_model_trained)
            
            # Train and predict with backward model if enough data
            if has_enough_backward_data:
                print(f"Training backward model for gap of size {gap_size}")
                # Train backward model
                backward_model_trained = trainer.train_backward(data_after_gap, backward_model)
                
                # Predict using backward model
                backward_model_pred = trainer.predict_backward(data_after_gap, gap_size, backward_model_trained)
            
            # Combine predictions based on available models
            if has_enough_forward_data and has_enough_backward_data:
                # Combine forward and backward predictions
                if self.combination_method == 'wdbi':
                    # Use WDBI method
                    final_pred = self.combine_predictions_wdbi(
                        forward_model_pred, 
                        backward_model_pred, 
                        data_before_gap, 
                        gap_size, 
                        len(imputed_data)
                    )
                else:
                    # Use simple mean
                    final_pred = (forward_model_pred + backward_model_pred) / 2
            elif has_enough_forward_data:
                # Only use forward predictions
                final_pred = forward_model_pred
            else:
                # Only use backward predictions
                final_pred = backward_model_pred
            
            # Fill the gap with predictions
            imputed_data[start:end] = final_pred
        
        return imputed_data