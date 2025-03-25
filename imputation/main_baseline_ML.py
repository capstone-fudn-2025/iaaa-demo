import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

from utils.draw_fig import draw_fig
from utils.preprocessing import Preprocessing

def find_nan_gap(data):
    """To find nan gap from data and return the start and end index of the gap.
    Args:
        data (np.ndarray): The data to be processed.
    
    Returns:
        list: The list (start, end) index of the gap.
    """
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

def create_windows(data, window_size):
    """Transform univariate time series to multivariate data using sliding windows.
    
    Args:
        data (np.ndarray): The univariate time series data.
        window_size (int): Size of the sliding window.
        
    Returns:
        tuple: X (feature windows) and y (target values)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    return np.array(X), np.array(y)

def train_forward_model(data, window_size, model):
    """Train a model in the forward direction.
    
    Args:
        data (np.ndarray): The training data.
        window_size (int): Size of the sliding window.
        model: The machine learning model to train.
        
    Returns:
        model: The trained model.
    """
    # Create windowed data
    X, y = create_windows(data, window_size)
    
    # Fit the model
    model.fit(X, y)
    
    return model

def train_backward_model(data, window_size, model):
    """Train a model in the backward direction.
    
    Args:
        data (np.ndarray): The training data (should already be reversed).
        window_size (int): Size of the sliding window.
        model: The machine learning model to train.
        
    Returns:
        model: The trained model.
    """
    # Create windowed data (data should already be reversed)
    X, y = create_windows(data, window_size)
    
    # Fit the model
    model.fit(X, y)
    
    return model

def predict_gap_forward(data_before_gap, gap_size, window_size, model):
    """Predict the gap using a forward model.
    
    Args:
        data_before_gap (np.ndarray): Data before the gap.
        gap_size (int): Size of the gap to predict.
        window_size (int): Size of the sliding window.
        model: The trained model.
        
    Returns:
        np.ndarray: Predicted values for the gap.
    """
    predictions = []
    current_window = data_before_gap[-window_size:]
    
    for _ in range(gap_size):
        # Reshape for prediction
        X_pred = current_window.reshape(1, -1)
        
        # Make prediction
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        
        # Update window for next prediction
        current_window = np.append(current_window[1:], pred)
    
    return np.array(predictions)

def predict_gap_backward(data_after_gap, gap_size, window_size, model):
    """Predict the gap using a backward model.
    
    Args:
        data_after_gap (np.ndarray): Data after the gap.
        gap_size (int): Size of the gap to predict.
        window_size (int): Size of the sliding window.
        model: The trained model.
        
    Returns:
        np.ndarray: Predicted values for the gap (in correct order).
    """
    # The model was trained on reversed data, so we need to reverse data_after_gap
    reversed_data = data_after_gap[::-1]
    
    predictions = []
    current_window = reversed_data[:window_size]  # Take the first window from reversed data
    
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

def combine_predictions_wdbi(forward_pred, backward_pred, data_before_gap, gap_size, total_length):
    """Combine forward and backward predictions using the WDBI (Weighted Distance-Based Imputation) method.
    
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

def imputed_missing_gaps(data, forward_model=None, backward_model=None, window_size=7, min_gap_size=5, combination_method='wdbi'):
    # Make a copy to avoid modifying the original data
    imputed_data = deepcopy(data)
    
    # Find gaps of NaN values
    gaps = find_nan_gap(imputed_data)
    
    # Time index
    time_index = np.arange(len(data))
    
    # First pass: fill all gaps with interpolation
    non_nan_indices = np.where(~np.isnan(imputed_data))[0]
    if len(non_nan_indices) > 0:
        # Use linear interpolation for all gaps initially
        x_obs = time_index[non_nan_indices]
        y_obs = imputed_data[non_nan_indices]
        
        # Interpolate all missing values
        imputed_data = np.interp(time_index, x_obs, y_obs)
    
    # Second pass: for larger gaps, use ML models
    for start, end in gaps:
        gap_size = end - start
        
        # Skip small gaps (already handled by interpolation)
        if gap_size < min_gap_size:
            continue
        
        # Get data before and after gap
        data_before_gap = imputed_data[:start]
        data_after_gap = imputed_data[end:]
        
        # Check if enough data to train forward models
        has_enough_forward_data = len(data_before_gap) >= window_size + 1
        
        # Check if enough data to train backward models
        has_enough_backward_data = len(data_after_gap) >= window_size + 1
        
        # Skip if not enough data in either direction
        if not has_enough_forward_data and not has_enough_backward_data:
            continue
        
        # Initialize predictions as None
        forward_model_pred = None
        backward_model_pred = None
        
        # Train and predict with forward models if enough data
        if has_enough_forward_data:
            # Train forward models
            forward_model_trained = train_forward_model(data_before_gap, window_size, deepcopy(forward_model))
            
            # Predict using forward models
            forward_model_pred = predict_gap_forward(data_before_gap, gap_size, window_size, forward_model_trained)
        
        # Train and predict with backward models if enough data
        if has_enough_backward_data:
            # Train backward models (reverse the data first)
            reversed_data_after_gap = data_after_gap[::-1]
            backward_model_trained = train_backward_model(reversed_data_after_gap, window_size, deepcopy(backward_model))
            
            # Predict using backward models
            backward_model_pred = predict_gap_backward(data_after_gap, gap_size, window_size, backward_model_trained)
            # Reverse the predictions to get the correct order
            backward_model_pred = backward_model_pred[::-1]

        # Combine predictions based on available models
        if has_enough_forward_data and has_enough_backward_data:
            # Combine forward and backward predictions
            if combination_method == 'wdbi':
                # Use WDBI method
                final_pred = combine_predictions_wdbi(
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

if __name__ == "__main__":
    # constants
    COMBINATION_METHOD = 'wdbi'
    WINDOW_SIZE = 7
    MIN_GAP_SIZE = 5

    # Load data
    dataframe = pd.read_csv('data/CAF003.csv')
    univariate_raw_data = dataframe[['VW_30cm']]
    
    # Preprocessing data
    preprocessing = Preprocessing()
    preprocessed_data = preprocessing.flow(univariate_raw_data)
    univariate_raw_data = preprocessed_data['VW_30cm'].to_numpy()
    data = deepcopy(univariate_raw_data)

    # Create ML models
    forward_adaboost = AdaBoostRegressor(random_state=42)
    backward_adaboost = AdaBoostRegressor(random_state=42)
    
    imputed_data = imputed_missing_gaps(
        data, 
        forward_model=forward_adaboost, 
        backward_model=backward_adaboost,
        combination_method=COMBINATION_METHOD,
        window_size=WINDOW_SIZE,
        min_gap_size=MIN_GAP_SIZE
    )

    # Optional: Visualize the results
    draw_fig(
        imputed_data=imputed_data,
        original_data=univariate_raw_data,
        title=f"Gap Imputation - Combination Method: {COMBINATION_METHOD} - Model: AdaBoost",
    )
    
    # # Save the imputed data
    # dataframe = pd.read_csv('data/CAF003.csv')
    # dataframe['VW_30cm_imputed'] = imputed_data
    # dataframe.to_csv('data/CAF003_imputed.csv', index=False)
    
    print("Imputation completed and saved to data/CAF003_imputed.csv")