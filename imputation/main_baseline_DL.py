import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.draw_fig import draw_fig
from utils.preprocessing import Preprocessing

# Define the 1D CNN model using PyTorch
class Attention(nn.Module):
    name = "Attention"
    def __init__(self, window_size, hidden_channels=64):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.hidden_channels = hidden_channels
        
        # Embedding layer
        self.embedding = nn.Linear(window_size, hidden_channels)
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=1)
        self.fc_1 = nn.Linear(hidden_channels, 2 * hidden_channels)
        self.fc_2 = nn.Linear(2 * hidden_channels, 1)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding(x)
        x, _ = self.attention(x, x, x)
        x = self.fc_1(x)
        x = nn.ReLU()(x)
        x = self.fc_2(x)
        return x.squeeze(-1)

def find_nan_gap(data):
    """To find nan gap from data and return the start and end index of the gap.
    Args:
        data (np.ndarray or pd.Series): The data to be processed.
    
    Returns:
        list: The list (start, end) index of the gap.
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
    """Train a PyTorch model in the forward direction.
    
    Args:
        data (np.ndarray): The training data.
        window_size (int): Size of the sliding window.
        model (nn.Module): The PyTorch model to train.
        
    Returns:
        model: The trained model.
    """
    # Create windowed data
    X, y = create_windows(data, window_size)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).to(model.device)
    y_tensor = torch.FloatTensor(y).to(model.device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model.train()
    num_epochs = 50
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
    
    return model

def train_backward_model(data, window_size, model):
    """Train a PyTorch model in the backward direction.
    
    Args:
        data (np.ndarray): The training data (should already be reversed).
        window_size (int): Size of the sliding window.
        model (nn.Module): The PyTorch model to train.
        
    Returns:
        model: The trained model.
    """
    # Create windowed data (data should already be reversed)
    X, y = create_windows(data, window_size)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).to(model.device)
    y_tensor = torch.FloatTensor(y).to(model.device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    model.train()
    num_epochs = 50
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
    
    return model

def predict_gap_forward(data_before_gap, gap_size, window_size, model):
    """Predict the gap using a forward PyTorch model.
    
    Args:
        data_before_gap (np.ndarray): Data before the gap.
        gap_size (int): Size of the gap to predict.
        window_size (int): Size of the sliding window.
        model (nn.Module): The trained PyTorch model.
        
    Returns:
        np.ndarray: Predicted values for the gap.
    """
    model.eval()  # Set model to evaluation mode
    predictions = []
    current_window = data_before_gap[-window_size:]
    
    with torch.no_grad():  # No need to track gradients
        for _ in range(gap_size):
            # Convert to tensor for prediction
            X_pred = torch.FloatTensor(current_window).unsqueeze(0).to(model.device)  # Add batch dimension
            
            # Make prediction
            pred = model(X_pred).cpu().item()
            predictions.append(pred)
            
            # Update window for next prediction
            current_window = np.append(current_window[1:], pred)
    
    return np.array(predictions)

def predict_gap_backward(data_after_gap, gap_size, window_size, model):
    """Predict the gap using a backward PyTorch model.
    
    Args:
        data_after_gap (np.ndarray): Data after the gap.
        gap_size (int): Size of the gap to predict.
        window_size (int): Size of the sliding window.
        model (nn.Module): The trained PyTorch model.
        
    Returns:
        np.ndarray: Predicted values for the gap (in correct order).
    """
    model.eval()  # Set model to evaluation mode
    # The model was trained on reversed data, so we need to reverse data_after_gap
    # Use .copy() to ensure the reversed array has positive strides
    reversed_data = data_after_gap[::-1].copy()
    
    predictions = []
    current_window = reversed_data[:window_size].copy()  # Take the first window from reversed data
    
    with torch.no_grad():  # No need to track gradients
        for _ in range(gap_size):
            # Convert to tensor for prediction
            X_pred = torch.FloatTensor(current_window).unsqueeze(0).to(model.device)  # Add batch dimension
            
            # Make prediction
            pred = model(X_pred).cpu().item()
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

def imputed_missing_gaps(data, window_size=7, min_gap_size=5, combination_method='wdbi', device='cpu'):
    # Make a copy to avoid modifying the original data
    imputed_data = deepcopy(data)
    
    # Find gaps of NaN values
    gaps = find_nan_gap(imputed_data)
    
    # Time index
    time_index = np.arange(len(data))
    
    # First pass: fill all gaps with interpolation
    non_nan_indices = np.nonzero(~np.isnan(imputed_data))[0]
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
            print(f"Training forward model for gap of size {gap_size}")

            # Initialize 1D CNN model for forward prediction
            forward_model = Attention(window_size=window_size).to(device)
            
            # Train forward model
            forward_model_trained = train_forward_model(data_before_gap, window_size, forward_model)
            
            # Predict using forward model
            forward_model_pred = predict_gap_forward(data_before_gap, gap_size, window_size, forward_model_trained)
        
        # Train and predict with backward models if enough data
        if has_enough_backward_data:
            print(f"Training backward model for gap of size {gap_size}")
            
            # Initialize 1D CNN model for backward prediction
            backward_model = Attention(window_size=window_size).to(device)
            
            # Train backward model (reverse the data first)
            reversed_data_after_gap = data_after_gap[::-1].copy()  # Create explicit copy to avoid negative strides
            backward_model_trained = train_backward_model(reversed_data_after_gap, window_size, backward_model)
            
            # Predict using backward model
            backward_model_pred = predict_gap_backward(data_after_gap, gap_size, window_size, backward_model_trained)

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
    
    # Set PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    dataframe = pd.read_csv('data/CAF003.csv')
    # Drop unnecessary columns
    dataframe = dataframe.drop(columns=['Location', 'Date'])
    for col in dataframe.columns:
        univariate_raw_data = dataframe[[col]]
        print(f"Imputing missing values for column: {col}")
        
        # Preprocessing data
        preprocessing = Preprocessing()
        preprocessed_data = preprocessing.flow(univariate_raw_data)
        # univariate_raw_data = preprocessed_data[col]
        data = deepcopy(preprocessed_data[col].values)
        
        # Run imputation with 1D CNN models
        imputed_data = imputed_missing_gaps(
            data,
            combination_method=COMBINATION_METHOD,
            window_size=WINDOW_SIZE,
            min_gap_size=MIN_GAP_SIZE,
            device=device
        )
    
        # Revert preprocessing
        imputed_data = preprocessing.reverse_flow(pd.DataFrame(imputed_data, columns=[col]))[col].values

        # Optional: Visualize the results
        draw_fig(
            imputed_data=imputed_data,
            original_data=univariate_raw_data,
            title=f"Gap Imputation - Combination Method: {COMBINATION_METHOD} - {col} - Model: Attention",
            save_path=f"plots/imputation_{col}.png",
            is_show_fig=False
        )
        
        # Save the temporarily imputed data
        dataframe[col] = imputed_data
        # If not exist file, create new file else append to existing file
        dataframe.to_csv('data/CAF003_imputed.csv', index=False)        
            
    print("Imputation completed and saved to data/CAF003_imputed.csv")