import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate similarity metric between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        T: Number of missing values
    
    Returns:
        Similarity score in range [0,1] (higher is better)
    """
    # Get max and min of predicted values
    y_max = np.max(y_pred)
    y_min = np.min(y_pred)
    
    # Avoid division by zero
    denominator = y_max - y_min
    if denominator == 0:
        denominator = np.finfo(float).eps
        
    # Calculate similarity according to formula
    similarities = 1 / (1 + np.abs(y_true - y_pred) / denominator)
    
    # Return average similarity
    return np.sum(similarities) / y_pred.shape[0]

def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Normalized Mean Absolute Error (NMAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        masks: Binary mask where 1 indicates observed value and 0 indicates missing value
    
    Returns:
        NMAE value (lower is better)
    """
    # Get Vmax and Vmin from non-missing values in the original time series
    v_max = np.max(y_true)
    v_min = np.min(y_true)
    
    # Avoid division by zero
    denominator = v_max - v_min
    if denominator == 0:
        denominator = np.finfo(float).eps
    

    abs_diff = np.abs(y_true - y_pred)
    
    return np.sum(abs_diff / denominator) / y_pred.shape[0]

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R-squared value (higher is better)
    """
    # Calculate means
    y_true_mean = np.mean(y_true)
    
    # Numerator: sum of (x_i - x_mean)(y_i - y_mean)
    numerator = np.sum((y_true - y_true_mean) * (y_pred - np.mean(y_pred)))
    
    # Denominators: sqrt of sum of (x_i - x_mean)^2 and sum of (y_i - y_mean)^2
    denominator_x = np.sqrt(np.sum((y_true - y_true_mean) ** 2))
    denominator_y = np.sqrt(np.sum((y_pred - np.mean(y_pred)) ** 2))
    
    # Calculate Pearson correlation coefficient R
    R = numerator / (denominator_x * denominator_y)
    
    # R-squared is the square of the Pearson correlation coefficient
    return R ** 2

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value (lower is better)
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def fsd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Fractional Standard Deviation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        FSD value (lower is better)
    """
    y_true_sd = np.std(y_true)
    y_pred_sd = np.std(y_pred)
    denominator = y_true_sd + y_pred_sd
    # Avoid division by zero
    if denominator == 0:
        return 0.0
    return 2 * np.abs(y_true_sd - y_pred_sd) / denominator

def fb(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Fractional Bias.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        FB value (lower is better)
    """
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    denominator = y_pred_mean + y_true_mean
    # Avoid division by zero
    if denominator == 0:
        return 0.0
    return 2 * (y_true_mean - y_pred_mean) / denominator

def fa2(y_true: np.ndarray, y_pred: np.ndarray, upper_bound: float = 2.0, lower_bound: float = 0.5) -> float:
    """Calculate Factor of 2 (FA2).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        upper_bound: Upper bound for FA2
        lower_bound: Lower bound for FA2
        
    Returns:
        FA2 value (higher is better)
    """
    # Avoid division by zero by adding small epsilon where y_true is zero
    ratio = np.divide(y_pred, y_true, out=np.zeros_like(y_pred), where=y_true!=0)
    return np.mean((ratio >= lower_bound) & (ratio <= upper_bound))


def calculate_metrics(original_data, imputed_data, gaps=None):
    """
    Calculate evaluation metrics for imputation.
    
    Args:
        original_data (np.ndarray or pd.Series): The original data without gaps (ground truth).
        imputed_data (np.ndarray or pd.Series): The imputed data.
        gaps (list, optional): List of (start, end) pairs indicating the gaps.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Convert to numpy arrays if they are pandas Series
    if isinstance(original_data, pd.Series):
        original_data = original_data.values
    if isinstance(imputed_data, pd.Series):
        imputed_data = imputed_data.values
    
    # If gaps are provided, only evaluate at gap positions
    if gaps:
        mask = np.zeros_like(original_data, dtype=bool)
        for start, end in gaps:
            mask[start:end] = True
        
        original_gaps = original_data[mask]
        imputed_gaps = imputed_data[mask]
    else:
        # Otherwise, use only non-NaN values for evaluation
        mask = ~np.isnan(original_data)
        original_gaps = original_data[mask]
        imputed_gaps = imputed_data[mask]
    
    # Calculate metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(original_gaps, imputed_gaps)),
        'MAE': mean_absolute_error(original_gaps, imputed_gaps),
        'R2': r2_score(original_gaps, imputed_gaps)
    }
    
    return metrics