# International Conference on Intelligent Aerial Access and Applications (IAAA)

A framework for time series data imputation using bidirectional modeling with various machine learning and deep learning approaches.

## Overview

This project implements a bidirectional modeling approach to time series imputation where both forward and backward prediction models are trained to fill gaps in time series data. The predictions are combined using weighted methods to achieve better accuracy.

## Features

- Support for various machine learning models (AdaBoost, SVR, XGBoost)
- Deep learning models (Attention, CNN1D, GRU)
- Bidirectional prediction strategy
- Customizable weighting strategy (WDBI - Weighted Distance-Based Imputation)
- Comprehensive evaluation metrics
- Visualization tools for imputation results
- Configuration-based experiment setup

## Project Structure

```
iaaa_demo/
│
├── config/                  # Configuration files
│   └── default_config.yaml  # Default settings
│
├── imputation/              # Imputation implementation
│   ├── main_baseline_ML.py  # ML-based imputation
│   └── main_baseline_DL.py  # DL-based imputation
│
├── models/                  # Model definitions
│   ├── dl/                  # Deep learning models
│   │   ├── attention.py
│   │   └── cnn.py
│   ├── ml/                  # Machine learning models
│   │   ├── ada.py
│   │   └── svm.py
│   └── predicts/            # Prediction models
│       └── attention_predict.py
│
├── predict/                 # Prediction scripts
│   ├── predict_Adaboost.py
│   ├── predict_Attention.py
│   ├── predict_CNN1D.py
│   ├── predict_SVR.py
│   └── predict_XGBoost.py
│
├── trainers/                # Model training implementations
│   ├── base_trainer.py
│   ├── dl_trainer.py
│   └── ml_trainer.py
│
├── utils/                   # Utility functions
│   ├── draw_fig.py          # Plotting utilities
│   ├── metrics.py           # Evaluation metrics
│   ├── predict_transform_excel.py  # Results processing
│   ├── preprocessing.py     # Data preprocessing
│   └── utils.py             # General utilities
│
└── main.py                  # Main execution script
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd iaaa_demo
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- XGBoost
- PyYAML

## Usage

### Basic Usage

Run the main script with default configuration:

```bash
python main.py
```

### Using Custom Configuration

Specify a custom configuration file:

```bash
python main.py --config config/my_config.yaml
```

### Command Line Arguments

The main script supports several command line arguments:

- `--config`: Path to configuration YAML file
- `--data_path`: Path to input data file (overrides config)
- `--output_path`: Path for saving imputed data (overrides config)
- `--plots_dir`: Directory for saving plots (overrides config)
- `--model_type`: Model type ('ml' or 'dl', overrides config)
- `--model_name`: Model name (overrides config)

### Configuration Parameters

Key parameters in the configuration file:

- `data_name`: Name of the dataset
- `model_type`: 'ml' for machine learning or 'dl' for deep learning
- `model_name`: Model to use (e.g., 'AdaBoost', 'SVR', 'Attention')
- `window_size`: Size of the sliding window for prediction
- `min_gap_size`: Minimum size of gaps to impute with the model
- `combination_method`: Method to combine forward and backward predictions

## Available Models

### Machine Learning Models
- AdaBoost: Adaptive Boosting Regressor
- SVR: Support Vector Regression
- XGBoost: Extreme Gradient Boosting

### Deep Learning Models
- Attention: Attention-based neural network
- CNN1D: 1D Convolutional Neural Network
- GRU: Gated Recurrent Unit network

## Evaluation Metrics

The framework provides several metrics for evaluating imputation quality:

- Similarity: Measures similarity between true and predicted values
- NMAE: Normalized Mean Absolute Error
- RMSE: Root Mean Square Error
- R2: R-squared (coefficient of determination)
- FSD: Fractional Standard Deviation
- FB: Fractional Bias
- FA2: Factor of 2

## Example

```python
# Set configuration parameters
config = {
    'data_name': 'CAF003',
    'model_type': 'ml',
    'model_name': 'SVR',
    'window_size': 30,
    'min_gap_size': 5,
    'combination_method': 'wdbi'
}

# Run imputation
python main.py --model_type ml --model_name SVR
```

## Output

The framework generates:

1. Imputed data CSV files in the output directory
2. Visualization plots of original vs. imputed data
3. Performance metrics for evaluation
