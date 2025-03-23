import os
import sys
import argparse
import yaml
import pandas as pd
from copy import deepcopy
import torch

from imputer import Imputer
from trainers import get_trainer
from models import get_model
from utils.preprocessing import Preprocessing
from utils.draw_fig import draw_fig
from utils.metrics import calculate_metrics
from utils.utils import seed_everything
import matplotlib

matplotlib.use('Agg')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Time series imputation')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--data_path', type=str,
                        help='Path to the data CSV file (overrides config)')
    parser.add_argument('--output_path', type=str,
                        help='Path to save the imputed data (overrides config)')
    parser.add_argument('--plots_dir', type=str,
                        help='Directory to save the plots (overrides config)')
    parser.add_argument('--model_type', type=str, choices=['ml', 'dl'],
                        help='Model type: ml (machine learning) or dl (deep learning) (overrides config)')
    parser.add_argument('--model_name', type=str,
                        help='Model name: AdaBoost, SVR, or Attention (overrides config)')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    if args.data_path:
        config['data_path'] = args.data_path
    if args.output_path:
        config['output_path'] = args.output_path
    if args.plots_dir:
        config['plots_dir'] = args.plots_dir
    if args.model_type:
        config['model_type'] = args.model_type
    if args.model_name:
        config['model_name'] = args.model_name
    
    return config

def main():
    """Main function for time series imputation."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Extract configuration variables
    data_name = config['data_name']
    data_path = config['data_path']
    output_path = config['output_path']
    plots_dir = config['plots_dir']
    model_type = config['model_type']
    model_name = config['model_name']
    window_size = config['window_size']
    min_gap_size = config['min_gap_size']
    combination_method = config['combination_method']
    seed = config['seed']
    columns = config.get('columns', None)
    
    seed_everything(seed)
    print(f"Seed everything with seed: {seed}")
    # Add model name to output path
    output_path = os.path.join(output_path, model_name, str(window_size), f"{data_name}_imputed.csv")
    # Create output directories
    os.makedirs(os.path.join(plots_dir, model_name, str(window_size)), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Device for DL models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model name: {model_type} - {model_name}")
    print(f"Using device: {device}")
    print(f"With window size: {window_size}")
    
    # Load data
    dataframe = pd.read_csv(f"{data_path}/{data_name}.csv")
    
    # Drop unnecessary columns if they exist
    if 'Location' in dataframe.columns:
        dataframe = dataframe.drop(columns=['Location'])
    if 'Date' in dataframe.columns:
        dataframe = dataframe.drop(columns=['Date'])
    
    # Select columns to process
    if columns:
        columns = [col for col in columns if col in dataframe.columns]
    else:
        columns = dataframe.columns
        
    # Create imputer
    imputer = Imputer(window_size=window_size, 
                      min_gap_size=min_gap_size, 
                      combination_method=combination_method)
    
    # Get trainer
    trainer = get_trainer(model_type, window_size=window_size)
    
    # Process each column
    for col in columns:
        print(f"\nProcessing column: {col}")
        
        # Get univariate data
        univariate_raw_data = dataframe[[col]]
        
        # Preprocessing data
        preprocessing = Preprocessing()
        preprocessed_data = preprocessing.flow(univariate_raw_data)
        data = deepcopy(preprocessed_data[col].values)
        
        # Get model instances
        if model_type == 'ml':
            forward_model = get_model(model_name).get_model()
            backward_model = get_model(model_name).get_model()
        else:  # model_type == 'dl'
            forward_model = get_model(model_name, window_size=window_size)
            backward_model = get_model(model_name, window_size=window_size)
        
        # Run imputation
        imputed_data = imputer.impute(data, forward_model, backward_model, trainer)
        
        # Revert preprocessing
        imputed_data = preprocessing.reverse_flow(pd.DataFrame(imputed_data, columns=[col]))[col].values
        
        # Visualize the results
        draw_fig(
            imputed_data=imputed_data,
            original_data=univariate_raw_data[col].values,
            title=f"Gap Imputation - {combination_method} - {col} - Model: {model_name}",
            save_path=f"{plots_dir}/{model_name}/{window_size}/{col} - {combination_method}.png",
            is_show_fig=False
        )
        
        # Update the dataframe with imputed values
        dataframe[col] = imputed_data
    
    # Save the imputed data
    dataframe.to_csv(output_path, index=False)
    print(f"\nImputation completed and saved to {output_path}")

if __name__ == "__main__":
    model_dict = {
        'ml': ['AdaBoost', 'SVR', 'XGBoost'],
        'dl': ['Attention', 'CNN1D']
    }

    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    for model_type, models in model_dict.items():
        for model_name in models:
            # Reset sys.argv to original
            sys.argv = original_argv.copy()
            
            # Add or modify the model type and name arguments
            if '--model_type' in sys.argv:
                type_index = sys.argv.index('--model_type')
                sys.argv[type_index + 1] = model_type
            else:
                sys.argv.extend(['--model_type', model_type])
                
            if '--model_name' in sys.argv:
                name_index = sys.argv.index('--model_name')
                sys.argv[name_index + 1] = model_name
            else:
                sys.argv.extend(['--model_name', model_name])
            
            # Run the main function with the modified arguments
            print(f"\n\n{'='*50}")
            print(f"Running with model_type={model_type}, model_name={model_name}")
            print(f"{'='*50}\n")
            main()