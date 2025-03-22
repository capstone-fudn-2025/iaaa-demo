# Import trainers
from trainers.base_trainer import BaseTrainer
from trainers.ml_trainer import MLTrainer
from trainers.dl_trainer import DLTrainer

# Map of model types to trainer classes
TRAINER_MAP = {
    'ml': MLTrainer,
    'dl': DLTrainer
}

def get_trainer(model_type, **kwargs):
    """
    Factory function to get a trainer by model type.
    
    Args:
        model_type (str): Type of the model ('ml' or 'dl').
        **kwargs: Additional arguments to pass to the trainer constructor.
        
    Returns:
        trainer: The initialized trainer.
    """
    if model_type not in TRAINER_MAP:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(TRAINER_MAP.keys())}")
    
    return TRAINER_MAP[model_type](**kwargs)