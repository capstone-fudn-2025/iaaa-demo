# Import models for easier access
from models.ml.ada import AdaBoostModel
from models.ml.svm import SVRModel
from models.dl.attention import Attention
from models.dl.cnn import CNN1D

# Map of model names to model classes
MODEL_MAP = {
    'AdaBoost': AdaBoostModel,
    'SVR': SVRModel,
    'Attention': Attention,
    'CNN1D': CNN1D,
    
}

def get_model(model_name, **kwargs):
    """
    Factory function to get a model by name.
    
    Args:
        model_name (str): Name of the model to get.
        **kwargs: Additional arguments to pass to the model constructor.
        
    Returns:
        model: The initialized model.
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_MAP.keys())}")
    
    return MODEL_MAP[model_name](**kwargs)