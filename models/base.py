from abc import ABC, abstractmethod
from copy import deepcopy

class BaseModel(ABC):
    """Base class for all models used in time series imputation."""
    
    @property
    @abstractmethod
    def name(self):
        """Return the name of the model."""
        pass
    
    @abstractmethod
    def get_model(self):
        """Return a copy of the model for training."""
        pass