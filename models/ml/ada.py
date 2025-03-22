from sklearn.ensemble import AdaBoostRegressor
from copy import deepcopy
from models.base import BaseModel

class AdaBoostModel(BaseModel):
    """
    AdaBoost regressor model for time series imputation.
    """
    @property
    def name(self):
        return "AdaBoost"
    
    def __init__(self, n_estimators=50, random_state=42):
        """
        Initialize the AdaBoost regressor model.
        
        Args:
            n_estimators (int): Number of estimators for AdaBoost.
            random_state (int): Random state for reproducibility.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = AdaBoostRegressor(n_estimators=n_estimators, random_state=random_state)
    
    def get_model(self):
        """
        Get a copy of the model for training.
        
        Returns:
            AdaBoostRegressor: A copy of the model.
        """
        return deepcopy(self.model)