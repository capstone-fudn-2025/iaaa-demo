from xgboost import XGBRegressor
from copy import deepcopy
from models.base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost model for time series imputation.
    """
    @property
    def name(self):
        return "XGBoost"
    
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, objective='reg:squarederror'):
        """
        Initialize the XGBoost model.
        
        Args:
            max_depth (int): Maximum tree depth for boosting.
            learning_rate (float): Boosting learning rate.
            n_estimators (int): Number of boosting rounds.
            objective (str): Learning task and objective.
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective=objective)
    
    def get_model(self):
        """
        Get a copy of the model for training.
        
        Returns:
            XGBRegressor: A copy of the model.
        """
        return deepcopy(self.model)