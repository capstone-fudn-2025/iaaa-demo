from sklearn.svm import SVR
from copy import deepcopy
from models.base import BaseModel

class SVRModel(BaseModel):
    """
    Support Vector Regression model for time series imputation.
    """
    @property
    def name(self):
        return "SVR"
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        """
        Initialize the SVR model.
        
        Args:
            kernel (str): Kernel type for SVR ('linear', 'poly', 'rbf', 'sigmoid').
            C (float): Regularization parameter.
            epsilon (float): Epsilon in the epsilon-SVR model.
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    
    def get_model(self):
        """
        Get a copy of the model for training.
        
        Returns:
            SVR: A copy of the model.
        """
        return deepcopy(self.model)