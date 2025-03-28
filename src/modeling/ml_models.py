from sklearn.linear_model import LinearRegression, ElasticNet
from colorama import Fore, Style
from src.utils.training_config_loader import ModelsConfig
import pandas as pd
from typing import Dict
from sklearn.base import RegressorMixin
import numpy as np

class MLModels:
    def __init__(self, config: ModelsConfig):
        self.models = {}
        for model_name, model_config in config.items():
            if model_config.enabled:
                if model_name == "LinearRegression":
                    self.models["LinearRegression"] = LinearRegression()
                elif model_name == "ElasticNet":
                    params = {k: v for k, v in model_config.dict().items() if k != "enabled" and v is not None}
                    self.models["ElasticNet"] = ElasticNet(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, RegressorMixin]:
        for name, model in self.models.items():
            print(f"{Fore.BLUE}Training {name}{Style.RESET_ALL}")
            model.fit(X_train, y_train)
        return self.models
    
    def predict(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
        return predictions