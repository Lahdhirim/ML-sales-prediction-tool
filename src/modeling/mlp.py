from sklearn.neural_network import MLPRegressor
import numpy as np
from colorama import Fore, Style
import pandas as pd
from typing import Dict
from src.utils.training_config_loader import MLPConfig

class MLPModel:
    def __init__(self, config: MLPConfig):
        if config.enabled:
            params = {
                'hidden_layer_sizes': config.hidden_layers_sizes or (100, 100),
                'activation': config.activation_function or 'relu',
                'solver': config.solver or 'adam',
                'max_iter': config.max_iter or 10000,
                'early_stopping': config.early_stopping or True,
                'random_state': 42
            }
            self.model = MLPRegressor(**params)
        else:
            self.model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self.model is not None:
            print(f"{Fore.BLUE}Training MLP Model{Style.RESET_ALL}")
            self.model.fit(X_train, y_train)
            return self.model

    def predict(self, X_test: pd.DataFrame, y_test: pd.Series, 
                predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.model is not None:
            y_pred = self.model.predict(X_test)
            predictions['MLP'] = y_pred
        return predictions