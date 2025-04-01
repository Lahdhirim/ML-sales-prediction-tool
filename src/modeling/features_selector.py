import pandas as pd
import json
from typing import Tuple
from src.utils.training_config_loader import FeatureSelectorConfig

class FeatureSelector():
    """
    A class to select features and the target column from a dataset.
    """
    def __init__(self, features_selector_config: FeatureSelectorConfig):
        self.features_path = features_selector_config.features_path
        self.target_column = features_selector_config.target_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        with open(self.features_path, 'r') as json_file:
            features = json.load(json_file)["features"]
        assert all(feature in X.columns for feature in features), "Some features are not present in the dataset"
        return X[features], X[self.target_column]
