import pandas as pd
from src.utils.schema import DatasetSchema
from src.utils.processing_config_loader import PreprocessingConfig

class Level3Preprocessing():
    """
    Applies time series preprocessing steps such as temporal features, rolling statistics, 
    and lag-based features.
    """
    def __init__(self, processing_config: PreprocessingConfig):
        self.processing_config = processing_config
    
    def tmp_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy[DatasetSchema.MONTH] = data_copy[DatasetSchema.YEAR_MONTH].dt.month
        data_copy[DatasetSchema.SEASON] = data_copy[DatasetSchema.MONTH].apply(lambda x: (x % 12 + 3) // 3)
        return data_copy

    def rolling_features(self, data: pd.DataFrame, column:str) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy = data_copy.sort_values(by=[DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH])

        windows = list(range(self.processing_config.min_window_size, self.processing_config.max_window_size + 1))
        methods = ['mean', 'median', 'std', 'sum']
        for window in windows:
            for method in methods:
                feature_name = f"{DatasetSchema.ROLLING}_{column}_{method}_{window}m"
                data_copy[feature_name] = (
                    data_copy.groupby(DatasetSchema.CUSTOMER_ID)[column]
                    .shift(self.processing_config.nb_months_to_predict) # To exclude values that are not available during inference
                    .rolling(window=window, min_periods=1 if method != 'std' else 2)
                    .agg(method)
                    .reset_index(0, drop=True)
                )
                
                data_copy[feature_name] = data_copy[feature_name].fillna(0)
        
        return data_copy
    
    def lag_features(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy = data_copy.sort_values(by=[DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH])

        lags = list(range(self.processing_config.min_window_size, self.processing_config.max_window_size + 1))
        for lag in lags:
            lag_feature_name = f"{DatasetSchema.LAG}_{column}_{lag}m"
            data_copy[lag_feature_name] = (
                data_copy.groupby(DatasetSchema.CUSTOMER_ID)[column]
                .shift(lag)
                .fillna(0)
            )
        
        return data_copy

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy = self.tmp_features(data_copy)
        data_copy = self.rolling_features(data_copy, column=DatasetSchema.NB_TRANSACTIONS)
        data_copy = self.rolling_features(data_copy, column=DatasetSchema.FUTURE_TRANSACTIONS)
        data_copy = self.lag_features(data_copy, column=DatasetSchema.NB_TRANSACTIONS)
        data_copy = self.lag_features(data_copy, column=DatasetSchema.FUTURE_TRANSACTIONS)
        return data_copy