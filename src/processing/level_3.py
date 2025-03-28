import pandas as pd
from src.utils.schema import DatasetSchema

class Level3Preprocessing():
    
    def tmp_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy[DatasetSchema.MONTH] = data_copy[DatasetSchema.YEAR_MONTH].dt.month
        data_copy[DatasetSchema.SEASON] = data_copy[DatasetSchema.MONTH].apply(lambda x: (x % 12 + 3) // 3)
        return data_copy

    def rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy = data_copy.sort_values(by=[DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH])

        # [MEDIUM]: clean rolling features section
        # Rolling mean
        data_copy[DatasetSchema.ROLLING_MEAN_3M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        data_copy[DatasetSchema.ROLLING_MEAN_6M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)

        # Exponential moving average
        data_copy[DatasetSchema.EMA_3M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].ewm(span=3, min_periods=1).mean().reset_index(0, drop=True)
        data_copy[DatasetSchema.EMA_6M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].ewm(span=6, min_periods=1).mean().reset_index(0, drop=True)

        # Rolling median
        data_copy[DatasetSchema.ROLLING_MEDIAN_3M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].rolling(window=3, min_periods=1).median().reset_index(0, drop=True)
        data_copy[DatasetSchema.ROLLING_MEDIAN_6M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].rolling(window=6, min_periods=1).median().reset_index(0, drop=True)

        # Rolling standard deviation
        data_copy[DatasetSchema.ROLLING_STD_3M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].rolling(window=3, min_periods=2).std().reset_index(0, drop=True)
        data_copy[DatasetSchema.ROLLING_STD_3M] = data_copy[DatasetSchema.ROLLING_STD_3M].fillna(0)
        data_copy[DatasetSchema.ROLLING_STD_6M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].rolling(window=6, min_periods=2).std().reset_index(0, drop=True)
        data_copy[DatasetSchema.ROLLING_STD_6M] = data_copy[DatasetSchema.ROLLING_STD_6M].fillna(0)

        # Rolling sum
        data_copy[DatasetSchema.ROLLING_SUM_3M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
        data_copy[DatasetSchema.ROLLING_SUM_6M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].rolling(window=6, min_periods=1).sum().reset_index(0, drop=True)

        # Percentage change
        data_copy[DatasetSchema.PERCENTAGE_CHANGE_3M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].pct_change(3)
        data_copy[DatasetSchema.PERCENTAGE_CHANGE_3M] = data_copy[DatasetSchema.PERCENTAGE_CHANGE_3M].fillna(0)
        data_copy[DatasetSchema.PERCENTAGE_CHANGE_6M] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].pct_change(6)
        data_copy[DatasetSchema.PERCENTAGE_CHANGE_6M] = data_copy[DatasetSchema.PERCENTAGE_CHANGE_6M].fillna(0)

        # Lag features
        data_copy[DatasetSchema.LAG_3] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].shift(3)
        data_copy[DatasetSchema.LAG_3] = data_copy[DatasetSchema.LAG_3].fillna(0)
        data_copy[DatasetSchema.LAG_6] = data_copy.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].shift(6)
        data_copy[DatasetSchema.LAG_6] = data_copy[DatasetSchema.LAG_6].fillna(0)
        return data_copy

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy = self.tmp_features(data_copy)
        data_copy = self.rolling_features(data_copy)
        return data_copy