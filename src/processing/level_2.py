import pandas as pd
from src.utils.processing_config_loader import PreprocessingConfig
from colorama import Fore, Style
from src.utils.schema import DatasetSchema

class Level2Preprocessing():
    def __init__(self, processing_config: PreprocessingConfig):
        self.processing_config = processing_config
    
    def add_label(self, data: pd.DataFrame) -> pd.DataFrame:
        data[DatasetSchema.YEAR_MONTH] = data[DatasetSchema.DATE].dt.to_period("M")
        transactions_per_customer_month = data.groupby([DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH]).size().reset_index(name=DatasetSchema.NB_TRANSACTIONS)
        transactions_per_customer_month[DatasetSchema.FUTURE_TRANSACTIONS] = transactions_per_customer_month.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].shift(-self.processing_config.nb_months_to_predict)
        transactions_per_customer_month = transactions_per_customer_month.dropna(subset=[DatasetSchema.FUTURE_TRANSACTIONS])
        transactions_per_customer_month[DatasetSchema.FUTURE_TRANSACTIONS] = transactions_per_customer_month[DatasetSchema.FUTURE_TRANSACTIONS].astype(int)
        return transactions_per_customer_month

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy = data_copy[~data_copy[DatasetSchema.PRODUCT_ID].isin([self.processing_config.unknown_product_id])]
        data_copy = self.add_label(data_copy)
        return data_copy