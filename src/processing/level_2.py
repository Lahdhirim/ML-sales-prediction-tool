import pandas as pd
from src.utils.processing_config_loader import PreprocessingConfig
from src.utils.schema import DatasetSchema

class Level2Preprocessing():
    def __init__(self, processing_config: PreprocessingConfig):
        self.processing_config = processing_config

    def calculate_monthly_transactions(self, data: pd.DataFrame) -> pd.DataFrame:
        data[DatasetSchema.YEAR_MONTH] = data[DatasetSchema.DATE].dt.to_period("M")
        transactions_per_customer_month = data.groupby([DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH]).size().reset_index(name=DatasetSchema.NB_TRANSACTIONS)
        return transactions_per_customer_month

    def fill_values(self, data: pd.DataFrame) -> pd.DataFrame:
        result_list = []
        for customer in data[DatasetSchema.CUSTOMER_ID].unique():
            customer_data = data[data[DatasetSchema.CUSTOMER_ID] == customer]

            min_month = customer_data[DatasetSchema.YEAR_MONTH].min().to_timestamp()
            max_month = customer_data[DatasetSchema.YEAR_MONTH].max().to_timestamp()
            all_months = pd.date_range(start=min_month, end=max_month, freq='MS')
            
            customer_full = pd.DataFrame({DatasetSchema.CUSTOMER_ID: customer, DatasetSchema.YEAR_MONTH: all_months})

            customer_data[DatasetSchema.YEAR_MONTH] = customer_data[DatasetSchema.YEAR_MONTH].dt.to_timestamp()
            customer_merged = pd.merge(customer_full, customer_data, on=[DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH], how='left')
            
            customer_merged[DatasetSchema.NB_TRANSACTIONS] = customer_merged[DatasetSchema.NB_TRANSACTIONS].fillna(0)
            
            result_list.append(customer_merged)

        final_df = pd.concat(result_list, ignore_index=True)
        final_df[DatasetSchema.YEAR_MONTH] = final_df[DatasetSchema.YEAR_MONTH].dt.to_period("M")
        final_df[DatasetSchema.NB_TRANSACTIONS] = final_df[DatasetSchema.NB_TRANSACTIONS].astype(int)
        return final_df

    
    def add_label(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values([DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH])
        data[DatasetSchema.FUTURE_TRANSACTIONS] = data.groupby(DatasetSchema.CUSTOMER_ID)[DatasetSchema.NB_TRANSACTIONS].apply(lambda x: x.rolling(window=self.processing_config.nb_months_to_predict).sum().shift(-self.processing_config.nb_months_to_predict)).reset_index(drop=True)
        data = data.dropna(subset=[DatasetSchema.FUTURE_TRANSACTIONS])
        data[DatasetSchema.FUTURE_TRANSACTIONS] = data[DatasetSchema.FUTURE_TRANSACTIONS].astype(int)
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()
        data_copy = data_copy[~data_copy[DatasetSchema.PRODUCT_ID].isin([self.processing_config.unknown_product_id])]
        data_copy = self.calculate_monthly_transactions(data_copy)
        data_copy = self.fill_values(data_copy)
        data_copy = self.add_label(data_copy)
        return data_copy