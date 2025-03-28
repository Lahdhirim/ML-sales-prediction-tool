import pandas as pd
from src.utils.schema import DatasetSchema

class TimeSeriesSplitter:
    def __init__(self, min_training_months: int, testing_months: int):
        self.min_training_months = min_training_months
        self.testing_months = testing_months
    
    def split(self, df: pd.DataFrame):
        
        df[DatasetSchema.YEAR_MONTH] = pd.to_datetime(df[DatasetSchema.YEAR_MONTH], format='%Y-%m').dt.to_period('M')
        
        min_month = df[DatasetSchema.YEAR_MONTH].min()
        max_month = df[DatasetSchema.YEAR_MONTH].max()
        print("Data starts from:", min_month, " to:", max_month)

        month_diff = (max_month - min_month).n
        if month_diff < (self.min_training_months + self.testing_months * 2):
            raise ValueError("Not enough data to split")

        train_start_month = min_month
        train_end_month = train_start_month + self.min_training_months
        val_end_month = train_end_month + self.testing_months
        test_end_month = val_end_month + self.testing_months

        split_index = 1
        while test_end_month <= max_month:
            train = df[(df[DatasetSchema.YEAR_MONTH] >= train_start_month) & (df[DatasetSchema.YEAR_MONTH] < train_end_month)]
            val = df[(df[DatasetSchema.YEAR_MONTH] >= train_end_month) & (df[DatasetSchema.YEAR_MONTH] < val_end_month)]
            test = df[(df[DatasetSchema.YEAR_MONTH] >= val_end_month) & (df[DatasetSchema.YEAR_MONTH] < test_end_month)]

            yield split_index, train, val, test

            train_end_month = val_end_month
            val_end_month = test_end_month
            test_end_month = val_end_month + self.testing_months
            split_index += 1