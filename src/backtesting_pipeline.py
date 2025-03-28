from src.utils.splitter import TimeSeriesSplitter
import pandas as pd
from colorama import Fore, Style
from src.utils.schema import DatasetSchema

class backtestingPipeline:
    def __init__(self, training_config):
        self.training_config = training_config
    
    def load_processed_data(self):
        try :
            processed_data = pd.read_csv(self.training_config.train_data_path)
        except FileNotFoundError:
            raise FileNotFoundError("Processed data not found at the specified path")
        return processed_data


    def run(self):
        print(Fore.YELLOW + "Running backtesting pipeline..." + Style.RESET_ALL)
        df = self.load_processed_data()

        splitter = TimeSeriesSplitter(min_training_months = self.training_config.splitter["min_training_months"], 
                                      testing_months = self.training_config.splitter["testing_months"])
        for split_index, train, val, test in splitter.split(df):
            print("-" * 40)
            print(f"{Fore.YELLOW}Split {split_index}{Style.RESET_ALL}")
            print("Train:", train[DatasetSchema.YEAR_MONTH].min(), train[DatasetSchema.YEAR_MONTH].max())
            print("Val:", val[DatasetSchema.YEAR_MONTH].min(), val[DatasetSchema.YEAR_MONTH].max())
            print("Test:", test[DatasetSchema.YEAR_MONTH].min(), test[DatasetSchema.YEAR_MONTH].max())
        
        print(Fore.GREEN + "Backtesting pipeline completed successfully" + Style.RESET_ALL)