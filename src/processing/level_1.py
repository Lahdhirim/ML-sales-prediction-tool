import pandas as pd
from src.utils.processing_config_loader import PreprocssingConfig
from colorama import Fore, Style
from src.utils.schema import DatasetSchema

class Level1Preprocessing():
    def __init__(self, processing_config: PreprocssingConfig):
        self.processing_config = processing_config
    
    def load_data(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for path in self.processing_config.raw_data_paths:
            try:
                data = pd.concat([data, pd.read_csv(path)])
            except FileNotFoundError:
                print(Fore.RED + f"File not found: {path}" + Style.RESET_ALL)
                continue 
        return data

    def convert_date(self, data: pd.DataFrame) -> pd.DataFrame:
        data[DatasetSchema.DATE] = pd.to_datetime(data[DatasetSchema.DATE], errors="coerce")
        return data

    def run(self):
        data = self.load_data()
        data = data[self.processing_config.columns_to_keep]
        data = data.drop_duplicates()
        data = self.convert_date(data)
        return data

