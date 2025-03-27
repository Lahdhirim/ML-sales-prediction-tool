from src.utils.processing_config_loader import PreprocssingConfig
from src.processing.level_1 import Level1Preprocessing
from src.processing.level_2 import Level2Preprocessing
import pandas as pd
from colorama import Fore, Style

class processingPipeline():
    def __init__(self, processing_config: PreprocssingConfig):
        self.processing_config = processing_config

    def run(self):
        print(Fore.YELLOW + "Running processing pipeline..." + Style.RESET_ALL)
        data = Level1Preprocessing(self.processing_config).run()
        print("Shape of the data after level 1 preprocessing: ", data.shape)
        data = Level2Preprocessing(self.processing_config).run(data)
        print("Shape of the data after level 2 preprocessing: ", data.shape)
        data.to_csv(self.processing_config.processed_data_path, index=False)
        print(Fore.GREEN + "Processing pipeline completed successfully" + Style.RESET_ALL)
        
