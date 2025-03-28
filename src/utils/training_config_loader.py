import json
from typing import List,  Optional, Dict

class TrainingConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            config = json.load(file)

        self.train_data_path: str = config.get("train_data_path")
        if not self.train_data_path:
            raise ValueError("train_data_path is required in the config")
        
        self.splitter: Dict = config.get("splitter")
        if not self.splitter:
            raise ValueError("splitter is required in the config")
