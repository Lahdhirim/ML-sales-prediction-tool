import json
from typing import List,  Optional

class PreprocssingConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            config = json.load(file)

        self.raw_data_paths: List[str] = config.get("raw_data_path")
        if not self.raw_data_paths:
            raise ValueError("raw_data_paths cannot be None or empty.")
        
        self.columns_to_keep: List[str] = config.get("columns_to_keep")
        if not self.columns_to_keep:
            raise ValueError("columns_to_keep cannot be None or empty.")

        self.unknown_product_id: Optional[str] = config.get("unknown_product_id")

        self.nb_months_to_predict: int = config.get("nb_months_to_predict")
        if not self.nb_months_to_predict or self.nb_months_to_predict < 0:
            raise ValueError("nb_months_to_predict cannot be None or negative.")
        
        self.processed_data_path: str = config.get("processed_data_path")
