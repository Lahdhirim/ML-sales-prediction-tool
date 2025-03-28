import json
from pydantic import BaseModel, Field
from typing import List, Optional

class PreprocessingConfig(BaseModel):
    raw_data_paths: List[str] = Field(..., min_items=1, description="List of raw data file paths.")
    columns_to_keep: List[str] = Field(..., min_items=1, description="List of columns to keep.")
    unknown_product_id: Optional[List[str]] = Field(default=None, description="List of unknown product IDs to remove.")
    nb_months_to_predict: int = Field(..., gt=0, description="Number of months to predict.")
    processed_data_path: str = Field(..., description="Path to save processed data.")

def preprocessing_config_loader(config_path: str) -> PreprocessingConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return PreprocessingConfig(**config)
