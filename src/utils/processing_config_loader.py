import json
from pydantic import BaseModel, Field
from typing import List, Optional

class PreprocessingConfig(BaseModel):
    raw_data_paths: List[str] = Field(..., min_items=1, description="List of raw data file paths.")
    columns_to_keep: List[str] = Field(..., min_items=1, description="List of columns to keep.")
    unknown_product_id: Optional[List[str]] = Field(default=None, description="List of unknown product IDs to remove.")
    nb_months_to_predict: int = Field(..., gt=0, description="Number of months to predict.")
    lvl1_processed_data_path: str = Field(..., description="Path to save Level 1 processed data (useful for clustering).")
    processed_data_path: str = Field(..., description="Path to save processed data.")
    max_window_size: int = Field(..., ge=2, description="Maximum value for window size to calculate rolling features.")
    min_window_size: int = Field(..., ge=2, description="Minimum value for window size to calculate rolling features.")
    max_lag: int = Field(..., ge=3, description="Maximum value for lag to calculate lag values.")
    min_lag: int = Field(..., ge=3, description="Minimum value for lag to calculate lag values.")

def preprocessing_config_loader(config_path: str) -> PreprocessingConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return PreprocessingConfig(**config)
