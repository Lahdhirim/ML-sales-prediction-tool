import json
from pydantic import BaseModel, Field
from typing import Literal

class InferenceConfig(BaseModel):
    nb_months_to_predict: int = Field(..., gt=0, description="Number of months to predict.")
    lvl1_processed_data_path: str = Field(..., description="Path to load Level 1 processed data.")
    processed_data_path: str = Field(..., description="Path to load processed data.")
    saved_models_path: str = Field(..., description="Path to load trained models.")
    n_models: int = Field(..., gt=0, description="Number of trained models to use.")
    weghting_method: Literal["uniform", "weighted"] = Field(..., description="Weighting method (uniform or weighted) to use for predictions (only useful if n_models > 1).")
    raw_predictions_path: str = Field(..., description="Path to save raw predictions.")

def inference_config_loader(config_path: str) -> InferenceConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return InferenceConfig(**config)
