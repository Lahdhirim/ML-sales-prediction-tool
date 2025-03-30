import json
from pydantic import BaseModel, Field

class InferenceConfig(BaseModel):
    nb_months_to_predict: int = Field(..., gt=0, description="Number of months to predict.")
    lvl1_processed_data_path: str = Field(..., description="Path to load Level 1 processed data.")
    processed_data_path: str = Field(..., description="Path to load processed data.")
    saved_models_path: str = Field(..., description="Path to load trained models.")
    n_models: int = Field(..., gt=0, description="Number of trained models to use.")
    weghting_method: str = Field(..., description="Weighting method to use for predictions.")

def inference_config_loader(config_path: str) -> InferenceConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return InferenceConfig(**config)
