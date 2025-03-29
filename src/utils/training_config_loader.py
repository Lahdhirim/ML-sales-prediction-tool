from pydantic import BaseModel, Field
from typing import Dict, Optional
import json

class SplitterConfig(BaseModel):
    min_training_months: int = Field(..., gt=0, description="Minimum number of months for training.")
    testing_months: int = Field(..., gt=0, description="Number of months for testing (and validating).")

class FeatureSelectorConfig(BaseModel):
    features_path: str = Field(..., description="Path to features.")
    target_column: str = Field(..., description="Target column.")

class ModelConfig(BaseModel):
    enabled: bool = Field(..., description="Enable or disable the model.")
    alpha: Optional[float] = None
    l1_ratio: Optional[float] = None
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    learning_rate: Optional[float] = None
    random_state: Optional[int] = None

class MLPConfig(BaseModel):
    enabled: bool = Field(..., description="Enable or disable the MLP model.")
    hidden_layers_sizes: Optional[list] = None
    activation_function: Optional[str] = None
    solver: Optional[str] = None
    max_iter: Optional[int] = None
    random_state: Optional[int] = None

class ModelsConfig(BaseModel):
    MLModels: Dict[str, ModelConfig]
    MLP: MLPConfig 

class TrainingConfig(BaseModel):
    train_data_path: str = Field(..., description="Path to training data.")
    splitter: SplitterConfig
    features_selector: FeatureSelectorConfig
    models_params: ModelsConfig
    raw_predictions_path: str = Field(..., description="Path to save raw predictions.")
    kpis_path: str = Field(..., description="Filename to save KPIs.")

def training_config_loader(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return TrainingConfig(**config)