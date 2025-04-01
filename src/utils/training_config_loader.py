from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import json

class SplitterConfig(BaseModel):
    min_training_months: int = Field(..., gt=0, description="Minimum number of months for training.")
    testing_months: int = Field(..., gt=0, description="Number of months for testing (and validating).")

class ClusteringProcessorConfig(BaseModel):
    product_mapping_path: str = Field(..., description="Path to product mapping file.")
    lvl1_processed_data_path: str = Field(..., description="Path to load Level 1 processed data.")
    features: Dict[str, List[str]]
    max_clusters: int = Field(..., gt=0, description="Max number of clusters to try during Elbow method.")
    default_cluster_size: int = Field(..., gt=0, description="Number of clusters to use if automatic optimal detection fails.")


class FeatureSelectorConfig(BaseModel):
    features_path: str = Field(..., description="Path to features.")
    target_column: str = Field(..., description="Target column.")

class ModelConfig(BaseModel):
    enabled: bool = Field(..., description="Enable or disable the model.")
    alpha: Optional[float] = Field(None, description="Alpha value for ElasticNet model.")
    l1_ratio: Optional[float] = Field(None, description="L1 ratio for ElasticNet model.")
    n_estimators: Optional[int] = Field(None, description="Number of estimators.")
    max_depth: Optional[int] = Field(None, description="Max depth of the tree.")
    learning_rate: Optional[float] = Field(None, description="Learning rate for models like XGBoost.")
    random_state: Optional[int] = Field(None, description="Random state for reproducibility.")
    n_neighbors: Optional[int] = Field(None, description="Number of neighbors for KNeighborsRegressor.")
    weights: Optional[str] = Field(None, description="Weights for KNeighborsRegressor.")

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
    clustering_processor: ClusteringProcessorConfig
    features_selector: FeatureSelectorConfig
    models_params: ModelsConfig
    raw_predictions_path: str = Field(..., description="Path to save raw predictions.")
    kpis_path: str = Field(..., description="Filename to save KPIs.")
    trained_models_path: str = Field(..., description="Path to save trained models.")

def training_config_loader(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return TrainingConfig(**config)