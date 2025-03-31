class DatasetSchema:
    # Main columns
    CUSTOMER_ID = 'customer_id'
    PRODUCT_ID = 'product_id'
    DATE = 'date'

    # Added columns
    YEAR_MONTH = 'year_month'
    NB_TRANSACTIONS = 'nb_transactions'
    FUTURE_TRANSACTIONS = 'future_transactions'
    SPLIT_INDEX = 'split_index'
    YEAR = 'year'
    MONTH = 'month'
    SEASON = 'season'
    ROLLING = 'rolling'
    LAG = 'lag'
    CLUSTER_ID = 'cluster_id'
    PREDICTION_AGGREGATED = 'prediction_aggregated'

class MappingSchema:
    GROUP = 'group'
    COUNTRY = 'country'
    CATEGORY = 'category'

class EvaluatorSchema:
    # Evaluation metrics
    RMSE = 'RMSE'
    MAE = 'MAE'

    # Frequency
    YEARLY = 'yearly'
    MONTHLY = 'monthly'
    PER_SPLIT = 'per_split'

class PipelinesDictSchema:
    CLUSTERIING_PROCESSOR = "clustering_processor"
    FEATURE_SELECTOR = "features_selector"
    SCALER = "scaler"
    ML_MODELS = "ml_models"
    MLP_MODEL = "mlp_model"