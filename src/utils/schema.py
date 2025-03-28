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

class EvaluatorSchema:
    # Evaluation metrics
    RMSE = 'RMSE'
    MAE = 'MAE'

    # Frequency
    YEARLY = 'yearly'
    MONTHLY = 'monthly'
    PER_SPLIT = 'per_split'