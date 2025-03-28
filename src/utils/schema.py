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

    # Rolling features
    ROLLING_MEAN_3M = 'rolling_mean_3m'
    ROLLING_MEAN_6M = 'rolling_mean_6m'
    ROLLING_MEDIAN_3M = 'rolling_median_3m'
    ROLLING_MEDIAN_6M = 'rolling_median_6m'
    ROLLING_STD_3M = 'rolling_std_3m'
    ROLLING_STD_6M = 'rolling_std_6m'
    ROLLING_SUM_3M = 'rolling_sum_3m'
    ROLLING_SUM_6M = 'rolling_sum_6m'
    PERCENTAGE_CHANGE_3M = 'percentage_change_3m'
    PERCENTAGE_CHANGE_6M = 'percentage_change_6m'
    LAG_3 = 'lag_3'
    LAG_6 = 'lag_6'
    EMA_3M = 'ema_3m'
    EMA_6M = 'ema_6m'

class EvaluatorSchema:
    # Evaluation metrics
    RMSE = 'RMSE'
    MAE = 'MAE'

    # Frequency
    YEARLY = 'yearly'
    MONTHLY = 'monthly'
    PER_SPLIT = 'per_split'