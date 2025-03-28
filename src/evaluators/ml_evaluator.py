import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from colorama import Fore, Style
from typing import List
from src.utils.training_config_loader import TrainingConfig
from src.utils.schema import DatasetSchema, EvaluatorSchema

class MLEvaluator:

    def __init__(self, config: TrainingConfig, models: List[str]):
        self.target_column = config.features_selector["target_column"]
        self.models = models
        self.ouput_filename = config.kpis_filename

    def _compute_metrics(self, df):
        metrics = {}
        for model in self.models:
            y_true = df[self.target_column]
            y_pred = df[model]
            metrics[model] = {
                EvaluatorSchema.RMSE: mean_squared_error(y_true, y_pred) ** 0.5,
                EvaluatorSchema.MAE: mean_absolute_error(y_true, y_pred)
            }
        return metrics

    @staticmethod
    def result_formatter(metrics, ouput_filename):
        with pd.ExcelWriter(ouput_filename) as writer:
            for frequency, kpis in metrics.items():
                df = pd.DataFrame(kpis.tolist(), index=kpis.index)
                df_flat = df.apply(lambda row: pd.Series({f"{model}_{metric}": value 
                                                        for model, metrics in row.items() 
                                                        for metric, value in metrics.items()}), axis=1)
                df_flat.to_excel(writer, sheet_name=frequency)
            print(f"KPIs Results saved to {ouput_filename}")

    def evaluate(self, df):
        metrics = {}
        results_df = df.copy()

        results_df[DatasetSchema.YEAR] = results_df[DatasetSchema.YEAR_MONTH].dt.year
        yearly_metrics = results_df.groupby(DatasetSchema.YEAR).apply(self._compute_metrics)
        metrics[EvaluatorSchema.YEARLY] = yearly_metrics

        results_df[DatasetSchema.MONTH] = results_df[DatasetSchema.YEAR_MONTH].dt.month
        monthly_metrics = results_df.groupby([DatasetSchema.YEAR, DatasetSchema.MONTH]).apply(self._compute_metrics)
        metrics[EvaluatorSchema.MONTHLY] = monthly_metrics

        per_split_metrics = results_df.groupby(DatasetSchema.SPLIT_INDEX).apply(self._compute_metrics)
        metrics[EvaluatorSchema.PER_SPLIT] = per_split_metrics

        # Save results to file
        self.result_formatter(metrics = metrics, ouput_filename = self.ouput_filename)
        return metrics