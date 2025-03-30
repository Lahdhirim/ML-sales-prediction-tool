from src.utils.inference_config_loader import InferenceConfig
import pandas as pd
from colorama import Fore, Style
from src.utils.schema import DatasetSchema
from src.utils.model_loader import load_models

class inferencePipeline:
    def __init__(self, inference_config: InferenceConfig):
        self.inference_config = inference_config
    
    def load_processed_data(self):
        try:
            processed_data = pd.read_csv(self.inference_config.processed_data_path)
        except FileNotFoundError:
            raise FileNotFoundError("Processed data not found at the specified path")
        return processed_data
    
    def filter_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data[DatasetSchema.YEAR_MONTH] = pd.to_datetime(data[DatasetSchema.YEAR_MONTH]).dt.to_period("M")
        return data[data[DatasetSchema.YEAR_MONTH] > data[DatasetSchema.YEAR_MONTH].max() - self.inference_config.nb_months_to_predict]
    
    def load_models(self) -> dict:
        pipelines_dict = load_models(saved_models_path = self.inference_config.saved_models_path)
        if len(pipelines_dict) < self.inference_config.n_models:
            print(f"{Fore.RED}Not enough trained models found. Please reduce the number of trained models to at most {len(pipelines_dict)}")
            return None
        else:
            last_n_models = dict(sorted(pipelines_dict.items())[-self.inference_config.n_models:])
            return last_n_models

    def run(self):
        print(Fore.YELLOW + "Running inference pipeline..." + Style.RESET_ALL)
        df = self.load_processed_data()
        df_filtered = self.filter_prediction_data(df)
        print("Shape of the filtered data: ", df_filtered.shape)

        pipelines_dict = self.load_models()
        results = pd.DataFrame()
        for pipeline_index, steps in pipelines_dict.items():

            # Clustering
            clustering_processor = steps["clustering_processor"]
            test_processed = clustering_processor.process_data(X=df_filtered)
            test = clustering_processor.predict(X=test_processed, input_df=df_filtered)

            # Feature selection
            features_selector = steps["features_selector"]
            X_test, _ = features_selector.transform(test)

            # Scaling
            scaling = steps["scaler"]
            X_test_scaled = scaling.transform(X_test)

            # Prediction
            predictions = {}
            if "ml_models" in steps:
                ml_models = steps["ml_models"]
                predictions = ml_models.predict(X_test_scaled, predictions)
            
            if "mlp_model" in steps:
                mlp_model = steps["mlp_model"]
                predictions = mlp_model.predict(X_test_scaled, predictions)
            
            # Concatenate predictions
            predictions_df = pd.DataFrame(predictions)
            input_df = pd.DataFrame({DatasetSchema.SPLIT_INDEX: pipeline_index, 
                                     DatasetSchema.CUSTOMER_ID: test[DatasetSchema.CUSTOMER_ID],
                                     DatasetSchema.YEAR_MONTH: test[DatasetSchema.YEAR_MONTH],
                                     DatasetSchema.NB_TRANSACTIONS: test[DatasetSchema.NB_TRANSACTIONS]})
            results = results._append(pd.concat([input_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1), ignore_index=True)
        results.to_csv("data/preds.csv", index=False)
        print(Fore.GREEN + "Inference pipeline completed successfully" + Style.RESET_ALL)