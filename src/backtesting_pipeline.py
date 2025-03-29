from src.utils.splitter import TimeSeriesSplitter
from src.modeling.features_selector import FeatureSelector
from src.modeling.clustering_processor import ClusteringProcessor
import pandas as pd
from colorama import Fore, Style
from src.utils.schema import DatasetSchema
from sklearn.preprocessing import StandardScaler
from src.modeling.ml_models import MLModels
from src.modeling.mlp import MLPModel
from src.evaluators.ml_evaluator import MLEvaluator
from src.utils.training_config_loader import TrainingConfig

class backtestingPipeline:
    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config
    
    def load_processed_data(self):
        try :
            processed_data = pd.read_csv(self.training_config.train_data_path)
        except FileNotFoundError:
            raise FileNotFoundError("Processed data not found at the specified path")
        return processed_data


    def run(self):
        print(Fore.YELLOW + "Running backtesting pipeline..." + Style.RESET_ALL)
        df = self.load_processed_data()

        splitter = TimeSeriesSplitter(splitter_config = self.training_config.splitter)
        results = pd.DataFrame()
        for split_index, train, val, test in splitter.split(df):
            print("-" * 40)
            print(f"{Fore.YELLOW}Split {split_index}{Style.RESET_ALL}")
            print("Train:", train[DatasetSchema.YEAR_MONTH].min(), train[DatasetSchema.YEAR_MONTH].max())
            print("Val:", val[DatasetSchema.YEAR_MONTH].min(), val[DatasetSchema.YEAR_MONTH].max())
            print("Test:", test[DatasetSchema.YEAR_MONTH].min(), test[DatasetSchema.YEAR_MONTH].max())

            #[MEDIUM]: Build a pipeline that integrates all steps before training
            # Clustering
            clustering_processor = ClusteringProcessor(clustering_processor_config = self.training_config.clustering_processor)
            train_processed = clustering_processor.process_data(X=train)
            val_processed = clustering_processor.process_data(X=val)
            test_processed = clustering_processor.process_data(X=test)
            clustering_processor.fit(X=train_processed)
            train = clustering_processor.predict(X=train_processed, input_df=train)
            val = clustering_processor.predict(X=val_processed, input_df=val)
            test = clustering_processor.predict(X=test_processed, input_df=test)

            # Feature selection
            features_selector = FeatureSelector(features_selector_config = self.training_config.features_selector)
            X_train, y_train = features_selector.transform(train)
            X_val, y_val = features_selector.transform(val)
            X_test, y_test = features_selector.transform(test)

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # Train models
            predictions = {}

            ml_models = MLModels(config = self.training_config.models_params.MLModels)
            ml_models.train(X_train_scaled, y_train, X_val_scaled, y_val)
            predictions = ml_models.predict(X_test_scaled, predictions)

            mlp_model = MLPModel(config = self.training_config.models_params.MLP)
            mlp_model.train(X_train_scaled, y_train)
            predictions = mlp_model.predict(X_test_scaled, predictions)

            # Concatenate predictions
            predictions_df = pd.DataFrame(predictions)
            input_df = pd.DataFrame({DatasetSchema.SPLIT_INDEX: split_index, 
                                     DatasetSchema.CUSTOMER_ID: test[DatasetSchema.CUSTOMER_ID],
                                     DatasetSchema.YEAR_MONTH: test[DatasetSchema.YEAR_MONTH],
                                     DatasetSchema.NB_TRANSACTIONS: test[DatasetSchema.NB_TRANSACTIONS],
                                     self.training_config.features_selector.target_column: y_test})
            results = results._append(pd.concat([input_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1), ignore_index=True)
        
        # Save raw predictions
        results.to_csv(self.training_config.raw_predictions_path, index=False)

        # Evaluate the predictions
        ml_evaluator = MLEvaluator(config = self.training_config, models = predictions.keys())
        ml_evaluator.evaluate(results)
        print(Fore.GREEN + "Backtesting pipeline completed successfully" + Style.RESET_ALL)