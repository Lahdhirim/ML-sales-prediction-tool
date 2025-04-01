import pandas as pd
from src.utils.training_config_loader import ClusteringProcessorConfig
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from src.utils.schema import DatasetSchema, MappingSchema

class ClusteringProcessor():
    """
    A class for processing and clustering customer transaction data.

    This class handles:
    - Loading and processing transaction data.
    - Encoding categorical features and scaling numerical features.
    - Computing customer statistics.
    - Training a KMeans clustering model to group customers.
    - Predicting cluster assignments for new data.
    """
    
    def __init__(self, clustering_processor_config: ClusteringProcessorConfig):
        self.product_mapping_path = clustering_processor_config.product_mapping_path
        self.lvl1_processed_data_path = clustering_processor_config.lvl1_processed_data_path
        self.categorical_features = clustering_processor_config.features["categorical_features"]
        self.numerical_features = clustering_processor_config.features["numerical_features"]
        self.max_clusters = clustering_processor_config.max_clusters
        self.default_cluster_size = clustering_processor_config.default_cluster_size
        self.kmeans = None
        self.label_encoders = {}
        self.scaler = None

    def load_lvl1_processed_data(self):
        try:
            processed_data = pd.read_csv(self.lvl1_processed_data_path)
            processed_data[DatasetSchema.YEAR_MONTH] = pd.to_datetime(processed_data[DatasetSchema.YEAR_MONTH], errors="coerce")
            processed_data[DatasetSchema.YEAR_MONTH] = processed_data[DatasetSchema.YEAR_MONTH].dt.to_period("M")
            return processed_data
        except FileNotFoundError:
            raise FileNotFoundError("Processed data not found at the specified path")
    
    def load_mapping(self) -> pd.DataFrame:
        try:
            product_mapping_file = pd.read_excel(self.product_mapping_path)
            return product_mapping_file
        except FileNotFoundError:
            raise FileNotFoundError("Product mapping not found at the specified path")
    
    # [MEDIUM]: Enhance code quality for the function calculate_statistics
    def calculate_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        # Average monthly transactions
        customer_transactions = X.groupby(DatasetSchema.CUSTOMER_ID).agg(avg_transactions_per_month=(DatasetSchema.YEAR_MONTH, "count")).reset_index()

        # Most bought product
        most_bought_product = X.groupby([DatasetSchema.CUSTOMER_ID, DatasetSchema.PRODUCT_ID]).size().reset_index(name="count")
        most_bought_product = most_bought_product.loc[most_bought_product.groupby(DatasetSchema.CUSTOMER_ID)["count"].idxmax(), [DatasetSchema.CUSTOMER_ID, DatasetSchema.PRODUCT_ID]]

        # Most bought group
        most_bought_group = X.groupby([DatasetSchema.CUSTOMER_ID, MappingSchema.GROUP]).size().reset_index(name="count")
        most_bought_group = most_bought_group.loc[most_bought_group.groupby(DatasetSchema.CUSTOMER_ID)["count"].idxmax(), [DatasetSchema.CUSTOMER_ID, MappingSchema.GROUP]]

        # Most country chosen
        most_chosen_country = X.groupby([DatasetSchema.CUSTOMER_ID, MappingSchema.COUNTRY]).size().reset_index(name="count")
        most_chosen_country = most_chosen_country.loc[most_chosen_country.groupby(DatasetSchema.CUSTOMER_ID)["count"].idxmax(), [DatasetSchema.CUSTOMER_ID, MappingSchema.COUNTRY]]

        # Most bought category
        most_bought_category = X.groupby([DatasetSchema.CUSTOMER_ID, MappingSchema.CATEGORY]).size().reset_index(name="count")
        most_bought_category = most_bought_category.loc[most_bought_category.groupby(DatasetSchema.CUSTOMER_ID)["count"].idxmax(), [DatasetSchema.CUSTOMER_ID, MappingSchema.CATEGORY]]

        # Concatenate all the statistics
        customer_transactions = customer_transactions.merge(most_bought_product, on=DatasetSchema.CUSTOMER_ID, how="inner") \
                                       .merge(most_bought_group, on=DatasetSchema.CUSTOMER_ID, how="inner") \
                                       .merge(most_chosen_country, on=DatasetSchema.CUSTOMER_ID, how="inner") \
                                       .merge(most_bought_category, on=DatasetSchema.CUSTOMER_ID, how="inner")
        return customer_transactions
    
    def process_data(self, X: pd.DataFrame) -> None:
        X_copy = X.copy()
        processed_data = self.load_lvl1_processed_data()
        filtered_data = processed_data.merge(X_copy[[DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH]], 
                                         on=[DatasetSchema.CUSTOMER_ID, DatasetSchema.YEAR_MONTH], 
                                         how='inner')
        mapping_file = self.load_mapping()
        transactions = pd.merge(filtered_data, mapping_file, on=DatasetSchema.PRODUCT_ID, how="inner")
        transactions = self.calculate_statistics(transactions)
        return transactions
        

    def fit(self, X: pd.DataFrame) -> None:
        X_copy = X.copy()

        # Encoding
        for feature in self.categorical_features:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_copy[feature] = encoder.fit_transform(X_copy[[feature]])
            self.label_encoders[feature] = encoder
        
        # Scaling
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(X_copy[self.categorical_features + self.numerical_features])

        # Clustering
        inertias = []
        for i in range(1, self.max_clusters):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_features)
            inertias.append(kmeans.inertia_)
        
        # Find the optimal number of clusters using the kneeLocator
        knee_locator = KneeLocator(range(1, self.max_clusters), inertias, curve="convex", direction="decreasing")
        optimal_k = knee_locator.elbow
        if optimal_k is None:
            print("No optimal k found, the default value will be used")
            optimal_k = self.default_cluster_size

        # Fit the KMeans model with the optimal number of clusters
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        self.kmeans.fit(scaled_features)
        return None


    def predict(self, X: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        assert all(feature in X_copy.columns for feature in self.categorical_features), f"Missing categorical features: {[feature for feature in self.categorical_features if feature not in X_copy.columns]}"
        assert all(feature in X_copy.columns for feature in self.numerical_features), f"Missing numerical features: {[feature for feature in self.numerical_features if feature not in X_copy.columns]}"

        # Encoding
        for feature in self.categorical_features:
            if feature not in self.label_encoders:
                raise ValueError(f"Label encoder for feature '{feature}' has not been fitted. Please call fit() before predict.")
            X_copy[feature] = self.label_encoders[feature].transform(X_copy[[feature]])
        
        # Scaling
        if self.scaler is None:
            raise ValueError("StandardScaler has not been fitted. Please call fit() before predict.")
        X_copy_scaled = self.scaler.transform(X_copy[self.categorical_features + self.numerical_features])

        # Predict clusters
        if self.kmeans is None:
            raise ValueError("KMeans model has not been fitted. Please call fit() before predict.")
        X_copy[DatasetSchema.CLUSTER_ID] = self.kmeans.predict(X_copy_scaled)

        # Merge cluster labels with input_df
        input_df = input_df.merge(X_copy[[DatasetSchema.CUSTOMER_ID, DatasetSchema.CLUSTER_ID]], on=DatasetSchema.CUSTOMER_ID, how='inner')
        return input_df
