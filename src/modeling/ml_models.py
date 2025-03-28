from sklearn.linear_model import LinearRegression, ElasticNet
from colorama import Fore, Style

class MLModels:
    def __init__(self, config):
        self.models = {}
        if config.get("LinearRegression", {}).get("enabled", False):
            params = {k: v for k, v in config["LinearRegression"].items() if k != "enabled"}
            self.models["LinearRegression"] = LinearRegression(**params)

        if config.get("ElasticNet", {}).get("enabled", False):
            params = {k: v for k, v in config["ElasticNet"].items() if k != "enabled"}
            self.models["ElasticNet"] = ElasticNet(**params)

    def train(self, X_train, y_train, X_val, y_val):
        for name, model in self.models.items():
            print(f"{Fore.BLUE}Training {name}{Style.RESET_ALL}")
            model.fit(X_train, y_train)
        return self.models
    
    def predict(self, X_test, y_test, predictions):
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
        return predictions