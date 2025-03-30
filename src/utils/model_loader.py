import pickle

def load_models(saved_models_path: str) -> dict:
    with open(saved_models_path, "rb") as file:
        pipelines_dict = pickle.load(file)
    print(f"Models loaded from {saved_models_path}")
    return pipelines_dict