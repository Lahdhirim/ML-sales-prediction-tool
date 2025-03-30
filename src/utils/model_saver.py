import pickle

def save_models(pipelines_dict: dict, output_file_name: str) -> None:
    with open(output_file_name, "wb") as file:
        pickle.dump(pipelines_dict, file)
    print(f"Trained models saved to {output_file_name}")