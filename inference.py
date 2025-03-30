from src.inference_pipeline import inferencePipeline
from src.utils.inference_config_loader import inference_config_loader

if __name__ == '__main__':
    inference_config = inference_config_loader("config/inference_config.json")
    inference_pipeline = inferencePipeline(inference_config = inference_config)
    inference_pipeline.run()