import argparse
import subprocess
from src.processing_pipeline import processingPipeline
from src.utils.processing_config_loader import preprocessing_config_loader
from src.backtesting_pipeline import backtestingPipeline
from src.utils.training_config_loader import training_config_loader
from src.inference_pipeline import inferencePipeline
from src.utils.inference_config_loader import inference_config_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sales Prediction Tool")
    parser.add_argument("mode", choices=["train_with_ui", "train", "inference"],
                        default="train", nargs="?", help="Choose mode: train_with_ui, train, or inference")
    args = parser.parse_args()

    if args.mode == "train_with_ui":
        subprocess.run(["python", "src/config_editor/app.py"])

    elif args.mode == "train":
        processing_config = preprocessing_config_loader(config_path = "config/processing_config.json")
        processing_pipeline = processingPipeline(processing_config = processing_config)
        processing_pipeline.run()

        training_config = training_config_loader(config_path = "config/training_config.json")
        backtesting_pipeline = backtestingPipeline(training_config = training_config)
        backtesting_pipeline.run()
    
    elif args.mode == "inference":
        inference_config = inference_config_loader(config_path = "config/inference_config.json")
        inference_pipeline = inferencePipeline(inference_config = inference_config)
        inference_pipeline.run()
    
    else:
        print("Invalid mode. Please choose 'train_with_ui', 'train', or 'inference'.")
