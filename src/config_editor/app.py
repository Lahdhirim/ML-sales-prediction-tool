from flask import Flask, request
from flask_cors import CORS
import os
import subprocess
import signal
from threading import Timer
import json

# Add path to project root directory for importing modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import pipelines and schema
from src.utils.processing_config_loader import preprocessing_config_loader
from src.processing_pipeline import processingPipeline
from src.utils.training_config_loader import training_config_loader
from src.backtesting_pipeline import backtestingPipeline
from src.utils.schema import PipelinesDictSchema

app = Flask(__name__)
CORS(app)

PROCESSING_CONFIG_PATH = "config/processing_config.json"
TRAINING_CONFIG_PATH = "config/training_config.json"

angular_process = None

def run_angular_ui():
    global angular_process
    angular_process = subprocess.Popen(
        ["ng", "serve", "--open"], cwd="src/config_editor/angular_ui", shell=True)

@app.route("/load_processing_config", methods=["GET"])
def read_processing_config():
    try:
        processing_config = preprocessing_config_loader(config_path = PROCESSING_CONFIG_PATH)
        processing_config = processing_config.dict() # To let processing_config be JSON serializable
        return {"message": "Processing Config loaded successfully", "config": processing_config}, 200
    except Exception as e:
        app.logger.error(f"Error loading processing config: {e}")
        return {"message": str(e)}, 500

@app.route("/load_training_config", methods=["GET"])
def read_training_config():
    try:
        training_config = training_config_loader(config_path = TRAINING_CONFIG_PATH)
        training_config = training_config.dict() # To let training_config be JSON serializable
        for model_name, model_hyperparameters in training_config[PipelinesDictSchema.MODELS_PARAMS][PipelinesDictSchema.ML_MODELS].items():
            training_config[PipelinesDictSchema.MODELS_PARAMS][PipelinesDictSchema.ML_MODELS][model_name] = \
            {hyper_param: value for hyper_param, value in model_hyperparameters.items() if value is not None} # To avoid assigning null values to non relevant hype-parameter during update_training_config
        return {"message": "Training Config loaded successfully", "config": training_config}, 200
    except Exception as e:
        app.logger.error(f"Error loading training config: {e}")
        return {"message": str(e)}, 500

@app.route("/update_processing_config", methods=["POST"])
def update_processing_config():
    try:
        new_config = request.json
        with open(PROCESSING_CONFIG_PATH, "w") as file:
            json.dump(new_config, file, indent=4, ensure_ascii=False)
        return {"message": "Processing Config updated successfully"}, 200
    except Exception as e:
        return {"message": str(e)}, 500

@app.route("/update_training_config", methods=["POST"])
def update_training_config():
    try:
        new_config = request.json
        with open(TRAINING_CONFIG_PATH, "w") as file:
            json.dump(new_config, file, indent=4, ensure_ascii=False)
        return {"message": "Training Config updated successfully"}, 200
    except Exception as e:
        return {"message": str(e)}, 500

@app.route("/run_backtesting", methods=["POST"])
def run_backtesting():
    try:
        processing_config = preprocessing_config_loader(config_path = PROCESSING_CONFIG_PATH)
        processing_pipeline = processingPipeline(processing_config = processing_config)
        processing_pipeline.run()

        training_config = training_config_loader(config_path = TRAINING_CONFIG_PATH)
        backtesting_pipeline = backtestingPipeline(training_config = training_config)
        backtesting_pipeline.run()
        return {"message": "Backtesting completed"}, 200
    except Exception as e:
        return {"message": str(e)}, 500

@app.route("/shutdown", methods=["POST"])
def shutdown():
    try:
        global angular_process
        if angular_process:
            if os.name == 'posix':  # Linux/macOS
                subprocess.run(f"pkill -f 'ng serve'", shell=True)
            elif os.name == 'nt':  # Windows
                subprocess.run(f"taskkill /F /IM node.exe", shell=True)

            angular_process.wait()
            angular_process = None

        def shutdown_server():
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)
        Timer(0.5, shutdown_server).start()
        return {"message": "Server shutting down..."}, 200
    except Exception as e:
        return {"message": str(e)}, 500

if __name__ == "__main__":
    run_angular_ui()
    app.run(port=5000, host="127.0.0.1")