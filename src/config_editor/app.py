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

# Import pipelines
from src.utils.processing_config_loader import preprocessing_config_loader
from src.processing_pipeline import processingPipeline
from src.utils.training_config_loader import training_config_loader
from src.backtesting_pipeline import backtestingPipeline

app = Flask(__name__)
CORS(app)

PROCESSING_CONFIG_PATH = "config/processing_config.json"
TRAINING_CONFIG_PATH = "config/training_config.json"

angular_process = None

def run_angular_ui():
    global angular_process
    angular_process = subprocess.Popen(
        ["ng", "serve", "--open"], cwd="src/config_editor/angular_ui", shell=True)

@app.route("/load_config", methods=["GET"])
def read_config():
    try:
        processing_config = preprocessing_config_loader("config/processing_config.json")
        training_config = training_config_loader(config_path = "config/training_config.json")
        return {"message": "Config loaded successfully"}, 200
    except Exception as e:
        return {"message": str(e)}, 500

@app.route("/update_processing_config", methods=["POST"])
def update_processing_config():
    try:
        new_config = request.json
        with open(PROCESSING_CONFIG_PATH, "w") as file:
            json.dump(new_config, file, indent=4)
        return {"message": "Processing Config updated successfully"}, 200
    except Exception as e:
        return {"message": str(e)}, 500

@app.route("/update_training_config", methods=["POST"])
def update_training_config():
    try:
        new_config = request.json
        with open(TRAINING_CONFIG_PATH, "w") as file:
            json.dump(new_config, file, indent=4)
        return {"message": "Training Config updated successfully"}, 200
    except Exception as e:
        return {"message": str(e)}, 500

@app.route("/run_backtesting", methods=["POST"])
def run_backtesting():
    try:
        processing_config = preprocessing_config_loader("config/processing_config.json")
        processing_pipeline = processingPipeline(processing_config = processing_config)
        processing_pipeline.run()

        training_config = training_config_loader(config_path = "config/training_config.json")
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