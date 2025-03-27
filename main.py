from src.processing_pipeline import processingPipeline
from src.utils.processing_config_loader import PreprocssingConfig

if __name__ == '__main__':
    processing_config = PreprocssingConfig("config/processing_config.json")
    processing_pipeline = processingPipeline(processing_config = processing_config)
    processing_pipeline.run()