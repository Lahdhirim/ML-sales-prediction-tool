from src.processing_pipeline import processingPipeline
from src.utils.processing_config_loader import PreprocssingConfig
from src.backtesting_pipeline import backtestingPipeline
from src.utils.training_config_loader import TrainingConfig

if __name__ == '__main__':
    processing_config = PreprocssingConfig("config/processing_config.json")
    processing_pipeline = processingPipeline(processing_config = processing_config)
    processing_pipeline.run()

    training_config = TrainingConfig("config/training_config.json")
    backtesting_pipeline = backtestingPipeline(training_config = training_config)
    backtesting_pipeline.run()