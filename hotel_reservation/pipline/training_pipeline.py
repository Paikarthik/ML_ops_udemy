from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import * 

if __name__ == "__main__":
    # Data ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    # data processing 
    data_processor = DataPreprocessing(
        train_path=TRAIN_FILE_PATH, 
        test_path=TEST_FILE_PATH, 
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
        )
    data_processor.process()

    # model training 
    trainer = ModelTraining(
        train_path = PROCESSED_TRAIN_DATA_PATH,
        test_path = PROCESSED_TEST_DATA_PATH,
        model_output_path = MODEL_OUTPUT_PATH
        )
    trainer.run()