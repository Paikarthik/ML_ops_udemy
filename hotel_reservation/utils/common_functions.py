import os
import pandas
import yaml
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found in the given directory {file_path}"
            )
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the YAML file")
            return config

    except Exception as e:
        logger.error("Error while reading the YAML file")
        raise CustomException("Failed to read the YAML file", e)
