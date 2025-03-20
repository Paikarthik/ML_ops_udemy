import os 
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.config import *
from utils.common_function import read_yaml, load_data
from skleearn.emsemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from src.logger import get_logger

logger = get_logger(logger_name = __name__)

class DataPreprocessing:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
        
    def preprocess_data(self, df):
        try:
            logger.info("Start data preprocessing")
            logger.info("Dropping Columns")
            df.drop(columns = ["Booking_ID", "Unnamed: 0"], inplace = True)
            df.dropduplicate(inplace = True)

            

