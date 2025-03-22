import os 
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from src.config import *
from utils.common_functions import read_yaml, load_data
from sklearn.emsemble import RandomForestClassifier
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

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]
            
            logger.info("Applying label encoder")

            label_encoder = LabelEncoder()

            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logger.info("Label mappings are:")

            for col, mapping in mappings.items():
                logger.info(f"{col} :{mapping}")

            logger.info("Skewness handling")

            skew_threshold = self.config["data_processing"]["skewness_threshold"]

            skewness = df[num_cols].apply(lambda x: x.skew())

            for col in skewness[skewness > skew_threshold].index:
                df[col] = np.log1p(df[col])

            return df
        
        except Exception as e :
            logger.error(f"Error during preprocessing step {e}")
            raise CustomException("Error while pre-processing data", e)
        
    
    def balance_data(self, df):
        try: 
            logger.info("Handling imbalance data")
            x= df.drop(columns="booking_status")
            y = df["booking_status"]
            smote = SMOTE(random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x,y)
            balanaced_df  = pd.DataFrame(x_resampled, columns=x.columns)
            balanaced_df["booking_status"] = y_resampled

            logger.info("Data balanced successfully")
            return df
        
        except Exception as e :
            logger.error(f"Error during balancing data {e}")
            raise CustomException("Error while data balancing", e)

    def select_features(self, df):
        try:
            logger.info("Started feature selection step")

            x= df.drop(columns="booking_status")
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(x,y)
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature' : x.columns,
                'importance': feature_importance
            })
            top_features_df = feature_importance_df.sort_values(by = "importance", ascending = False)

            
            

        except Exception as e :
            pass



                

