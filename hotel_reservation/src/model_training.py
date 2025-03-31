# model training and experiment tracking 

import os 
import pandas as pd
import joblib  # model saving 
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint 
import mlflow 
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining():

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            x_train = train_df.drop(columns = ["booking_status"])
            y_train = train_df["booking_status"]

            x_test = test_df.drop(columns = ["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data split successfull for model training")

            return x_train, y_train, x_test, y_test
        
        except Exception as e :
            logger.error(f"Error while loading the data {e}")
            raise CustomException("Failed to load data due to exception", e)
        
    def train_lgbm(self, x_train, y_train):
        try:
            logger.info("Initializing the model")

            lgbm_model = lgb.LGBMClassifier(random_state = self.random_search_params["random_state"])
            logger.info("Hyper parameter tuning initialized")

            random_search = RandomizedSearchCV(
                estimator= lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]

            )
            logger.info("Hyper parameter tuning Started")
            random_search.fit(x_train, y_train)

            logger.info("Hyper-parameter tuning completed")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters are {best_params}")
            return best_lgbm_model


        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model due to exception", e)
        
    def evaluate_model(self, model, x_test, y_test):
        try:
            logger.info("Model evaluation")

            y_pred = model.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            precison = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Accuracy score: {accuracy}")
            logger.info(f"Precision score: {precison}")
            logger.info(f"Recall score: {recall}")
            logger.info(f"F1 score: {f1}")

            return {
                "accuracy" : accuracy,
                "precison" : precison,
                "recall" : recall,
                "f1": f1
            }

        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model due to exception", e)
        
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info("Saving the model")
            joblib.dump(model,self.model_output_path)
            logger.info(f"Model saved to path {self.model_output_path}")
        
        except Exception as e:
            logger.error(f"Error while saving the model {e}")
            raise CustomException("Failed to save model due to exception", e)
        
    def run(self):
        try:
            with mlflow.start_run():

                logger.info("Initiaizing model training pipeline")
                logger.info("Starting our Mflow experimentation")
                logger.info("saving the Training data into ML flow")
                # save dataset and model trained using artifact
                mlflow.log_artifact(self.train_path, artifact_path = "datasets")
                mlflow.log_artifact(self.test_path, artifact_path = "datasets")


                x_train, y_train, x_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(x_train=x_train, y_train=y_train)
                metrics = self.evaluate_model(x_test=x_test, y_test=y_test, model=best_lgbm_model)
                self.save_model(best_lgbm_model)
                logger.info("Logging model into MLFlow")
                mlflow.log_artifact(self.model_output_path)
                logger.info("Logging params and metrics")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)
                
                logger.info("Model Trainning pipeline completed")
        except Exception as e:
            logger.error(f"Error in model training pipeline {e}")
            raise CustomException("Failed to complete model training pipeline due to exception", e)
        

if __name__ == "__main__":
    trainer = ModelTraining(
        train_path = PROCESSED_TRAIN_DATA_PATH,
        test_path = PROCESSED_TEST_DATA_PATH,
        model_output_path = MODEL_OUTPUT_PATH
        )
    trainer.run()
        
            


            




