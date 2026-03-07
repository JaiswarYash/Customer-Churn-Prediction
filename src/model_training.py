import os
import json
import pandas as pd
from src.config import processed_data_dir
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)
class ModelTraining:

    def __init__(self):
        self.processed_dir = processed_data_dir

    def load_data(self, file_name):
        file_path = os.path.join(self.processed_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError (f"File {file_name} not found in {self.processed_dir}")
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise Exception(f"Error loading file {file_name}: {str(e)}")



    def train_n_test(self, data,test_size = 0.2, random_state=42):
        X = data.drop(columns=['Churn'])
        y = data['Churn']
        return train_test_split(X,y, test_size=test_size, random_state=random_state)
    
    def _default_params(self):
        return RandomForestClassifier(
                    n_estimators=200,
                    max_features='sqrt',
                    max_depth=None,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42
                )
    

    def pipeline(self, X_train, y_train, model=None):
        mlflow.set_experiment("churn-prediction")

        if model is None:
            model = self._default_params()
        with mlflow.start_run():
            
            mlflow.log_param('n_estimators',200)
            mlflow.log_param("max_depth", "None")
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("train_size", len(X_train))
            pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
            trained = pipeline.fit(X_train, y_train)

            with open('models/feature_names.json', 'w') as f:
                json.dump(list(X_train.columns), f)
                logger.info(f"Feature names saved: {list(X_train.columns)}")

            mlflow.sklearn.log_model(trained,'churn model')
        return trained
    