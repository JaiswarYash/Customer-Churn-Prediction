import os
import pandas as pd
from config import processed_data_dir, model_path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

class ModelTraining:

    def __init__(self):
        self.processed_dir = processed_data_dir
        self.model_path = model_path
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
        

    def pipeline(self, X_train, y_train):

        pipeline = Pipeline((
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators = 200,max_features ='sqrt',max_depth =None,
                                            min_samples_split =10, min_samples_leaf = 5,class_weight='balanced',
                                            bootstrap = True, random_state = 42, criterion = 'gini', 
                                            ))
            ))
        return pipeline.fit(X_train, y_train)
        
