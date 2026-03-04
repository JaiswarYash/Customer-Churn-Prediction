# feature engineering module
import os
import pandas as pd
from src.config import processed_data_dir

class FeatureEngineering:

    def __init__(self):
        self.directory = processed_data_dir # directory where cleaned data is stored

    # load clean data
    def load_clean_data(self, file_name):
        file_path = os.path.join(self.directory, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_name} not found in {self.directory}")
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise Exception(f"Error loading file {file_name}: {e}")
    
    # feature engineering
    def engineer_features(self, data):
        
        # one-hot encoding
        df_encoded = pd.get_dummies(data, columns=['InternetService', 'PaymentMethod'], drop_first=True,dtype=int)
        
        # ordinal encoding
        df_encoded['Contract'] = df_encoded['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
        
        # drop weak features
        df_encoded.drop(columns=['gender','PaymentMethod_Mailed check','OnlineBackup', 
                                'DeviceProtection','PhoneService','MultipleLines',
                                'StreamingMovies','PaymentMethod_Credit card', 'StreamingTV'], inplace=True)
        return df_encoded
    
    # save
    def save_data(self, data, file_name):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        file_path = os.path.join(self.directory, file_name)
        try:
            data.to_csv(file_path, index=False)
            print(f"Featured data saved to: {file_path}")
        except Exception as e:
            raise Exception(f"Error saving file {file_name}: {str(e)}")
