# feature engineering module
import os
import pandas as pd
import logging
from src.config import processed_data_dir

logger = logging.getLogger(__name__)

WEAK_FEATURES = ['PaymentMethod_Credit card', 'PhoneService']

class FeatureEngineering:

    def __init__(self):
        self.directory = processed_data_dir # directory where cleaned data is stored

    # load clean data
    def load_clean_data(self, file_name):
        file_path = os.path.join(self.directory, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_name} not found in {self.directory}")
       
        data = pd.read_csv(file_path)
        logger.info(f"Loaded clean data: {data.shape}")
        return data
    
    # feature engineering
    def engineer_features(self, data):
        
        df_encoded = pd.get_dummies(
            data,
            columns=['InternetService', 'PaymentMethod', 'Contract'],
            drop_first=True,
            dtype=int
        )

        existing = [f for f in WEAK_FEATURES if f in df_encoded.columns]
        df_encoded.drop(columns=existing, inplace=True)

        return df_encoded
    
    # save
    def save_data(self, data, file_name):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        file_path = os.path.join(self.directory, file_name)
        data.to_csv(file_path, index=False)
        logger.info(f"Featured data saved to: {file_path}")
        
