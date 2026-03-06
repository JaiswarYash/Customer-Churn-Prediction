# data ingestion module
import os
import pandas as pd
import logging
from src.config import data_dir, processed_data_dir

logger = logging.getLogger(__name__)

# constat
Binary_columns = ['gender','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
                  'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']

No_internet_service = ['MultipleLines','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

Columns_to_drop = ['TotalCharges', 'customerID']

class DataIngestion:
    def __init__(self):
        self.data_dir = data_dir # directory where data files are stored
        self.directory = processed_data_dir # directory where cleaned data will be saved

    # file loading
    def load_data(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_name} not found in {self.data_dir}")
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {file_name} — {len(data)} rows")
            return data
        except Exception as e:
            raise Exception(f"Error loading {file_name}: {str(e)}")
    # concatenate files into a single dataframe
    def concatenate_files(self, file_names):
        
        data_frames = []
        for file_name in file_names:
            data = self.load_data(file_name)
            # converting totalcharges to numeric, coercing errors to NaN
            if 'TotalCharges' in data.columns:
                data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            data_frames.append(data)
        
        combined = pd.concat(data_frames, ignore_index=True)
        return combined
    
    # cleaning data
    def clean_data(self, data):

        # drop duplicates
        data = data.drop_duplicates()
        # no missing values after dropping TotalCharges
        # dropna() not required for this dataset

        data = data.drop(columns=Columns_to_drop, errors='ignore')
        
        # mapping categorical columns
        for col in No_internet_service:
            data[col] = data[col].replace({'No internet service': 'No', 'No phone service': 'No'})

        # 1. mapping churn column to binary values
        for col in Binary_columns:
            if col == 'gender':
                data[col] = data[col].apply(lambda x: 1 if x == 'Male' else 0)
            else:
                data[col] = data[col].map({'Yes': 1, 'No': 0})
        return data
    
    # save cleaned data to a new file
    def save_data(self, data, file_name):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        file_path = os.path.join(self.directory, file_name)
        try:
            data.to_csv(file_path, index=False)
            logger.info(f"Cleaned data saved to {file_path}")
        except Exception as e:
            raise Exception(f"Error saving file {file_name}: {str(e)}")