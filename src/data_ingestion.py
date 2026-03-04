# data ingestion module
import os
import pandas as pd
from src.config import data_dir, processed_data_dir

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
            return data
        except Exception as e:
            raise Exception(f"Error loading file {file_name}: {str(e)}")
    
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

        # droping irrelevant columns
        list_of_columns_to_drop = ['customerID', 'TotalCharges','MonthlyCharges'] # dropping totalcharges as it has many missing values and is not crucial for analysis
        data = data.drop(columns=list_of_columns_to_drop, errors='ignore')
        
        # mapping categorical columns
        cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

        for col in cols:
            data[col] = data[col].replace({'No internet service': 'No', 'No phone service': 'No'})

        # 1. mapping churn column to binary values
        cols = ['Churn','Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

        for col in cols:
            data[col] = data[col].map({'Yes': 1, 'No': 0})
        
        # mapping churn column to binary values
        # if data['Churn'].dtype == 'object':
        #     data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        # else:
        #     data['Churn'] = data['Churn'].astype(int)

        # 2. mapping gender column to binary values
        data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)

        return data
    
    # save cleaned data to a new file
    def save_data(self, data, file_name):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        file_path = os.path.join(self.directory, file_name)
        try:
            data.to_csv(file_path, index=False)
            print(f"Cleaned data saved to {file_path}")
        except Exception as e:
            raise Exception(f"Error saving file {file_name}: {str(e)}")