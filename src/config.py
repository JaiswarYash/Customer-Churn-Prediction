import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

data_dir = os.getenv('RAW_DATA_PATH', 'data/raw')
processed_data_dir = os.getenv('PROCESSED_DATA_PATH', 'data/processed')
model_path = os.getenv('MODEL_PATH', 'models')
report_path = os.getenv('REPORT_PATH', 'monitoring/reports')