import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

data_dir = os.getenv('raw_DATA_PATH')
processed_data_dir = os.getenv('processed_DATA_PATH')
model_path = os.getenv('MODEL_PATH')
report_path = os.getenv('report_dir')