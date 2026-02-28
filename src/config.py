import dotenv
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

data_dir = os.getenv('raw_DATA_PATH')
processed_data_dir = os.getenv('processed_DATA_PATH')
model_path = os.getenv('model_path')