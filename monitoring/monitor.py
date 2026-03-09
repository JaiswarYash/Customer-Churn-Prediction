import os
import json
import pandas as pd
import logging
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from src.config import processed_data_dir, report_path
from evidently import ColumnMapping
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self):
        self.referance_data = processed_data_dir
        self.report_dir = report_path
        os.makedirs(self.report_dir, exist_ok=True)
    
    def load_reference_data(self):
        if not os.path.exists(self.referance_data):
            raise FileNotFoundError(f"Reference data not found at {self.reference_path}")
        return pd.read_csv(self.reference_path)
    
    def generate_report(self):
        pass


        