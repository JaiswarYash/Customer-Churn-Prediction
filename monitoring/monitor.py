import os
import json
import pandas as pd
import logging
import mlflow
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
    
    def load_reference_data(self) -> pd.DataFrame:
        if not os.path.exists(self.referance_data):
            raise FileNotFoundError(f"Reference data not found at {self.reference_path}")
        return pd.read_csv(self.referance_data)
    
    def monitor_n_generate_report(self,current_data: pd.DataFrame, experiment_name="Model Monitoring") -> dict:

        reference_data = self.load_reference_data

        with mlflow.start_run():

            # prediction_columns
            column_mapping = ColumnMapping(
                target="Churn",
                prediction=None
            )
            
            # create report
            report = Report(
                metrics=[
                    DataDriftPreset(threshold=0.3),
                    ClassificationPreset
                ])
            report.run(
                reference_data = reference_data,
                current_data = current_data,
                predict = column_mapping
            )

            # save HTML report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.report_dir, f"drift_report_{timestamp}.html")
            report.save_html(report_path)
            logger.info(f"Report saved to {report_path}")

            # Extract data
            results = report.as_dict()
            for metric in results["metrics"]:
                if "drift_score" in metric:
                    mlflow.log_metric(metric["column_name"], metric["drift_score"])
            summary = {
                "timestamp": timestamp,
                "drift_detected": drift_detected,
                "report_path": report_path,
                "n_features_drifted": results["metrics"][0]["result"]["number_of_drifted_columns"],
                "n_features_total": results["metrics"][0]["result"]["number_of_columns"],
            }
        logger.info(f"Drift detected: {drift_detected}")
        logger.info(f"Features drifted: {summary['n_features_drifted']}/{summary['n_features_total']}")

        return summary