import os
import pandas as pd
import logging
import mlflow
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from src.config import processed_data_dir, report_path


logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self):
        # Use env var or fall back to default path
        processed = os.getenv('PROCESSED_DATA_PATH', 'data/processed')
        report = os.getenv('REPORT_PATH', 'monitoring/reports')
        
        self.reference_data = os.path.join(processed, "featured_data.csv")
        self.report_dir = report
        os.makedirs(self.report_dir, exist_ok=True)
    
    def load_reference_data(self) -> pd.DataFrame:
        if not os.path.exists(self.reference_data):
            raise FileNotFoundError(f"Reference data not found at {self.reference_data}")
        return pd.read_csv(self.reference_data)
    
    def monitor_n_generate_report(self,current_data: pd.DataFrame, experiment_name="Model Monitoring") -> dict:

        reference_data = self.load_reference_data()

        with mlflow.start_run():
            # create report
            report = Report(
                metrics=[
                    DataDriftPreset(threshold=0.3),
                    ClassificationPreset()
                ])
            report.run(
                reference_data = reference_data,
                current_data = current_data
            )

            # save HTML report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.report_dir, f"drift_report_{timestamp}.html")
            report.save_html(report_file)
            logger.info(f"Report saved to {report_path}")

            # Extract data
            results = report.as_dict()
            drift_detected = results["metrics"]["0"]["drift_detected"]
            for metric in results["metrics"]:
                if "drift_score" in metric:
                    mlflow.log_metric(metric["column_name"], metric["drift_score"])
            summary = {
                "timestamp": timestamp,
                "drift_detected": drift_detected,
                "n_features_drifted": results["metrics"][0]["result"]["number_of_drifted_columns"],
                "n_features_total": results["metrics"][0]["result"]["number_of_columns"],
                "report_path": report_file,
            }
        logger.info(f"Drift detected: {drift_detected}")
        logger.info(f"Features drifted: {summary['n_features_drifted']}/{summary['n_features_total']}")

        return summary