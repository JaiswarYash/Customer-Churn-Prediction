import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from src.config import model_path
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelEval:
    def __init__(self):
        self.save_path = model_path
    
    # evaluate
    def evaluation(self, pipeline, X_train, y_train, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        logger.info(f"CV F1: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    # save model
    def save_model(self, pipeline, model_name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        save_path = os.path.join(self.save_path, model_name)
        joblib.dump(pipeline, save_path)
        logger.info(f"Model saved to {save_path}")
    
    # load model
    def load_model(self, model_name):
        model_file = os.path.join(self.save_path, model_name)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model not found at {model_file}")
        return joblib.load(model_file)

