import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from src.config import model_path
import joblib

class ModelEval:

    def __init__(self):
        self.save_path = model_path
    
    # evaluate
    def evaluation(self,pipeline,X_test, y_test):
        y_pred = pipeline.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        scores = cross_val_score(pipeline,X_test, y_test, cv=5, scoring='f1')
        print(f"fi accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

    # save model
    def save_model(self, pipeline, model_name='model.pkl'):
    
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        save_path = os.path.join(self.save_path, model_name)
        joblib.dump(pipeline, save_path)
        print(f"Model saved to {save_path}")
    
    # load model
    def load_model(self):
        pipeline = joblib.load(self.save_path)
        return pipeline

