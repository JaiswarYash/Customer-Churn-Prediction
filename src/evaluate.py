from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from config import model_path
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
    def save_model(self,pipeline):
        joblib.dump(pipeline, self.save_path)
        print(f"Model saved to {self.save_path}")
    
    # load model
    def load_model(self):
        pipeline = joblib.load(self.save_path)
        return pipeline
