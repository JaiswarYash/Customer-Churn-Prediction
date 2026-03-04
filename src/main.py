from src.data_ingestion import DataIngestion
from src.Feature_engineering import FeatureEngineering
from src.model_training import ModelTraining
from src.evaluate import ModelEval
from src.config import data_dir
import os

# ingestion
def run_ingestion():
    ingestion = DataIngestion()

    file_name = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_name.append(file)
    
    merged = ingestion.concatenate_files(file_name)
    cleaned_data = ingestion.clean_data(merged)
    ingestion.save_data(cleaned_data, 'clean_data.csv')

# feature Engineering
def run_feature_engineering():
    fe = FeatureEngineering()
    data = fe.load_clean_data('clean_data.csv')
    featured = fe.engineer_features(data)
    fe.save_data(featured,"featured_data.csv")

# model training
def run_modelTraining():
    mt = ModelTraining()
    data = mt.load_data("featured_data.csv")
    X_train, X_test, y_train, y_test = mt.train_n_test(data)
    return mt.pipeline(X_train, y_train), X_test, y_test

# run eval
def run_eval(pipeline, X_test, y_test):
    model_eval = ModelEval()
    model_eval.evaluation(pipeline, X_test, y_test)
    model_eval.save_model(pipeline, 'random_forest_model.pkl')

if __name__ == '__main__':
    run_ingestion()
    run_feature_engineering()
    pipeline, X_test, y_test = run_modelTraining()
    run_eval(pipeline, X_test, y_test)
    print("Pipeline completed successfully!")