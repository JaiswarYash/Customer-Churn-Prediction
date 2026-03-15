from src import main

def test_ingestion():
    main.run_ingestion()
    print("Ingestion done!")

def test_featuresEngineer():
    main.run_feature_engineering()
    print("Feature Engineering is done sucessfully!")

def test_modelTraining():
    pipeline, X_train, X_test, y_train, y_test = main.run_modelTraining()
    print("model training is doing fine!")
    return pipeline, X_train, X_test, y_train, y_test

def test_evaluation():
    pipeline, X_train, X_test, y_train, y_test = test_modelTraining()  # ← retrains
    main.run_eval(pipeline, X_train, X_test, y_train, y_test)
    print("model is doing well.")
    
def test_model_saved():
    import os
    from src.config import model_path
    assert os.path.exists(model_path), "Model was not saved!"
    print(f"✅ Model saved at {model_path}")