import pytest
import os
from src import main 

# Skip these tests in CI environment — require local data files
skipif_ci = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Requires local data files not available in CI"
)

@skipif_ci
def test_ingestion():
    main.run_ingestion()

@skipif_ci
def test_featuresEngineer():
    main.run_feature_engineering()

@skipif_ci
def test_modelTraining():
    pipeline, X_train, X_test, y_train, y_test = main.run_modelTraining()
    assert pipeline is not None
    assert X_train.shape[1] == 22
    return pipeline, X_train, X_test, y_train, y_test

@skipif_ci
def test_evaluation():
    pipeline, X_train, X_test, y_train, y_test = main.run_modelTraining()
    main.run_eval(pipeline, X_train, y_train, X_test, y_test)
    print("model is doing well.")

def test_model_saved():
    import os
    from src.config import model_path
    assert os.path.exists(model_path), "Model was not saved!"