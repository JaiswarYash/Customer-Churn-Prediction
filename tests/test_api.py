from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_returns_correct_fields():
    payload = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "MultipleLines": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "PaperlessBilling": 1,
    "MonthlyCharges": 65.5,
    "InternetService_Fiber_optic": 1,
    "InternetService_No": 0,
    "PaymentMethod_Bank_transfer_automatic": 0,
    "PaymentMethod_Credit_card_automatic": 0,
    "PaymentMethod_Electronic_check": 1,
    "PaymentMethod_Mailed_check": 0,
    "Contract_One_year": 0,
    "Contract_Two_year": 0
    }
    response = client.post("/churn/predict", json=payload)
    assert response.status_code == 200
    report = response.json()

    assert "will_churn" in report
    assert "churn_probability" in report
    assert "risk_level" in report


def test_predict_returns_valid_values():
    payload = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 24,
        "MultipleLines": 0,
        "OnlineSecurity": 1,
        "OnlineBackup": 0,
        "DeviceProtection": 0,
        "TechSupport": 1,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "PaperlessBilling": 0,
        "MonthlyCharges": 45.0,
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 0,
        "PaymentMethod_Bank_transfer_automatic": 1,
        "PaymentMethod_Credit_card_automatic": 0,
        "PaymentMethod_Electronic_check": 0,
        "PaymentMethod_Mailed_check": 0,
        "Contract_One_year": 1,
        "Contract_Two_year": 0
    }

    response = client.post("/churn/predict", json=payload)
    result = response.json()

    # probability must be between 0 and 1
    assert 0.0 <= result["churn_probability"] <= 1.0

    # risk level must be one of three values
    assert result["risk_level"] in ["high", "medium", "low"]

    # will_churn must be boolean
    assert isinstance(result["will_churn"], bool)