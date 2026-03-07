from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title = "Customer Churn API")

# load model
model = joblib.load('models/random_forest_model.pkl')

class CustomerData(BaseModel):
    gender: int                                    # 1=Male, 0=Female
    SeniorCitizen: int                             # 1=Yes, 0=No
    Partner: int                                   # 1=Yes, 0=No
    Dependents: int                                # 1=Yes, 0=No
    tenure: int                                    # months
    MultipleLines: int                             # 1=Yes, 0=No
    OnlineSecurity: int                            # 1=Yes, 0=No
    OnlineBackup: int                              # 1=Yes, 0=No
    DeviceProtection: int                          # 1=Yes, 0=No
    TechSupport: int                               # 1=Yes, 0=No
    StreamingTV: int                               # 1=Yes, 0=No
    StreamingMovies: int                           # 1=Yes, 0=No
    PaperlessBilling: int                          # 1=Yes, 0=No
    MonthlyCharges: float                          # e.g. 65.5
    InternetService_Fiber_optic: int               # 1=Yes, 0=No
    InternetService_No: int                        # 1=Yes, 0=No
    PaymentMethod_Bank_transfer_automatic: int     # 1=Yes, 0=No
    PaymentMethod_Credit_card_automatic: int       # 1=Yes, 0=No
    PaymentMethod_Electronic_check: int            # 1=Yes, 0=No
    PaymentMethod_Mailed_check: int                # 1=Yes, 0=No
    Contract_One_year: int                         # 1=Yes, 0=No
    Contract_Two_year: int                         # 1=Yes, 0=No
# get
@app.get('/')
def index():
    return {"message": "Customer Churn ML API", "status": "healthy"}

# predict
@app.post("/churn/predict")
async def predict_churn(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])

    input_df = input_df.rename(columns={
        'InternetService_Fiber_optic': 'InternetService_Fiber optic',
        'PaymentMethod_Bank_transfer_automatic': 'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit_card': 'PaymentMethod_Credit card',
        'PaymentMethod_Credit_card_automatic': 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic_check': 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed_check': 'PaymentMethod_Mailed check',
        'Contract_One_year': 'Contract_One year',
        'Contract_Two_year': 'Contract_Two year'
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "will_churn": bool(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": "high" if probability > 0.7 else "medium" if probability > 0.4 else "low"
    }