import pandas as pd
import joblib

def predict_new(data_path, model_path):
    """
    Load a saved model and predict churn for new customers.
    Returns both predictions and probabilities.
    """

    # Load model
    model = joblib.load(model_path)

    # Load new data
    new_data = pd.read_csv(data_path)

    # Predictions (0 = No churn, 1 = Churn)
    preds = model.predict(new_data)

    # Probabilities ([:, 1] = probability of churn = 1)
    probs = model.predict_proba(new_data)[:, 1]

    # Display nicely
    results = new_data.copy()
    results["Churn_Prediction"] = preds
    results["Churn_Probability"] = probs

    print("✅ Predictions complete\n")
    print(results[["Churn_Prediction", "Churn_Probability"]])

    return results


if __name__ == "__main__":
    # Example usage
    predict_new("data/processed/sample_customers.csv", "models/random_forest_churn.pkl")
