import os
import pandas as pd
from src.components.utils import load_data, load_model, evaluate_model, make_predictions, display_results


def main():
    print("Starting Customer Churn Prediction...")

    # 1. Load processed dataset
    df = load_data("notebook/data/processed/clean_data.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 2. Load trained model
    model = load_model("models/random_forest_churn.pkl")

    # 3. Evaluate on full dataset (optional, you might prefer test set)
    preds, probs = evaluate_model(model, X, y)

    # 4. Prepare results DataFrame
    results = X.copy().reset_index(drop=True)
    results["Churn_Prediction"] = preds
    results["Churn_Probability"] = probs

    # 5. Display first 10 predictions
    display_results(results, n=10)

    # 6. Save predictions to CSV
    results.to_csv("notebook/data/predictions/churn_predictions.csv", index=False)
    print("Predictions saved to data/predictions/churn_predictions.csv")


if __name__ == "__main__":
    main()