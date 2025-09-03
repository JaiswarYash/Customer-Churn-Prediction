"""
utils.py
Reusable helper functions for ML projects (loading data, splitting, saving models, evaluation, etc.)
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# -----------------------------
# Data Handling
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into stratified train/test sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# -----------------------------
# Model Persistence
# -----------------------------
def save_model(model, path: str):
    """Save trained model pipeline to disk."""
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


def load_model(path: str):
    """Load a trained model pipeline from disk."""
    return joblib.load(path)


# -----------------------------
# Evaluation & Predictions
# -----------------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model with classification report + ROC-AUC.
    Returns predictions & probabilities.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("✅ Model Evaluation")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}\n")

    return y_pred, y_prob


def make_predictions(model, X):
    """Make predictions and return labels + probabilities."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs


def display_results(results: pd.DataFrame, n=10):
    """Display prediction results (preview)."""
    print("✅ Predictions complete\n")
    print(results[["Churn_Prediction", "Churn_Probability"]].head(n))
    return results


# -----------------------------
# Example Usage (Test Run)
# -----------------------------
if __name__ == "__main__":
    # Load data
    df = load_data("data/processed/churn_data.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Load a pre-trained model (assumes it's already saved)
    model = load_model("models/random_forest_churn.pkl")

    # Evaluate model
    preds, probs = evaluate_model(model, X_test, y_test)

    # Prepare results DataFrame
    results = X_test.copy().reset_index(drop=True)
    results["Churn_Prediction"] = preds
    results["Churn_Probability"] = probs

    # Display predictions
    display_results(results)
