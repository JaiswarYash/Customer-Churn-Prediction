import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(input_path, output_path):
    # Load raw dataset
    df = pd.read_csv(input_path)

    # Drop columns not useful for modeling
    df = df.drop(["customerID", "TotalCharges"], axis=1, errors="ignore")

    # Copy dataset for backup (like in your notebook)
    churn_pred_copy = df.copy()

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Apply Label Encoding to categorical features
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"✅ Clean data saved to {output_path}")

    # Return both cleaned and backup copy (if used programmatically)
    return df, churn_pred_copy

if __name__ == "__main__":
    # Example run
    clean_data("data/raw/customer_churn.csv", "data/processed/clean_data.csv")
