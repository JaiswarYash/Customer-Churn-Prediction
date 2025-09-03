from components.utils import load_data, split_data, save_model
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def train_model(data_path, model_path):
    df = load_data(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    save_model(rf_pipeline, model_path)
