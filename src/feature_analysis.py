import pandas as pd
import joblib
import json

model = joblib.load('models/random_forest_model.pkl')
with open('models/feature_names.json') as f:
    feature_names = json.load(f)

# extract imprtance features from random forest
rf_model = model.named_steps['model']

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.to_string())
print(f"\nTotal features: {len(importance_df)}")
