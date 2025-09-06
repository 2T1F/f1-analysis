import joblib
import pandas as pd
import numpy as np

# Load the saved models
pipeline_dry = joblib.load("./models/pipeline_dry.pkl")
pipeline_wet = joblib.load("./models/pipeline_wet.pkl")

# Load the data to get feature names
df = pd.read_csv("./correlation-data/r22-24_processed.csv")
df["race"] = pd.factorize(df["race"])[0]
df = df.drop(columns=["race"])
df["LapTime"] = df["LapTime"] - df["racemean"]
df = df.drop(columns=["racemean", "SpL", "CornerForce", "TempRatio"])
df_dry = df[df["Rainfall"] == 0.0]
df_dry = df_dry.drop(columns=["Rainfall"])
X_dry = df_dry.drop(columns=['LapTime'])
df_wet = df[df["Rainfall"] == 1.0]
df_wet = df_wet.drop(columns=["Rainfall"])
X_wet = df_wet.drop(columns=['LapTime'])

# Remove wet compounds from dry dataset
X_dry = X_dry.drop(columns=["Compound_INTERMEDIATE", "Compound_WET"])

# Get feature names
dry_feature_names = pipeline_dry.named_steps["preprocessor"].get_feature_names_out(X_dry.columns.tolist())
dry_feature_names = [name.split('__')[-1] for name in dry_feature_names]

wet_feature_names = pipeline_wet.named_steps["preprocessor"].get_feature_names_out(X_wet.columns.tolist())
wet_feature_names = [name.split('__')[-1] for name in wet_feature_names]

# Get feature importances
dry_importances = pipeline_dry.named_steps['rf'].feature_importances_
wet_importances = pipeline_wet.named_steps['rf'].feature_importances_

# Sort by importance
dry_indices = dry_importances.argsort()[::-1]
wet_indices = wet_importances.argsort()[::-1]

print("=== DRY CONDITIONS - TOP 5 FEATURE IMPORTANCES ===")
for i in range(5):
    idx = dry_indices[i]
    feature_name = dry_feature_names[idx]
    importance = dry_importances[idx] * 100
    print(f"{i+1}. {feature_name}: {importance:.1f}%")

print("\n=== WET CONDITIONS - TOP 5 FEATURE IMPORTANCES ===")
for i in range(5):
    idx = wet_indices[i]
    feature_name = wet_feature_names[idx]
    importance = wet_importances[idx] * 100
    print(f"{i+1}. {feature_name}: {importance:.1f}%")
