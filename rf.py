import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import optuna

df = pd.read_csv("./correlation-data/r22-24_processed.csv")
df["LapTime"] = df["LapTime"] - df["racemean"]
df = df.drop(columns=["racemean"])
df_dry = df[df["Rainfall"] == 0.0]
df_dry = df_dry.drop(columns=["Rainfall"])
X_dry = df_dry.drop(columns=['LapTime'])
y_dry = df_dry['LapTime']
df_wet = df[df["Rainfall"] == 1.0]
df_wet = df_wet.drop(columns=["Rainfall"])
X_wet = df_wet.drop(columns=['LapTime'])
y_wet = df_wet['LapTime']
df = df.drop(columns=["Rainfall"])

X = df.drop(columns=['LapTime'])
y = df['LapTime']

categorical_feats = ['Compound', "Rainfall"]
categorical_feats = [c for c in X.columns if c.startswith(tuple(categorical_feats))]
numerical_features = [c for c in X.columns if not c.startswith(tuple(categorical_feats))]

X_dry_train, X_dry_test, y_dry_train, y_dry_test = train_test_split(X_dry, y_dry, test_size=0.2, random_state=36536)
X_wet_train, X_wet_test, y_wet_train, y_wet_test = train_test_split(X_wet, y_wet, test_size=0.2, random_state=3463463)

def run_cv(n_estimators, max_depth, X_train, y_train):
    """Create a pipeline with the given number of estimators."""
    col_transformer = ColumnTransformer(transformers=[
            ("scaler", StandardScaler(), numerical_features)],remainder='passthrough')
    pipeline = Pipeline([
        ("scaler", col_transformer),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=3735
        ))
        ])
    score = cross_val_score(pipeline, X_train, y_train, cv=KFold(n_splits=10, shuffle=True,random_state=57457), scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -score.mean()

def optimize(trial: optuna.Trial):
    """Optimize the number of estimators for the Random Forest."""
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=100)
    max_depth = trial.suggest_int("max_depth", 5, 50, step=5)
    return run_cv(n_estimators, max_depth, X_wet_train, y_wet_train)

study = optuna.create_study(direction="minimize")
study.optimize(optimize, n_trials=10)
print("Best hyperparameters dry:", study.best_params)

# For wet
def optimize_wet(trial: optuna.Trial):
    """Optimize the number of estimators for the Random Forest."""
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=100)
    max_depth = trial.suggest_int("max_depth", 5, 50, step=5)
    return run_cv(n_estimators, max_depth, X_wet_train, y_wet_train)

study_wet = optuna.create_study(direction="minimize")
study_wet.optimize(optimize_wet, n_trials=10)
print("Best hyperparameters wet:", study_wet.best_params)
# ----- END CV ----

# {'n_estimators': 150, 'max_depth': 50} for dry
# {'n_estimators': 350, 'max_depth': 20} for wet