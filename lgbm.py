import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
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

X_dry_train, X_dry_test, y_dry_train, y_dry_test = train_test_split(X_dry, y_dry, test_size=0.2, random_state=1241265)
X_wet_train, X_wet_test, y_wet_train, y_wet_test = train_test_split(X_wet, y_wet, test_size=0.2, random_state=1413647)

def run_cv(trial: optuna.Trial, X_train, y_train):
    preprocessor = ColumnTransformer(transformers=[
        ("scaler", StandardScaler(), numerical_features),
    ],remainder='passthrough')

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("lgbm", lgb.LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 4500, step=500),
            random_state=42486,
            metric='rmse',
            num_leaves=trial.suggest_int("num_leaves", 20, 50, step=10),
            max_depth=trial.suggest_int("max_depth", 3, 20, step=3),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
        ))
    ])

    score = cross_val_score(pipeline, X_train, y_train, cv=KFold(n_splits=10,shuffle=True), scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -score.mean()

def optimize(trial: optuna.Trial):
    return run_cv(trial, X_wet_train, y_wet_train)
study = optuna.create_study(direction="minimize")
study.optimize(optimize, n_trials=20)
print("Best hyperparameters wet:", study.best_params)

def optimize_dry(trial: optuna.Trial):
    return run_cv(trial, X_dry_train, y_dry_train)
study_dry = optuna.create_study(direction="minimize")
study_dry.optimize(optimize_dry, n_trials=20)
print("Best hyperparameters dry:", study_dry.best_params)

# Best hyperparameters dry: {'n_estimators': 2550, 'num_leaves': 50, 'max_depth': 12, 'learning_rate': 0.02}
# Best hyperparameters wet: {'n_estimators': 2050, 'num_leaves': 20, 'max_depth': 12, 'learning_rate': 0.05}