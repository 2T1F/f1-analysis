import matplotlib
import matplotlib.axes
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
# import linear regression
from sklearn.linear_model import RidgeCV
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import pyplot as plt
import numpy as np

categorical_feats = ['Compound', 'Rainfall']

def plot_feat_reg(feat_name, coef_map, intercept, ax : matplotlib.axes.Axes, c, deg):
    c0 = intercept
    x = np.arange(-2, 2, 0.1)
    y = np.zeros_like(x)
    y += c0
    for i in range(1,deg+1):
        if feat_name in coef_map:
            degree_key = f"{feat_name}^{i}" if i > 1 else feat_name
            y += coef_map[degree_key] * (x ** i)
    ax.plot(x, y, c)

def prep_degree_n(n, df_train, df_test):
    X_train_df = df_train.drop(columns=['LapTime', "racemean"])
    y_train = df_train['LapTime']
    X_test_df = df_test.drop(columns=['LapTime', "racemean"])
    y_test = df_test['LapTime']

    poly = PolynomialFeatures(degree=n, include_bias=False)
    X_train = poly.fit_transform(X_train_df)
    X_test = poly.transform(X_test_df)

    # 2) keep only those cols where feature_names[i] does NOT contain a space
    names = poly.get_feature_names_out(input_features=X_train_df.columns)
    mask  = [(" " not in name) for name in names]  # pure x^k only
    masked_names = [names[i] for i in range(len(names)) if mask[i]]
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]

    numeric_selector = [i for i in range(len(masked_names)) if not (masked_names[i].startswith(categorical_feats[0]) or masked_names[i].startswith(categorical_feats[1]))]
    preprocessor = ColumnTransformer(
        transformers=[
            ("ss", StandardScaler(), numeric_selector)
        ],
        remainder="passthrough"    # leave non-numeric cols untouched
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RidgeCV(cv=10))
    ])

    orig_names = X_train_df.columns.tolist()
    return (pipeline, X_train, y_train, X_test, y_test, masked_names, orig_names)

def train_deg_n(n, df_train, df_test):
    (pipeline, X_train, y_train, X_test, y_test, masked_names, orig_names) = prep_degree_n(n, df_train, df_test)
    score = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')
    return -score.mean()

df = pd.read_csv("./correlation-data/r22-24_processed.csv")
df = df.drop(columns=["race"])
df["LapTime"] = df["LapTime"] - df["racemean"]
dry_df = df[df['Rainfall'] == 0]
rain_df = df[df['Rainfall'] > 0]
baseline_rmse = root_mean_squared_error(dry_df['LapTime'], np.zeros_like(dry_df['LapTime']))
print(f"Baseline RMSE (zero model) for dry: {baseline_rmse}")
baseline_rmse_rain = root_mean_squared_error(rain_df['LapTime'], np.zeros_like(rain_df['LapTime']))
print(f"Baseline RMSE (zero model) for rain: {baseline_rmse_rain}")
dry_df_train, dry_df_test = train_test_split(dry_df, test_size=0.2, random_state=346346345)
rain_df_train, rain_df_test = train_test_split(rain_df, test_size=0.2, random_state=43534346)

# # For degree selection using Optuna
# def objective(trial : optuna.Trial):
#     n = trial.suggest_int("degree", 1, 5)
#     return train_deg_n(n, rain_df_train, rain_df_test)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=5)

# print("Best trial:")
# trial = study.best_trial
# print(f"  Value: {trial.value}")
# print(f"  Params: {trial.params}")

best_degree_dry = 1
best_degree_rain = 2

pipeline_dry, X_train, y_train, X_test, y_test, masked_names_d, orig_names  = prep_degree_n(best_degree_dry, dry_df_train, dry_df_test)
pipeline_dry.fit(X_train, y_train)
coef_dry = pipeline_dry.named_steps['model'].coef_
intercept_dry = pipeline_dry.named_steps['model'].intercept_

pipeline_rain, X_train_rain, y_train_rain, X_test_rain, y_test_rain, masked_names_r, orig_names = prep_degree_n(best_degree_rain, rain_df_train, rain_df_test)
pipeline_rain.fit(X_train_rain, y_train_rain)
coef_rain = pipeline_rain.named_steps['model'].coef_
intercept_rain = pipeline_rain.named_steps['model'].intercept_

feature_coef_dict_rain = {masked_names_r[i]: coef_rain[i] for i in range(len(masked_names_r))}
feature_coef_dict_dry = {masked_names_d[i]: coef_dry[i] for i in range(len(masked_names_d))}

scaler = StandardScaler()
scaler2 = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train_rain = scaler2.fit_transform(X_train_rain)

for feat in orig_names:
    fig,ax = plt.subplots()
    plot_feat_reg(feat, feature_coef_dict_rain, intercept_rain, ax, "blue", best_degree_rain)
    # plot_feat_reg(feat, feature_coef_dict_dry, intercept_dry, ax, "orange", best_degree_dry)
    ax.set_title(f"Polynomial Regression for {feat}")
    # Scatter rainy data blue dry data orange
    # ax.scatter(X_train[:, orig_names.index(feat)], y_train, color='red', label='Dry Data', alpha=0.5, s=0.5)
    # ax.scatter(X_train_rain[:, orig_names.index(feat)], y_train_rain, color='black', label='Rainy Data', alpha=0.5, s=0.5)

# Show RMSE for both
rmse_dry = root_mean_squared_error(y_test, pipeline_dry.predict(X_test))
rmse_rain = root_mean_squared_error(y_test_rain, pipeline_rain.predict(X_test_rain))
print(f"RMSE for dry data: {rmse_dry}")
print(f"RMSE for rain data: {rmse_rain}")
print(f"R^2 for dry data: {r2_score(y_test, pipeline_dry.predict(X_test))}")
print(f"R^2 for rain data: {r2_score(y_test_rain, pipeline_rain.predict(X_test_rain))}")

plt.show()
