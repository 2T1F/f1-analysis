import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
# import linear regression
from sklearn.linear_model import LinearRegression, RidgeCV
import numpy as np
from matplotlib import pyplot as plt

def make_poly_pipeline(degree):
    pipeline = Pipeline([("pf", PolynomialFeatures(degree=degree, include_bias=False)), ("r", RidgeCV(cv=10))])
    return pipeline

def make_linear_pipeline():
    pipeline = Pipeline([("r", LinearRegression(n_jobs=-1))])
    return pipeline

def bic(n, rss, k):
    return n * np.log(rss / n) + (k + 1) * np.log(n)

def fit_feature(X, y, feat_name, degree):
    pipeline = make_linear_pipeline()
    if degree > 1:
        pipeline = make_poly_pipeline(degree)
    pipeline.fit(X[[feat_name]],y)
    x_new = X[[feat_name]]
    y_p = pipeline.predict(x_new)
    rss = np.sum((y - y_p) ** 2)
    return [rss, pipeline]

def plot_regression(inc,coef,ax,x):
    scaler = StandardScaler()
    x = x.to_numpy().reshape(-1, 1)
    scaler = scaler.fit(x)
    x = x.reshape(-1,)
    x_normal_plot = np.arange(-2,2, 0.01)
    x_normal_plot = scaler.inverse_transform(x_normal_plot.reshape(-1, 1)).reshape(-1,)
    y = np.ones_like(x_normal_plot) * inc
    for i in range(len(coef)):
        y += coef[i] * (x_normal_plot ** (i+1))
    ax.plot(x_normal_plot, y, color='red')
    # ax.set_xlabel('Feature Value')
    # ax.set_ylabel('Target Value')
    return ax

degrees = [1, 2, 3, 4, 5]

df = pd.read_csv("./correlation-data/r22-24_processed.csv")

categorical_feats = ['Compound', 'Rainfall']
X = df.drop(columns=['LapTime'])
y = df['LapTime'] - df['racemean']
numeric_selector = X.columns[~X.columns.str.startswith(tuple(categorical_feats))].to_list()
print("Numeric features:", numeric_selector)
scaler = StandardScaler()
X[numeric_selector] = scaler.fit_transform(X[numeric_selector])

n = len(X)

best_degrees = {}
for feat_name in X.columns:
    bic_vals = []
    rss_vals = []
    for degree in degrees:
        rss, pipeline = fit_feature(X, y, feat_name, degree)
        rss_vals.append(rss)
        if degree == 1:
            bic_val = bic(n, rss, 1)
            bic_vals.append(bic_val)
            continue
        bic_val = bic(n, rss, len(pipeline.named_steps["pf"].get_feature_names_out()))
        bic_vals.append(bic_val)
    best_degree = degrees[np.argmin(bic_vals)]
    print(f"Best degree for {feat_name}: {best_degree} with ll: {min(bic_vals)}")
    best_degrees[feat_name] = best_degree
print("Best degrees for each feature:")
for feat_name, degree in best_degrees.items():
    print(f"{feat_name}: {degree}")
for feat_name, degree in best_degrees.items():
    rss, pipeline = fit_feature(X, y, feat_name, degree)
    coef = pipeline.named_steps["r"].coef_
    inc = pipeline.named_steps["r"].intercept_
    x = X[feat_name]
    fig, ax = plt.subplots()
    ax.set_title(f"Regression for {feat_name} (degree {degree})")
    ax.set_xlabel(feat_name)
    ax.set_ylabel("Lap Time")
    ax.scatter(x, y, s=1, alpha=0.5)
    ax = plot_regression(inc, coef, ax, x)
    # Show R^2 * 100 as text on the plot
    r2 = pipeline.score(X[[feat_name]], y)
    ax.text(0.05, 0.95, f"$R^2 percentage$ = {r2 * 100:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.show()