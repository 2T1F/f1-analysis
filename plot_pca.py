import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = "./correlation-data/r22-24_processed.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=["LapTime","racemean",])
feature_names = df.columns
# df = df[df["Rainfall"] == 0.0]

X_scaled = StandardScaler().fit_transform(df.to_numpy())
# X_scaled = df.values

# — 2) Fit PCA and look at explained variance —
pca = PCA().fit(X_scaled)
explained = pca.explained_variance_ratio_
cumvar    = np.cumsum(explained)

# How many PCs to reach 95%?
n_components_95 = np.searchsorted(cumvar, 0.95) + 1
print(f"Need {n_components_95} components to explain ≥95% variance")
# 2) Grab explained‐variance ratios and loadings
evr      = pca.explained_variance_ratio_           # shape (p,)
loadings = pca.components_                      # shape (n_features, p)

# 3) Compute per‐feature contributions
#    multiply each column i of loadings**2 by evr[i], then sum across PCs
contrib_matrix = loadings**2 * evr[np.newaxis, :]  # shape (n_features, p)
feature_contrib = contrib_matrix.sum(axis=1)       # shape (n_features,)

# 4) Package into a Series and sort
feat_contrib = pd.Series(feature_contrib, index=df.columns)
feat_contrib = feat_contrib.sort_values(ascending=False)
print("Feature  —  % of total variance explained")
print(np.cumsum(feat_contrib.mul(100)))

# 5) Plot the contributions
fig, ax = plt.subplots(figsize=(10, 8))
feat_contrib.plot(kind='barh', ax=ax)
ax.set_xlabel('% of total variance explained')
ax.set_title('Feature Contributions to PCA')
plt.tight_layout()

plt.show()