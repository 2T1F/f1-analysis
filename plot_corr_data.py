import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import standard scaler
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./correlation-data/r22-24_processed.csv")
corr_matrix = df.corr(method="spearman")
fig, ax = plt.subplots()
cax = ax.matshow(corr_matrix, cmap='coolwarm')
plt.colorbar(cax)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45)
ax.set_yticklabels(corr_matrix.columns)
for (i, j), val in np.ndenumerate(corr_matrix):
    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
plt.title('Correlation Matrix')

fig, ax = plt.subplots(figsize=(10, 8))
counts, bins = np.histogram(df["AvgCurvatureDiff"], bins=1000)
ax.stairs(counts, bins)
ax.set_xlabel("AvgCornerSpeed")
ax.set_ylabel("Counts")
ax.set_title("AvgCornerSpeed Distribution")


fig , ax7 = plt.subplots()
ax7.scatter(df["AvgCurvatureDiff"], df["LapTime"], s=1, c="blue")
# add explanation
ax7.set_xlabel("Variable")
ax7.set_ylabel("Lap Time")

plt.show()