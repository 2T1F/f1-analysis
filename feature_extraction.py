import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

name = "./correlation-data/r22-24"
# name = "./correlation-data/dry"

df = pd.read_csv(f"{name}.csv", index_col=0)
print("NA rows:", df.isna().sum().sum())
df = df.dropna()
print("Length of df:", len(df))

dfs = []
df.groupby("race").apply(lambda x : dfs.append(x))
for i in range(len(dfs)):
    group = dfs[i]
    group = group.drop(columns=["WindDirection"])
    group["TyreLife"] = group["TyreLife"] / group["LapNumber"].max()
    group["SpL"] = group["Stint"] / group["LapNumber"]
    group["LapNumber"] = group["LapNumber"] / group["LapNumber"].max()
    group["CornerForce"] = group["AvgCornerSpeed"]**2 / group["AvgCurvature"].mean()
    group["AvgCurvatureDiff"] = (group["AvgCurvature"] - group["AvgCurvature"].mean()) / group["AvgCurvature"].std()
    group["TempRatio"] = group["AirTemp"] / group["TrackTemp"]
    dfs[i] = group

df = pd.concat(dfs, ignore_index=True)
categorical_feats = ['Compound', 'Rainfall']
df = pd.get_dummies(df, columns=["Compound"], drop_first=False)

df.to_csv(f"{name}_processed.csv", index=False)