import pandas as pd
import numpy as np

df1 = pd.read_csv("./correlation-data/r2022-all.csv", index_col=0)
df2 = pd.read_csv("./correlation-data/r2023-curv.csv", index_col=0)
df3 = pd.read_csv("./correlation-data/r2024-all.csv", index_col=0)

# Ensure Compound_INTERMEDIATE and Compound_WET exists in all dataframes

def split_by_rainfall(df : pd.DataFrame):
    df = df.dropna()
    dfs : list[pd.DataFrame] = []
    df.groupby("race").apply(lambda x : dfs.append(x))
    for i in range(len(dfs)):
        group = dfs[i]
        if (group["Rainfall"] == 1.0).any():
            group["Rainfall"] = 1.0
        dfs[i] = group
    return pd.concat(dfs, ignore_index=True)

df1 = split_by_rainfall(df1)
df2 = split_by_rainfall(df2)
df3 = split_by_rainfall(df3)

pd.concat([df1, df2, df3]).to_csv("./correlation-data/r22-24.csv")