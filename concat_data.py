import pandas as pd
import numpy as np

df1 = pd.read_csv("./correlation-data/r2022-all.csv", index_col=0)
df2 = pd.read_csv("./correlation-data/r2023-curv.csv", index_col=0)
df3 = pd.read_csv("./correlation-data/r2024-all.csv", index_col=0)

df = pd.DataFrame(np.concatenate([df1.values, df2.values, df3.values]), columns=df1.columns)
df.to_csv("./correlation-data/r22-24.csv")