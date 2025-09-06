import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

path = "./correlation-data/r2024-all.npy"
newname = "./correlation-data/r2024-all.csv"

data = np.load(path, allow_pickle=True).item()
dfs = []
i = 0

for session,session_data in data.items():
    driver_datas = []
    for driver, driver_data in session_data.items():
        driver_datas.extend(driver_data)
    df = pd.DataFrame(driver_datas, columns=["LapTime", "AvgCornerSpeed", "Acceleration", "LapNumber", "Stint", "Compound", "TyreLife", "AirTemp", "Humidity", "Pressure", "TrackTemp", "WindDirection", "WindSpeed", "Rainfall", "AvgCurvature"])
    if len(df) == 0:
        print(f"Session {session} has no data for driver")
        continue
    df["race"] = session
    df["racemean"] = df["LapTime"].mean()
    dfs.append(df)
    i += 1
df = pd.concat(dfs, ignore_index=True)
df.to_csv(newname)