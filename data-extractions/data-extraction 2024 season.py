import os
import time
import fastf1
from fastf1 import Cache, get_session
import pandas as pd
from sqlalchemy import create_engine

# 0) Setup cache and fresh DB
os.makedirs("f1_cache", exist_ok=True)
Cache.enable_cache("f1_cache")

DB_PATH = "f1_2024_race_only.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

# 1) Pull race sessions only for 2024
season     = 2024
max_rounds = 25

for rnd in range(1, max_rounds+1):
    try:
        session = get_session(season, rnd, "R")
        session.load()
    except Exception:
        continue

    print(f"Processing 2024 Round {rnd} Race – drivers: {len(session.laps)}")

    # 1a) write laps
    laps = session.laps.copy()
    laps["season"] = season
    laps["round"]  = rnd
    laps[["season","round","Driver","LapNumber","LapTime","Position"]] \
        .to_sql("Race_Laps", engine, if_exists="replace", index=False)

    # 1b) write raw telemetry, converting Time → time_s (seconds)
    for _, lap in laps.iterlaps():
        tel = lap.get_telemetry().copy()

        # Convert Timedelta 'Time' to float seconds and drop it
        tel["time_s"] = tel["Time"].dt.total_seconds()
        tel = tel.drop(columns=["Time"])  # now no timedelta left

        # Annotate and push to SQL
        tel["season"]     = season
        tel["round"]      = rnd
        tel["driver"]     = lap["Driver"]
        tel["lap_number"] = lap["LapNumber"]

        tel.to_sql(
            "Race_Telemetry_Raw",
            engine,
            if_exists="append",
            index=False
        )

    time.sleep(1)

print("\nDone writing 2024 race laps and telemetry.")
