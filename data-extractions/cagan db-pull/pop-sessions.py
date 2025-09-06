import fastf1
from datetime import datetime
import fastf1.core
import fastf1.logger
import numpy as np
fastf1.Cache.enable_cache('./cache_dir')
import pandas as pd
import os
from time import sleep
fastf1.logger.set_log_level("ERROR")

def pop_session_data(year : int, gp : str, sessionName : str, session : fastf1.core.Session,loc : str):
    dirpath = f"./sessions/{year}-{gp}-{sessionName}/"
    os.makedirs(dirpath, exist_ok=True)
    os.makedirs(f"./corners/", exist_ok=True)
    sleep(0.1)
    session.load()
    try:
        session.get_circuit_info().corners.to_csv(f"./corners/{loc}.csv")
    except Exception as e:
        print(f"Error corners: {e}")
    session.weather_data.to_csv(dirpath + "weather.csv")

    resultsdf = session.results
    resultsdf["dID"] = resultsdf["DriverNumber"] + resultsdf["Abbreviation"]
    resultsdf.to_csv(dirpath + "results.csv")
    for (_,row) in resultsdf[["dID","DriverNumber","Abbreviation"]].iterrows():
        driver = row["dID"]
        driverNumber = row["DriverNumber"]
        os.makedirs(dirpath + f"telemetry/{driver}/", exist_ok=True)
        try:
            laps = session.laps.pick_drivers(driverNumber)
            for i in range(1,len(laps)+1):
                laps.pick_laps(i).get_telemetry().to_csv(dirpath + f"telemetry/{driver}/{i}.csv")
        except Exception as e:
            print(f"Error tel: {e}")
            continue
        
last_event_index = 0
def load_last_event_index():
    global last_event_index
    if os.path.exists("last_event_index.txt"):
        with open("last_event_index.txt", "r") as f:
            last_event_index = int(f.read().strip())
load_last_event_index()

def write_last_event_index():
    global last_event_index
    with open("last_event_index.txt", "w") as f:
        f.write(str(last_event_index))

events = pd.read_csv('events.csv')
for (i,row) in events.iloc[last_event_index:].iterrows():
    last_event_index = i
    write_last_event_index()
    year = row['year']
    gp = row["RoundNumber"]
    for i in range(1,6):
        try:
            sessionName = row["Session" + str(i)]
            session = fastf1.get_session(year, gp, sessionName)
            if session is not None:
                pop_session_data(year, gp, sessionName, session, row["Location"])
        except Exception as e:
            print(f"Error: {e}")
            continue
