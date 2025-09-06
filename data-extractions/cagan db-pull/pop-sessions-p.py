import fastf1
from datetime import datetime
import fastf1.core
import fastf1.logger
import numpy as np
import pandas as pd
import os
from time import sleep
os.makedirs('./cache_dir', exist_ok=True)
fastf1.logger.set_log_level("ERROR")
fastf1.Cache.enable_cache('./cache_dir')
from multiprocessing import Pool

NUM_PROCESSES = 4

def pop_session_data(year : int, gp : str, sessionName : str, session : fastf1.core.Session,loc : str):
    dirpath = f"./sessions/{year}-{gp}-{sessionName}/"
    os.makedirs(dirpath, exist_ok=True)
    os.makedirs(f"./corners/", exist_ok=True)
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

if __name__ == "__main__":
    events = pd.read_csv('events.csv')
    sessions = []
    for (i,row) in events.iloc[0:].iterrows():
        year = row['year']
        gp = row["RoundNumber"]
        for i in range(1,6):
            try:
                sessionName = row["Session" + str(i)]
                session = fastf1.get_session(year, gp, sessionName)
                if session is not None:
                    sessions.append([year, gp, sessionName, session, row["Location"]])
            except Exception as e:
                print(f"Error: {e}")
                continue
    for i in range(0, len(sessions), NUM_PROCESSES):
        with Pool(processes=NUM_PROCESSES) as pool:
            pool.starmap(pop_session_data, sessions[i:i+NUM_PROCESSES])
        print(f"Finished {i} to {i+NUM_PROCESSES}")
        sleep(0.5)