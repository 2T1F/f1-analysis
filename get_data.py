import fastf1
import fastf1.core
import pandas as pd
import numpy as np
from lap_analysis import LapAnalysis
from multiprocessing import Pool
import os

fastf1.Cache.enable_cache('cache')

NUM_PROCESSES = 8 # Change this based on your CPU cores before running the script
GET_ALL = False
GET_RAINFALL = False
folder_name = "./correlation-data/races"
year_interval = range(2024, 2025)
session_type = 'R' # Race session

def get_lap_data(session,driver,i,lap):
            try:
                lap_params = []
                lap_time = lap['LapTime']
                if lap_time > pd.Timedelta(seconds=0):
                    lap_params.append(np.float64(lap_time / pd.Timedelta(seconds=1)))
                    lap_analysis = LapAnalysis(session, driver, i)
                    corner_speeds = lap_analysis.calc_avg_corner_speed()
                    corner_speeds = corner_speeds[corner_speeds != False]
                    if len(corner_speeds) > 0:
                        lap_params.append(np.mean(corner_speeds))
                    else:
                        return None
                    # corner_lats = lap_analysis.calc_avg_corner_lat_acc()
                    # corner_lats = np.abs(corner_lats[corner_lats != False])
                    # if len(corner_lats) > 0:
                    #     lap_params.append(np.mean(corner_lats))
                    # else:
                    #     return None
                    avg_curvature = np.mean(np.abs(lap_analysis.get_curvature_spl()))
                    a = np.mean(np.abs(lap_analysis.get_acceleration()))
                    lap_number = np.float64(lap["LapNumber"])
                    stint = np.float64(lap["Stint"])
                    compound = lap["Compound"]
                    tyre_life = np.float64(lap["TyreLife"])
                    
                    weather_data = lap.get_weather_data()
                    air_temp = np.float64(weather_data["AirTemp"])
                    humidity = np.float64(weather_data["Humidity"])
                    pressure = np.float64(weather_data["Pressure"])
                    track_temp = np.float64(weather_data["TrackTemp"])
                    w_direction = np.float64(weather_data["WindDirection"])
                    w_speed = np.float64(weather_data["WindSpeed"])
                    rainfall = np.float64(weather_data["Rainfall"])
                    lap_params.extend([a,lap_number,stint,compound,tyre_life,air_temp,humidity,pressure,track_temp,w_direction,w_speed, rainfall, avg_curvature])
                    return lap_params
            except Exception as e:
                print(f"Error processing lap {i} for driver {driver}: {e}")
                return None
            
if __name__ == "__main__":
    sessions : list[fastf1.core.Session] = []
    for year in year_interval:
        year_events = fastf1.get_event_schedule(year,include_testing=False)["EventName"].tolist()
        for event in year_events:
            session = fastf1.get_session(year, event, session_type)
            session.load(messages=False, weather=True)
            sessions.append(session)
    
    data = dict()
    sc = 0
    for session in sessions:
        sc += 1
        if session.weather_data is None:
            continue
        wd : pd.DataFrame = session.weather_data
        is_rainy = (wd["Rainfall"] == True).any()
        is_ok = GET_ALL or (GET_RAINFALL and is_rainy) or (not GET_RAINFALL and not is_rainy)
        if not is_ok:
            continue

        data[session.event['EventName']] = dict()
        curr_dict = data[session.event['EventName']]
        arg_input = []
        for driver in session.drivers:
            driver_laps = session.laps.pick_drivers(driver).pick_compounds(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]).pick_accurate()
            mask =  ~(driver_laps["FastF1Generated"])
            driver_laps = driver_laps[mask]
            driver_arg_input = []
            for i in range(1,len(driver_laps)):
                lap = driver_laps.iloc[i]
                driver_arg_input.append([session, driver, i, lap])
            arg_input.append(driver_arg_input)
        # get driver alias
        c = 0
        for driver_arg_input in arg_input:
            c += 1
            print(f"Processing driver {c}/{len(arg_input)} of {sc} / {len(sessions)}")
            driver_results = []
            with Pool(NUM_PROCESSES) as pool:
                results = pool.starmap(get_lap_data, driver_arg_input)
                for result in results:
                    if result is not None:
                        driver_results.append(result)
            try:
                driver_num = driver_arg_input[0][1]
                driver_alias = session.get_driver(driver_num)["FullName"].replace(" ", "_")
                val = curr_dict.get(driver_alias, [])
                val.extend(driver_results)
                curr_dict[driver_alias] = val
                np.save(f"corr_data.npy", data)
            except Exception as e:
                print(f"Error processing driver {driver_num}: {e}")

    os.makedirs(folder_name, exist_ok=True)
    dfs : list[pd.DataFrame] = []
    session_names = []
    for session, data in data.items():
        session_names.append(session)
        grouped_data = []
        for driver, driver_data in data.items():
            if len(driver_data) == 0:
                continue
            grouped_data.extend(driver_data)
        df = pd.DataFrame(grouped_data, columns=["LapTime", "AvgCornerSpeed", "Acceleration", "LapNumber", "Stint", "Compound", "TyreLife", "AirTemp", "Humidity", "Pressure", "TrackTemp", "WindDirection", "WindSpeed", "Rainfall", "AvgCurvature"])
        dfs.append(df)
    for i in range(len(dfs)):
        df = dfs[i]
        df.to_csv(os.path.join(folder_name, f"{session_names[i]}.csv"), index=False)