import fastf1
from datetime import datetime
import numpy as np
fastf1.Cache.enable_cache('./cache_dir')
import pandas as pd

start = 2018
end = 2026
events = None
for year in range(start, end):
    event = fastf1.get_event_schedule(year)
    event["year"] = year
    events = pd.concat([events,event])
events.to_csv('events.csv')