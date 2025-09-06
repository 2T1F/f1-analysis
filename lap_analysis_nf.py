import fastf1
import fastf1.core
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.signal import savgol_filter, medfilt

def drop_useless_columns(df : pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['Status', 'SessionTime', 'Date', 'Source'])

class LapAnalysis:
    x : pd.Series
    y : pd.Series
    v : pd.Series
    time : pd.Series
    a : pd.Series
    ay : pd.Series
    corners : pd.Series
    g_thresh = 6
    dist_weight_factor = 0.01 # for get_exit_argmin()
    acc_window = 11
    prefilter_window = 51
    curvature_window = 51
    speed_window = 51
    lat_window = 101
    interpolation_freq = 50
    corner_perc_tresh = 1.0 - 0.9 # for get_exit()
    corner_exit_window = interpolation_freq * 5
    corner_match_window = interpolation_freq # used to maximize curvature of apex, use this window to find the correct apex

    def __init__(self, session : fastf1.core.Session, driver : str, lap : int):
        telemetry = session.laps.pick_drivers(driver).pick_laps(lap).get_telemetry()
        corner = session.get_circuit_info().corners
        self.time = telemetry["Time"] / np.timedelta64(1, "s")
        # self.v = telemetry["Speed"] / 3.6
        self.x = telemetry["X"] * 0.1
        self.y = telemetry["Y"] * 0.1
        self.dist = telemetry["Distance"]
        self.corners = corner["Distance"]
        self.cornersx = (corner["X"] * 0.1).to_numpy()
        self.cornersy = (corner["Y"] * 0.1).to_numpy()
        self.v = None
        self.a = None
        self.ay = None
        self.dx = None
        self.dy = None
        self.dx2 = None
        self.dy2 = None
        self.corner_match = None
        self.ssC = None
        self.csC = None
        self.exits = None
        self.avg_corner_speed = None
        self.avg_corner_lat_acc = None
        self.spline_interpolate()

    def spline_interpolate(self):
        self.ssx = make_smoothing_spline(self.time, self.x)
        self.ssy = make_smoothing_spline(self.time, self.y)
        self.ssd = make_smoothing_spline(self.time, self.dist)
        self.time = np.arange(0, self.time[len(self.time) - 1], 1 / self.interpolation_freq)
        self.x = self.ssx(self.time)
        self.y = self.ssy(self.time)
        self.dist = self.ssd(self.time)
        self.dx = self.ssx(self.time, 1)
        # self.dx = savgol_filter(self.dx, window_length=self.prefilter_window, polyorder=3)
        self.dy = self.ssy(self.time, 1)
        # self.dy = savgol_filter(self.dy, window_length=self.prefilter_window, polyorder=3)
        self.v = np.sqrt(self.dx**2 + self.dy**2)
        # self.v = savgol_filter(self.v, window_length=self.speed_window, polyorder=3)
        self.dx2 = self.ssx(self.time, 2)
        # self.dx2 = savgol_filter(self.dx2, window_length=self.prefilter_window, polyorder=3)
        self.dy2 = self.ssy(self.time, 2)
        # self.dy2 = savgol_filter(self.dy2, window_length=self.prefilter_window, polyorder=3)

    def get_acceleration(self):
        if self.a is None:
            a = (self.dx * self.dx2 + self.dy * self.dy2) / (np.sqrt(self.dx**2 + self.dy**2) + np.finfo(float).eps) / 9.81
            self.a = a
        return self.a
    
    def get_acceleration_nofilter(self):
        a = (self.dx * self.dx2 + self.dy * self.dy2) / (np.sqrt(self.dx**2 + self.dy**2) + np.finfo(float).eps) / 9.81
        return a
    
    def get_curvature_spl(self):
        if self.ssC is not None:
            return self.ssC
        ssC = (self.dx * self.dy2 - self.dy * self.dx2) / (np.power(self.dx**2 + self.dy**2, 1.5) + np.finfo(float).eps)
        self.ssC = ssC
        return self.ssC
    
    def get_curvature_nofilter(self):
        ssC = (self.dx * self.dy2 - self.dy * self.dx2) / (np.power(self.dx**2 + self.dy**2, 1.5) + np.finfo(float).eps)
        return ssC
    
    def get_lateral_acceleration(self):
        if self.ay is not None:
            return self.ay
        ay = np.square(self.v) * self.get_curvature_spl() / 9.81
        # ay = np.clip(ay, -self.g_thresh, self.g_thresh)
        self.ay = ay
        return self.ay
    
    def dis_lat_scatter(self, ax : Axes):
        ax.plot(self.dist, self.get_lateral_acceleration(), color='blue', label='Lateral Acceleration')

    def maximize_apex_curvature(self, corner_match : np.ndarray):
        # find the index of the apex
        if self.ssC is None:
            self.get_curvature_spl()
        if self.csC is None:
            self.csC = np.abs(self.ssC)
        # find the index of the apex for each index
        for i in range(len(corner_match)):
            # get the index of the apex
            if corner_match[i] == -1:
                continue
            # get the index of the apex
            start = corner_match[i] - int(self.corner_match_window)
            end = corner_match[i] + int(self.corner_match_window)
            # make sure the index is within bounds
            if start < 0:
                start = 0
            if end > len(self.csC):
                end = len(self.csC)
            # get the index of the maximum value
            max_index = np.argmax(self.csC[start:end]) + start
            corner_match[i] = max_index
        return corner_match
    
    def find_closest_distance_index(self):
        if self.corner_match is not None:
            return self.corner_match
        driverx = self.x.reshape(-1,1)
        drivery = self.y.reshape(-1,1)
        driverx = driverx.repeat(repeats=self.cornersx.shape[0],axis=1)
        drivery = drivery.repeat(repeats=self.cornersx.shape[0],axis=1)
        xdiff = driverx - self.cornersx
        ydiff = drivery - self.cornersy
        coord_diff = np.sqrt(xdiff**2 + ydiff**2)
        self.corner_match = coord_diff.argmin(axis=0)
        return self.corner_match
    
    def get_exits(self):
        if self.exits is not None:
            return self.exits
        apex_index_list = self.find_closest_distance_index()
        # apex_index_list = self.maximize_apex_curvature(apex_index_list)
        exits = []
        for i in range(len(apex_index_list)):
            exit = -1
            absc = np.abs(self.get_curvature_spl())
            curr_max = absc[apex_index_list[i]]
            curr_per_tresh = self.corner_perc_tresh
            curr_thresh = curr_max * curr_per_tresh
            max_index = 0
            if i == len(apex_index_list) - 1:
                max_index = len(absc)
            else:
                max_index = apex_index_list[i+1]
            max_index = min(max_index, apex_index_list[i] + self.corner_exit_window)
            while exit == -1:
                for j in range(apex_index_list[i], max_index):
                    if absc[j] > curr_max:
                        curr_max = absc[j]
                        curr_thresh = curr_max * curr_per_tresh
                        continue
                    if absc[j] < curr_thresh:
                        exit = j
                        break
                if exit == -1:
                    curr_per_tresh = curr_per_tresh * 2
                    if curr_per_tresh > 0.5:
                        break
                    curr_thresh = curr_max * curr_per_tresh
            if exit == -1:
                # print(f"Exit not found corner: {i}")
                exits.append(False)
                continue
            exits.append(exit)
        self.exits = exits
        return exits
    
    def get_exits_argmin(self):
        apex_index_list = self.find_closest_distance_index()
        abslat = np.abs(self.get_curvature_spl())
        exits = []
        for i in range(len(apex_index_list)):
            max_index = 0
            if i == len(apex_index_list) - 1:
                max_index = len(abslat)
            else:
                max_index = apex_index_list[i+1]
            corner_slice = abslat[apex_index_list[i]:max_index]
            index_list = np.arange(0,1,1/len(corner_slice))[:len(corner_slice)] * np.std(corner_slice)
            combined = corner_slice + index_list
            exits.append(combined.argmin() + apex_index_list[i])
        return exits
    
    def calc_avg_corner_lat_acc(self):
        if self.avg_corner_lat_acc is not None:
            return self.avg_corner_lat_acc
        # get the distance of the apex
        apex_index_list = self.find_closest_distance_index()
        # get the distance of the exit
        exits = self.get_exits()
        avg_lat_acc = []
        for i in range(len(apex_index_list)):
            if exits[i] == False:
                #take apex speed
                avg_lat_acc.append(False)
                continue
            vslice = self.get_lateral_acceleration()[apex_index_list[i]:exits[i]]
            # if the slice is actually one point then take the speed of the apex
            if len(vslice) == 0:
                avg_lat_acc.append(self.get_lateral_acceleration()[apex_index_list[i]])
                continue
            avg_lat_acc.append(np.mean(vslice))
        self.avg_corner_lat_acc = np.array(avg_lat_acc)
        return self.avg_corner_lat_acc
    
    def calc_avg_corner_speed(self):
        if self.avg_corner_speed is not None:
            return self.avg_corner_speed
        # get the distance of the apex
        apex_index_list = self.find_closest_distance_index()
        # get the distance of the exit
        exits = self.get_exits()
        avg_speed = []
        for i in range(len(apex_index_list)):
            if exits[i] == False:
                #take apex speed
                avg_speed.append(self.v[apex_index_list[i]])
                continue
            vslice = self.v[apex_index_list[i]:exits[i]]
            # if the slice is actually one point then take the speed of the apex
            if len(vslice) == 0:
                avg_speed.append(self.v[apex_index_list[i]])
                continue
            avg_speed.append(np.mean(vslice))
        self.avg_corner_speed = np.array(avg_speed)
        return self.avg_corner_speed
    
    def scatter_acc(self, ax : Axes):
        # scatter the position data to show the race map with coloring using lateral acceleration but resample 200Hz
        downsample_index = np.arange(0, len(self.x))
        ds_x = self.x[downsample_index]
        ds_y = self.y[downsample_index]
        ds_ay = self.get_lateral_acceleration()[downsample_index]
        ax.scatter(ds_x, ds_y, c=ds_ay, cmap='jet', s=5),

