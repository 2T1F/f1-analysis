import fastf1
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from lap_analysis_nf import LapAnalysis
from scipy.signal import medfilt
import os

season = 2024
gp = 'Abu Dhabi'
event = 'Q'
driver = 'HAM'

fastf1.Cache.enable_cache('cache')
session = fastf1.get_session(season, gp, event)
session.load()

lap_analysis = LapAnalysis(session, driver, 3)

fig, ax1 = plt.subplots()

apex_index_list = lap_analysis.find_closest_distance_index()
exits = lap_analysis.get_exits()
# exits_argmin = lap_analysis.get_exits_argmin()

used_exit = exits

def mark_corners_on_plot(ax : Axes, ex : np.ndarray):
    for i in range(len(lap_analysis.corners)):
        ax.axvline(x=lap_analysis.corners[i], color='blue', linestyle='--')
        # show the corner number
        ax.text(lap_analysis.corners[i], 0, str(i), color='blue', fontsize=8)
    ax.set_xlabel("Distance (m)")
    # mark exit points
    for i in range(len(ex)):
        if ex[i] == False:
            continue
        ax.axvline(x=lap_analysis.dist[ex[i]], color='red', linestyle='--')
        # show the corner number
        ax.text(lap_analysis.dist[ex[i]], 0, str(i), color='red', fontsize=8)

def mark_points_on_map(ax : Axes, entrance : np.ndarray, exit : np.ndarray, entrance_color : str, exit_color : str):
    for i in range(len(entrance)):
        ax.scatter(lap_analysis.x[entrance[i]], lap_analysis.y[entrance[i]], color=entrance_color, s=10)
        # show the corner number
        ax.text(lap_analysis.x[entrance[i]], lap_analysis.y[entrance[i]], str(i), color='black', fontsize=8)
    for i in range(len(exit)):
        if exit[i] == False:
            continue
        ax.scatter(lap_analysis.x[exit[i]], lap_analysis.y[exit[i]], color=exit_color, s=10)
        # show the corner number
        ax.text(lap_analysis.x[exit[i]], lap_analysis.y[exit[i]], str(i), color='red', fontsize=8)

def make_table(ax : Axes, data : np.ndarray, title : str, labels : list[str]):
    ax.axis('tight')
    ax.axis('off')
    table_data = []
    for i in range(len(data)):
        row_data = [i]
        for j in range(len(data[i])):
            row_data.append(data[i][j])
        table_data.append(row_data)
    ax.table(cellText=table_data, colLabels=labels, loc='center')
    ax.set_title(title)

# put the entrance and exit points on the map
# mark_points_on_map(ax1, apex_index_list, exits, 'blue', 'red')
lap_analysis.scatter_acc(ax1)
mark_points_on_map(ax1, apex_index_list, used_exit, 'blue', 'red')
ax1.set_title("Track No Filter")
os.makedirs("plots/trackplot/nofilter", exist_ok=True)
fig.savefig("plots/trackplot/nofilter/trackplot.svg")

fig, ax3 = plt.subplots()
# show the avg corner speed in a table
avg_speed = lap_analysis.calc_avg_corner_speed()
kmh = np.round(avg_speed * 3.6,2)

avg_lat_acc = lap_analysis.calc_avg_corner_lat_acc()
avg_lat_acc = np.round(avg_lat_acc, 2)

table_data = np.stack((kmh, avg_lat_acc), axis=1)
make_table(ax3, table_data, "Corner Averages No Filter", ["Corner", "Speed (km/h)", "Lat Acc (g)"])
fig.savefig("plots/trackplot/nofilter/corner_averages.svg")

fig, ax2 = plt.subplots()
lap_analysis.dis_lat_scatter(ax2)
# mark_corners_on_plot(ax2, used_exit)
ax2.set_title("Lateral Acceleration No Filter")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Lateral Acceleration (g)")
ax2.legend()
ax2.grid()
# Save the lateral acceleration plot
fig.savefig("plots/trackplot/nofilter/lateral_acceleration.svg")

#plot curvature like lat g
fig, ax4 = plt.subplots()
ax4.plot(lap_analysis.dist, lap_analysis.ssC, label='Curvature', color='green')
# mark_corners_on_plot(ax4, used_exit)
ax4.set_title("Curvature No Filter")
ax4.set_xlabel("Distance (m)")
ax4.set_ylabel("Curvature (1/m)")
ax4.legend()
ax4.grid()
# Save the curvature plot
fig.savefig("plots/trackplot/nofilter/curvature.svg")

#plot acceleration  
fig, ax5 = plt.subplots()
ax5.plot(lap_analysis.dist, lap_analysis.get_acceleration(), label='Acceleration', color='orange')
# mark_corners_on_plot(ax5, used_exit)
ax5.set_title("Acceleration No Filter")
ax5.set_xlabel("Distance (m)")
ax5.set_ylabel("Acceleration (G)")
ax5.legend()
ax5.grid()
# Save the acceleration plot
fig.savefig("plots/trackplot/nofilter/acceleration.svg")

#plot speed
fig, ax6 = plt.subplots()
ax6.plot(lap_analysis.dist, lap_analysis.v * 3.6, label='Speed', color='purple')
# mark_corners_on_plot(ax6, used_exit)
ax6.set_title("Speed No Filter")
ax6.set_xlabel("Distance (m)")
ax6.set_ylabel("Speed (km/h)")
ax6.legend()
ax6.grid()
# Save the speed plot
fig.savefig("plots/trackplot/nofilter/speed.svg")

print(f"Avg Corner Lat Acc |G|: {np.mean(np.abs(avg_lat_acc))}")
print(f"Avg Corner Speed km/h: {np.mean(kmh)}")
print(f"Avg |G|: {np.mean(np.abs(lap_analysis.get_acceleration()))}")
print(f"Avg Speed km/h: {np.mean(lap_analysis.v * 3.6)}")

plt.show()