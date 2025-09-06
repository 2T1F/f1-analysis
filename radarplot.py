import fastf1
import numpy as np
import matplotlib.pyplot as plt

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

def compute_driver_metrics(driver: str, start_year: int, end_year: int):
    """
    Compute average metrics for a driver over a range of seasons:
      - Peak braking (max negative longitudinal G)
      - Peak lateral acceleration (max lateral G)
      - Lap‐time consistency (std dev of qualifying lap times)
      - Improvement (arbitrary score 1–10)
      - Team Contribution (arbitrary score 1–10)
    """
    braking_vals, lateral_vals, consistency_vals = [], [], []

    for year in range(start_year, end_year + 1):
        schedule = fastf1.get_event_schedule(year)
        for gp in schedule['EventName']:
            try:
                session = fastf1.get_session(year, gp, 'Q')
                session.load()

                laps = session.laps.pick_drivers(driver)
                if laps.empty:
                    continue

                # Consistency: std dev of all Q lap times (in seconds)
                times = laps['LapTime'].dt.total_seconds()
                consistency_vals.append(np.std(times))

                # Telemetry of fastest lap
                lap = laps.pick_fastest()
                tel = (lap.get_telemetry()
                          .add_distance()
                          .set_index('Time')
                          .resample('1ms')
                          .interpolate()
                          .reset_index())

                # longitudinal acceleration (m/s² → G)
                t = tel['Time'] / np.timedelta64(1, 's')
                dt = np.gradient(t)
                v = tel['Speed'] / 3.6  # m/s
                a_long = np.gradient(v) / dt
                braking_vals.append(-np.min(a_long) / 9.81)

                # lateral acceleration via curvature
                x, y, dist = tel['X']*0.1, tel['Y']*0.1, tel['Distance']
                dx_ds  = np.gradient(x, dist)
                dy_ds  = np.gradient(y, dist)
                ddx_ds = np.gradient(dx_ds, dist)
                ddy_ds = np.gradient(dy_ds, dist)
                curvature = np.abs(dx_ds * ddy_ds - dy_ds * ddx_ds) / ((dx_ds**2 + dy_ds**2)**1.5 + 1e-8)
                a_lat = v**2 * curvature
                lateral_vals.append(np.max(a_lat) / 9.81)

            except Exception:
                continue

    return {
        'Peak Braking (G)':   np.mean(braking_vals)     if braking_vals     else 0,
        'Peak Lateral (G)':   np.mean(lateral_vals)     if lateral_vals     else 0,
        'Consistency (s)':    np.mean(consistency_vals) if consistency_vals else 0,
        'Improvement':        np.random.randint(1, 11),
        'Team Contribution':  np.random.randint(1, 11)
    }

def normalize_metrics(raw_metrics: dict):
    """
    Normalize selected raw metrics to a 1–10 scale so all axes are comparable.
    """
    norms = {}
    # define expected ranges for each metric
    ranges = {
        'Peak Braking (G)':   (0.0, 5.0),
        'Peak Lateral (G)':   (0.0, 6.0),
        'Consistency (s)':    (0.0, 1.0),
        # Improvement and Team already 1–10
    }
    for k, v in raw_metrics.items():
        if k in ranges:
            lo, hi = ranges[k]
            # clip then scale to 1–10
            clipped = max(lo, min(v, hi))
            norms[k] = 1 + (clipped - lo) * 9.0 / (hi - lo)
        else:
            # already in 1–10
            norms[k] = v
    return norms

def plot_radar(metrics: dict, title: str = "Driver Performance Radar"):
    """
    Plots a radar chart of the provided metrics.
    """
    categories = list(metrics.keys())
    values = list(metrics.values())
    # close loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='teal', linewidth=2)
    ax.fill(angles, values, color='teal', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([1, 3, 5, 7, 9])
    ax.set_ylim(1, 10)
    ax.set_title(title, y=1.1, fontweight='bold')
    ax.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    driver = 'HAM'
    start_year = 2024
    end_year = 2024

    raw_metrics = compute_driver_metrics(driver, start_year, end_year)
    print("Raw metrics:", raw_metrics)

    scaled = normalize_metrics(raw_metrics)
    print("Scaled metrics (1–10):", scaled)

    plot_radar(scaled, title=f"{driver} ({start_year}-{end_year}) Performance Radar")
