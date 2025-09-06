import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# 1) Connect to your DB
conn = sqlite3.connect("f1_2024_race_only.db")

# 2) See which rounds got loaded for 2024
rounds_df = pd.read_sql(
    "SELECT DISTINCT round FROM Race_Laps WHERE season = 2024 ORDER BY round",
    conn
)
print("Rounds available for 2024:", rounds_df["round"].tolist())

# Let’s assume Abu Dhabi is round 22—adjust if yours is different
abu_dhabi_round = 22

# 3) Pull throttle telemetry for VER in that round
telemetry = pd.read_sql(
    f"""
    SELECT time_s, Throttle
      FROM Race_Telemetry_Raw
     WHERE season = 2024
       AND round  = {abu_dhabi_round}
       AND driver = 'VER'
     ORDER BY lap_number, time_s
    """,
    conn
)
conn.close()

# 4) Plot throttle over the entire race
plt.figure(figsize=(10,4))
plt.plot(telemetry["time_s"], telemetry["Throttle"], linewidth=0.8)
plt.xlabel("Time since lap start (s)")
plt.ylabel("Throttle (%)")
plt.title(f"Max Verstappen Throttle Trace – 2024 Abu Dhabi GP (Round {abu_dhabi_round})")
plt.tight_layout()
plt.show()

