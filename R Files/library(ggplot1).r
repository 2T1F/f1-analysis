# Load required packages
library(f1dataR)    # for load_driver_telemetry() and theme_dark_f1()
library(dplyr)      # for data manipulation
library(purrr)      # for map_dfr()
library(ggplot2)    # for plotting
library(f1dataR)    




lec <- load_driver_telemetry(2022, 1, "Q", driver = "LEC", laps = "fastest")
ham <- load_driver_telemetry(2022, 1, "Q", driver = "HAM", laps = "fastest")
per <- load_driver_telemetry(2022, 1, "Q", driver = "PER", laps = "fastest")

telem <- bind_rows(lec, ham, per) %>%
  select(rpm, speed, n_gear, throttle, brake, drs, distance, time, driver_code) %>%
  mutate(drs = ifelse(drs == 12, 1, 0))

drivercolours <- c(
  "LEC" = get_driver_color("LEC", 2022, 1),
  "HAM" = get_driver_color("HAM", 2022, 1),
  "PER" = get_driver_color("PER", 2022, 1)
)

telem_plot_speed <- ggplot(telem, aes(x = distance, y = speed, color = driver_code)) +
  geom_path() +
  scale_color_manual(values = drivercolours) +
  theme_dark_f1(axis_marks = TRUE) +
  ggtitle("2022 Bahrain Grand Prix Qualifying Telemetry", subtitle = "Speed vs Distance in lap") +
  xlab("Distance (m)") +
  ylab("Speed") +
  labs(color = "Driver")


telem_plot_speed