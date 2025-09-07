# Formula 1 Lap Time Analysis: Understanding Performance Factors in Dry vs Wet Conditions

## Contributors

martelli.2108093@studenti.uniroma1.it - Flavio Martelli • sevinc.2109191@studenti.uniroma1.it - Özberk Sevinç • oral.2122393@studenti.uniroma1.it - Çağan Oral

## Project Overview

This project investigates **how influential are parameters on F1 lap times** and **how the contribution of parameters changes relative to track wetness**. Using telemetry data from 2022-2024 F1 seasons, we employ machine learning models to understand the relationships between various racing parameters and lap performance.

### Research Questions
1. **How influential are parameters on lap time?**
2. **How do the contribution of parameters change relative to track wetness?**

## Dataset

The project analyzes **63,149 laps** from **three F1 seasons (2022-2024)**, containing:

### Features Analyzed
- **Race Strategy**: Lap number, stint, tyre life, tyre compound
- **Weather Conditions**: Air/track temperature, humidity, pressure, wind speed/direction, rainfall
- **Car Performance**: Average corner exit speed, acceleration, curvature difference

### Target Variable
- **Lap Time** (normalized by subtracting race mean to account for track differences)

## Methodology

### Data Collection & Processing
1. **`get_data.py`**: Extracts telemetry data using FastF1 API with multiprocessing
2. **`feature_extraction.py`**: Normalizes features and creates engineered variables
3. **`concat_data.py`**: Merges multiple seasons into unified dataset

### Lap Analysis (`lap_analysis.py`)
- **Spline interpolation**
- **Curvature calculation**
- **Corner exit detection**
- **Acceleration calculation**

### Model Comparison
We evaluated multiple machine learning approaches:

1. **Random Forest** (`rf_analysis.ipynb`) - **FINAL MODEL**
2. **LightGBM** (`lgbm_analysis.ipynb`)
3. **GAM** (`gam.ipynb`)
4. **Polynomial Regression** (`poly_reg.py`)

## Final Analysis

You can check out the final slide (`f1.pptx`) for the analysis

## Project Structure

```
├── Data Pipeline
│   ├── get_data.py              # FastF1 telemetry extraction
│   ├── concat_data.py           # Multi-season data merging  
│   ├── feature_extraction.py    # Feature engineering
│   └── lap_analysis.py          # Telemetry analysis
│
├── Machine Learning Models
│   ├── rf_analysis.ipynb       # FINAL Random Forest analysis
│   ├── lgbm_analysis.ipynb     # LightGBM comparison
│   ├── gam.ipynb               # GAM interpretation
│   ├── rf.py                   # RF hyperparameter optimization
│   └── lgbm.py                 # LightGBM optimization
│
├── Analysis & Visualization
│   ├── plot_lap.py             # Lap data visualization
│   ├── plot_pca.py             # Principal component analysis
│   ├── plot_corr_data.py       # Correlation analysis
│   └── radarplot.py            # Performance comparison plots
│
├── Data Storage
│   ├── correlation-data/       # Processed datasets
│   ├── models/                 # Trained model artifacts
│   ├── shap/                   # SHAP explainer objects
│   └── plots/                  # Generated visualizations
```

## Quick Start

### Prerequisites
```bash
pip install fastf1 pandas scikit-learn matplotlib numpy scipy shap joblib lightgbm
```

### Running the Analysis

1. **Data Collection** (Takes several hours):
```bash
python get_data.py
```

2. **Feature Engineering**:
```bash
python feature_extraction.py
```

3. **Final Analysis**:
```bash
jupyter notebook rf_analysis.ipynb
```

## Key Visualizations

The project generates comprehensive visualizations:

- **Feature Importance Plots**: `plots/dry/` and `plots/wet/`
- **SHAP Summary Plots**: Understanding feature impact distributions
- **Individual Feature SHAP Plots**: Detailed per-feature analysis