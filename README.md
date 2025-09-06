# Formula 1 Lap Time Analysis: Understanding Performance Factors in Dry vs Wet Conditions

## 🏎️ Project Overview

This project investigates **how influential are parameters on F1 lap times** and **how the contribution of parameters changes relative to track wetness**. Using comprehensive telemetry data from 2022-2024 F1 seasons, we employ machine learning models to understand the complex relationships between various racing parameters and lap performance.

### 🎯 Research Questions
1. **How influential are parameters on lap time?**
2. **How do the contribution of parameters change relative to track wetness?**

## 📊 Dataset

The project analyzes **63,151 laps** from **three F1 seasons (2022-2024)**, containing:

### Features Analyzed
- **📍 Track Geometry**: Corner speeds, curvature, acceleration patterns
- **🏁 Race Strategy**: Lap number, stint, tyre life, tyre compound
- **🌤️ Weather Conditions**: Air/track temperature, humidity, pressure, wind speed/direction, rainfall
- **🚗 Car Performance**: Average corner exit speed, acceleration, curvature difference

### Target Variable
- **Lap Time** (normalized by subtracting race mean to account for track differences)

## 🔬 Methodology

### Data Collection & Processing
1. **`get_data.py`**: Extracts telemetry data using FastF1 API with multiprocessing
2. **`feature_extraction.py`**: Normalizes features and creates engineered variables
3. **`concat_data.py`**: Merges multiple seasons into unified dataset

### Advanced Lap Analysis (`lap_analysis.py`)
- **Spline interpolation** at 50Hz for smooth trajectory analysis
- **Curvature calculation** using differential geometry
- **Corner detection** and speed analysis at track geometry points
- **Acceleration computation** with median filtering for noise reduction

### Model Comparison
We evaluated multiple machine learning approaches:

1. **🌳 Random Forest** (`rf_analysis.ipynb`) - **FINAL MODEL**
2. **⚡ LightGBM** (`lgbm_analysis.ipynb`) - Gradient boosting comparison
3. **📈 GAM** (`gam.ipynb`) - Generalized Additive Models for interpretability
4. **📊 Polynomial Regression** (`poly_reg.py`) - Baseline comparison

## 🏆 Final Analysis: Random Forest Results

### Model Performance

| Condition | Test RMSE | Baseline RMSE | Improvement |
|-----------|-----------|---------------|-------------|
| **Dry** | **0.9241s** | 1.7491s | **47.2%** |
| **Wet** | **1.7822s** | 5.7180s | **68.8%** |

### Key Findings

#### 🌞 Dry Conditions - Top Feature Importance
1. **Lap Number** (0.37) - *Race progression and strategy timing*
2. **Pressure** (0.15) - *Atmospheric conditions affecting aerodynamics*
3. **Air Temperature** (0.12) - *Engine and aerodynamic performance*
4. **Humidity** (0.08) - *Weather stability and grip conditions*
5. **Average Corner Speed** (0.06) - *Cornering technique*

#### 🌧️ Wet Conditions - Top Feature Importance
1. **Average Corner Speed** (0.29) - *Critical for wet weather handling*
2. **Lap Number** (0.15) - *Strategy and tire degradation*
3. **Tyre Life** (0.12) - *Compound performance degradation*
4. **Air Temperature** (0.10) - *Thermal management in wet conditions*
5. **Intermediate Compound** (0.09) - *Tire compound selection*

### 🔍 SHAP Analysis Insights

The **SHAP (SHapley Additive exPlanations)** analysis reveals detailed feature impact patterns:

#### Dry Conditions
- **Lap Number**: Consistently increases lap times as race progresses (fuel load and tire degradation effects)
- **Pressure**: Generally negative SHAP values - higher atmospheric pressure improves performance
- **Air Temperature**: Mixed effects with some extreme outliers, suggesting optimal temperature ranges
- **Corner Exit Speed**: Predictably negative correlation - higher speeds reduce lap times
- **Tire Life**: Complex relationship with both positive and negative impacts depending on context

#### Wet Conditions  
- **Corner Speed**: Highly variable impact - extreme positive and negative values indicating significant driver skill differences
- **Intermediate Compounds**: Consistently positive SHAP values when used, suggesting suboptimal performance vs other compounds
- **Lap Number**: More variable impact than dry conditions, reflecting changing track conditions
- **Tire Life**: Stronger negative correlation in wet - fresher tires more critical
- **Temperature Effects**: Greater variance than dry conditions, indicating more sensitive thermal management requirements

### 🎯 Key Insights Summary

**Race Strategy Dominance**: Lap number being the top predictor in dry conditions highlights the massive impact of race progression, fuel load reduction, and tire degradation.

**Weather Sensitivity**: Wet conditions show much higher variance in feature impacts, particularly for driver-dependent factors like corner speed.

**Atmospheric Effects**: Pressure and humidity being significant predictors suggests aerodynamic performance is more sensitive to environmental conditions than typically assumed.

**Tire Strategy Complexity**: Different compound effects and tire life patterns between conditions demonstrate the strategic importance of tire management.

## 📁 Project Structure

```
├── 📊 Data Pipeline
│   ├── get_data.py              # FastF1 telemetry extraction
│   ├── concat_data.py           # Multi-season data merging  
│   ├── feature_extraction.py    # Feature engineering
│   └── lap_analysis.py          # Advanced telemetry analysis
│
├── 🤖 Machine Learning Models
│   ├── rf_analysis.ipynb        # 🏆 FINAL Random Forest analysis
│   ├── lgbm_analysis.ipynb      # LightGBM comparison
│   ├── gam.ipynb               # GAM interpretation
│   ├── rf.py                   # RF hyperparameter optimization
│   └── lgbm.py                 # LightGBM optimization
│
├── 📈 Analysis & Visualization
│   ├── plot_lap.py             # Lap data visualization
│   ├── plot_pca.py             # Principal component analysis
│   ├── plot_corr_data.py       # Correlation analysis
│   └── radarplot.py            # Performance comparison plots
│
├── 💾 Data Storage
│   ├── correlation-data/        # Processed datasets
│   ├── models/                 # Trained model artifacts
│   ├── shap/                   # SHAP explainer objects
│   └── plots/                  # Generated visualizations
│
└── 🏎️ Specialized Analysis
    ├── rain_split.py           # Wet/dry condition analysis
    └── load_npy_file.py        # Partial data recovery
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install fastf1 pandas scikit-learn matplotlib numpy scipy shap joblib lightgbm
```

### Running the Analysis

1. **Data Collection** (⚠️ Takes several hours):
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

## 📊 Key Visualizations

The project generates comprehensive visualizations:

- **Feature Importance Plots**: `plots/dry/` and `plots/wet/`
- **SHAP Summary Plots**: Understanding feature impact distributions
- **Individual Feature SHAP Plots**: Detailed per-feature analysis