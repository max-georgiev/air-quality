# Forecasting Air Quality with Linear Autoregressive Models

This repository contains the implementation for a final project of the *Applied Machine Learning in Python* course at LMU Munich, Summer Semester 2025. It focuses on forecasting air quality (specifically NO2 levels) using linear autoregressive models.  
It builds on materials from [AppliedML](https://github.com/max-georgiev/AppliedML), a fork of the original course repository by [@mselezniova](https://github.com/mselezniova/AppliedML).


## Requirements

To set up the development environment, first create and activate a Python virtual environment:

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source ./.venv/bin/activate
```
To install the required packages:

```setup
pip install -r requirements.txt
```

> **Note:** This project also depends on components from a [forked version of max-georgiev's repo](https://github.com/max-georgiev/AppliedML). Please make sure to clone that repository and use it alongside this one, as it provides essential utility functions and `__init__.py` files for modular imports.

## Data

This project uses the UCI Air Quality Dataset, originally from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality).

The dataset (`AirQualityUCI.csv`) is already included under `data/raw/`, so no manual download is necessary.

____________________________________________________________




## Training


To train a model for forecasting pollutant levels (e.g., NO₂), first ensure you’ve set the correct configuration in src/utils/config.py, including:

TARGET_POLLUTANT (e.g., 'NO2(GT)')

START_DATE and END_DATE

LAG_DEPTH (e.g., 24)


To train and evaluate the models for pollutant forecasting (e.g., NO₂), open and run the following Jupyter notebook:

`src/notebooks/main.ipynb`

This notebook performs the full pipeline:

- Data loading and preprocessing  
- Lagged feature engineering (e.g., using past 24 values)  
- Model training (e.g., gradient descent, ridge regression)  
- Evaluation using MSE, RMSE, and MAE  
- Visualizations such as:
  - Prediction vs. actual values  
  - Residual analysis  
  - ACF/PACF plots  
  - MAE by hour of day or weekday  
  - Model coefficient interpretation






## Results

The table below shows example test results from forecasting NO₂ (GT) levels using different models and lag depths.  
A lag depth of 24 yielded the best performance. Larger depths (e.g., 48) slightly increased test error, likely due to overfitting.

| Lag Depth | Model                         | MSE    | MAE    | RMSE   |
|-----------|-------------------------------|--------|--------|--------|
| 1         | Courselib LR (GD 0.001)       | 149.70 | 13.39  | 12.24  |
| 1         | Scikit-learn Ridge (α = 0.1)  | 149.71 | 13.39  | 12.24  |
| 2         | Courselib LR (GD 0.001)       | 137.37 | 13.01  | 11.72  |
| 2         | Scikit-learn Ridge (α = 0.1)  | 137.37 | 13.01  | 11.72  |
| 24        | Courselib LR (GD 0.001)       | 137.27 | 12.99  | 11.72  |
| 24        | Scikit-learn Ridge (α = 1.0)  | 137.42 | 13.01  | 11.72  |
| 24        | **Scikit-learn Ridge (α = 0.1)** | **137.25** | **12.99** | **11.72** |
| 48        | Courselib LR (GD 0.001)       | 140.03 | 12.98  | 11.83  |
| 48        | Scikit-learn Ridge (α = 0.1)  | 140.07 | 12.98  | 11.84  |

Overall, linear autoregressive models, especially ridge regression, performed well for short-term air quality prediction.  
The best-performing model (scikit-learn Ridge with α = 0.1) slightly outperformed both the course-provided gradient descent model and Ridge with α = 1.0.

*Example plots (predicted vs actual, error by hour, coefficient weights, etc.) can be found in [`notebooks/main.ipynb`](notebooks/main.ipynb).  
The notebook also contains additional analysis and visual reasoning used to guide model selection.*



## Contributing

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
