import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np


def plot_time_series(series: pd.Series, title: str = "Target Time Series"):
    """
    Plots the time series of the target pollutant.

    Parameters:
    -----------
    series (pd.Series): Time-indexed series of pollutant values.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(series, label=series.name)
    plt.title(title)
    plt.xlabel("DateTime")
    plt.ylabel(f"{series.name} (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_acf_pacf(series: pd.Series, lags: int = 48):
    """
    Plots ACF and PACF side by side.

    Parameters:
    -----------
    series (pd.Series): Target time series (must be stationary if using PACF seriously).
    lags (int): Number of lags to display.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation (ACF)")
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true: pd.Series, y_pred: pd.Series, title: str = "Predicted vs Actual"):
    """
    Plots predicted vs. actual values over time.

    Parameters:
    -----------
    y_true (pd.Series): Actual values.
    y_pred (pd.Series): Predicted values (must be aligned).
    title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle='--')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: pd.Series, y_pred: pd.Series, title: str = "Prediction Error (Residuals)"):
    """
    Plots residuals (errors between prediction and truth).

    Parameters:
    -----------
    y_true (pd.Series): Actual values.
    y_pred (pd.Series): Predicted values.
    title (str): Title of the plot.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 4))
    plt.plot(residuals, color='tomato', label='Residuals')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_coefficients(coefficients, feature_names=None, title="Lag Coefficients"):
    """
    Plots linear model coefficients.

    Parameters:
    -----------
    coefficients (np.ndarray): Coefficients from linear model.
    feature_names (list[str]): Optional list of feature names for x-axis.
    title (str): Plot title.
    """
    plt.figure(figsize=(10, 4))

    if feature_names:
        plt.bar(feature_names, coefficients)
        plt.xticks(rotation=90)
    else:
        plt.bar(range(len(coefficients)), coefficients)
        plt.xlabel("Feature Index")

    plt.ylabel("Coefficient Value")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()