import sys
import os
import pandas as pd
import logging

project_parent_path = os.path.abspath(os.getcwd())
if project_parent_path not in sys.path:
    sys.path.insert(0, project_parent_path)

courselib_parent_path = os.path.abspath(os.path.join(os.getcwd(), "..", "AppliedML"))
if courselib_parent_path not in sys.path:
    sys.path.insert(0, courselib_parent_path)

# External libraries
from courselib.utils.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set up logging for this module
logger = logging.getLogger(__name__)

def plot_acf_pacf(series: pd.Series, lags: int = 48):
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    of a time series.

    Parameters:
    -----------
    series : pd.Series
        Target time series (must be stationary if using PACF seriously).
    lags : int
        Number of lags to display.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation (ACF)")
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.suptitle("Autocorrelation and Partial Autocorrelation")
    plt.tight_layout()
    plt.show()
    plt.close() # Close plot to free memory

def plot_error_by_time_group(y_true, y_pred, group_by='hour'):
    """
    Plot MAE grouped by hour-of-day or weekday.

    Parameters:
    -----------
    y_true : pd.Series
        True values with datetime index.
    y_pred : pd.Series 
        Predicted values with aligned datetime index.
    group_by : str
        'hour' or 'weekday'.
    """
    if not isinstance(y_true.index, pd.DatetimeIndex):
        raise ValueError("y_true must have a DatetimeIndex.")
    if not isinstance(y_pred.index, pd.DatetimeIndex):
        raise ValueError("y_pred must have a DatetimeIndex.")
    if group_by not in ['hour', 'weekday']:
        raise ValueError("group_by must be either 'hour' or 'weekday'.")

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df.dropna(inplace=True)

    if group_by == 'hour':
        df['group'] = df.index.hour
        group_range = range(24)
        xlabel = 'Hour'
        title = 'Mean Absolute Error by Hour of Day'
    else:
        df['group'] = df.index.weekday
        group_range = range(7)
        xlabel = 'Weekday'
        title = 'Mean Absolute Error by Weekday'

    # Group and calculate MAE
    mae_by_group = df.groupby('group').apply(lambda g: mean_absolute_error(g['y_true'], g['y_pred']))

    # Ensure all expected groups are present (fill missing with 0)
    mae_by_group = mae_by_group.reindex(group_range, fill_value=0)

    if group_by == 'weekday':
        mae_by_group.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Plotting
    plt.figure(figsize=(10, 5))
    mae_by_group.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    plt.close() # Close plot to free memory