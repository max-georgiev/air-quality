from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.features.feature_engineer import LagFeatureEngineer
from sklearn.metrics import mean_absolute_error



def evaluate_lag_depth_effect(series, model_cls, lag_values, test_size=100):
    """
    Evaluate how different lag depths affect RMSE.

    Parameters:
    -----------
    series (pd.Series): Time series of the target pollutant.
    model_cls (sklearn model): Model class (e.g. Ridge).
    lag_values (list[int]): List of k values to test.
    test_size (int): Number of test samples.

    Returns:
    --------
    dict[int, float]: Mapping of lag depth to RMSE.
    """
    results = {}

    for k in lag_values:
        try:
            fe = LagFeatureEngineer(k)
            X, y = fe.prepare_supervised_data(series)
            X_train, X_test = X[:-test_size], X[-test_size:]
            y_train, y_test = y[:-test_size], y[-test_size:]

            model = model_cls()
            model.fit(X_train, y_train)
            y_pred = pd.Series(model.predict(X_test), index=y_test.index)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            results[k] = rmse
        except Exception as e:
            print(f"Skipped lag={k} due to error: {e}")

    # Plot
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("Lag Depth (k)")
    plt.ylabel("Test RMSE")
    plt.title("Lag Depth vs. Model Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results



def plot_error_by_time_group(y_true, y_pred, group_by='hour'):
    """
    Plot MAE grouped by hour-of-day or weekday.

    Parameters:
    -----------
    y_true (pd.Series): True values with datetime index.
    y_pred (pd.Series): Predicted values with aligned datetime index.
    group_by (str): 'hour' or 'weekday'.
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