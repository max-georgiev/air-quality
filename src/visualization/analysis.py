import sys
import os

project_parent_path = os.path.abspath(os.getcwd())
if project_parent_path not in sys.path:
    sys.path.insert(0, project_parent_path)

courselib_parent_path = os.path.abspath(os.path.join(os.getcwd(), "..", "AppliedML"))
if courselib_parent_path not in sys.path:
    sys.path.insert(0, courselib_parent_path)


from courselib.utils.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from src.features.feature_engineer import LagFeatureEngineer
from src.data.data_processor import AirQualityProcessor
from src.models.train_model_class import ModelEvaluator
from src.models import train_model
from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH
from courselib.models.linear_models import LinearRegression
from courselib.optimizers import GDOptimizer

# initialize processor
processor = AirQualityProcessor(
        target_pollutant=TARGET_POLLUTANT,
        start_date=START_DATE,
        end_date=END_DATE
    )

# get time series data
time_series = processor.get_target_time_series()

# assign optimizer
optimizer = GDOptimizer()

# lag values for testing
lag_values = (1, 2, 3, 6, 9, 12, 24, 36, 48)
def evaluate_lag_depth_effect_mine(lag_values):
    results = {}    # to store results
    for lag in lag_values:
        feature_engineer = LagFeatureEngineer(lag_depth=lag) # initialize engineer
        df_lagged = feature_engineer.prepare_supervised_data(time_series) # get lagged matrix and vector

        courselib_lr_model = LinearRegression(w=np.zeros(lag), b=0.0, optimizer=optimizer) # assign model

        evaluator = ModelEvaluator(df_lagged) # initialize evaluator

        evaluator.evaluate_model(
            "Linear Regression (courselib)", 
            courselib_lr_model, 
            is_courselib_model=True,
            num_epochs=2000, 
            batch_size=32, 
            compute_metrics=True, 
            metrics_dict={
                "RMSE": lambda y_true, y_pred: math.sqrt(mean_squared_error(y_pred, y_true)),
                "MAE": mean_absolute_error
            }
        )

        rmse = evaluator.get_metric_for_model("Linear Regression (courselib)", "rmse")

        results[lag] = rmse
    
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("Lag Depth (k)")
    plt.ylabel("Test RMSE")
    plt.title("Lag Depth vs. Model Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results


evaluate_lag_depth_effect_mine(lag_values)




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