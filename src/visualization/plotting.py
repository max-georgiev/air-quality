import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

# Set up logging for this module
logger = logging.getLogger(__name__)

def plot_time_series(series: pd.Series, title: str = "Target Time Series", ylabel: str = None):
    """
    Plots the time series of the target pollutant.

    Parameters:
    -----------
    series : pd.Series
        Time-indexed series of pollutant values.
    title : str
        Title of the plot.
    ylabel : str
        Label for the Y-axis. Defaults to series.name (µg/m³).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        logger.warning("Series does not have a DatetimeIndex. Plotting may not display time correctly.")

    plt.figure(figsize=(12, 4))
    plt.plot(series, label=series.name if series.name else "Series")
    plt.title(title)
    plt.xlabel("DateTime")
    plt.ylabel(ylabel if ylabel else f"{series.name} (µg/m³)" if series.name else "Value")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.close() # Close plot to free memory

def plot_residuals(y_true: pd.Series, y_pred: pd.Series, title: str = "Prediction Error (Residuals)"):
    """
    Plots residuals (errors between prediction and truth) over time.

    Parameters:
    -----------
    y_true : pd.Series
        Actual values
    y_pred : pd.Series
        Predicted values.
    title : str
        Title of the plot.
    """
    if not y_true.index.equals(y_pred.index):
        logger.warning("Indices of y_true and y_pred do not match. Attempting to align for residuals.")
        common_index = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_index]
        y_pred = y_pred.loc[common_index]
        if common_index.empty:
            logger.error("No common index found after aligning for residuals. Cannot plot.")
            return

    residuals = y_true - y_pred
    plt.figure(figsize=(12, 4))
    plt.plot(residuals, color='tomato', label='Residuals')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close() # Close plot to free memory

def plot_coefficients(coefficients: np.ndarray, feature_names = None, title: str = "Lag Coefficients"):
    """
    Plots linear model coefficients. If feature_names are not provided, it assumes
    they correspond to 'Lag 1', 'Lag 2', etc.

    Parameters:
    -----------
    coefficients : np.ndarray
        Coefficients from linear model
    feature_names : list[str]
        List of feature names for x-axis
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 4))

    if feature_names is None:
        # Assume coefficients correspond to 'Lag 1', 'Lag 2', ...
        feature_names = [f"Lag {i+1}" for i in range(len(coefficients))]
        logger.info(f"Generated default feature names: {feature_names}")
    
    if len(coefficients) != len(feature_names):
        logger.warning("Number of coefficients does not match number of feature names. Adjusting names or indices.")
        # Fallback to generic indices if mismatch
        plt.bar(range(len(coefficients)), coefficients)
        plt.xlabel("Feature Index")
    else:
        plt.bar(feature_names, coefficients)
        plt.xticks(rotation=90)
        plt.xlabel("Feature Name")


    plt.ylabel("Coefficient Value")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close() # Close plot to free memory

def plot_predictions_vs_actual(y_true_series: pd.Series, y_pred_series: pd.Series, model_name: str = "Model"):
    """
    Plots the actual vs. predicted values over time for visual comparison.
    Handles index alignment to ensure correct plotting.

    Parameters:
    - y_true_series : pd.Series
        Series of actual values (must have DatetimeIndex)
    - y_pred_series : pd.Series
        Series of predicted values (must have DatetimeIndex).
    - model_name : str
        Name of the model for the plot title.
    """
    if not isinstance(y_true_series, pd.Series) or not isinstance(y_pred_series, pd.Series):
        logger.error("Input y_true_series and y_pred_series must be pandas Series.")
        return

    if y_true_series.empty or y_pred_series.empty:
        logger.warning("One or both input series are empty. Cannot plot predictions vs actual.")
        return

    if not isinstance(y_true_series.index, pd.DatetimeIndex) or not isinstance(y_pred_series.index, pd.DatetimeIndex):
        logger.error("Both actual and predicted series must have a DatetimeIndex. Cannot plot.")
        return

    if not y_true_series.index.equals(y_pred_series.index):
        logger.warning("Indices of actual and predicted series do not match. Attempting to align.")
        common_index = y_true_series.index.intersection(y_pred_series.index)
        if common_index.empty:
            logger.error("No common index found after aligning actual and predicted series. Cannot plot.")
            return
        y_true_series = y_true_series.loc[common_index]
        y_pred_series = y_pred_series.loc[common_index]
        logger.info(f"Aligned series to common index of length: {len(common_index)}")

    plt.figure(figsize=(14, 7))
    plt.plot(y_true_series.index, y_true_series, label='Actual Values', color='blue', alpha=0.8)
    plt.plot(y_pred_series.index, y_pred_series, label=f'{model_name} Predictions', color='red', linestyle='--')
    plt.title(f'Actual vs. Predicted Values for {model_name} Over Test Period')
    plt.xlabel('Date')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close() # Close plot to free memory
    logger.info(f"Plot: Actual vs. Predicted values for {model_name} generated.")


def plot_error_by_time_group(y_true: pd.Series, y_pred: pd.Series, group_by: str = 'hour', model_name: str = "Model"):
    """
    Plots the mean absolute error (MAE) grouped by a specific time component (e.g., hour, day of week).

    Parameters:
    - y_true : pd.Series
        Actual values (must have DatetimeIndex)
    - y_pred : pd.Series
        Predicted values (must have DatetimeIndex)
    - group_by : str
        Time component to group by ('hour', 'dayofweek', 'dayofyear', 'month', 'year').
                      Default is 'hour'.
    - model_name : str
        Name of the model for the plot title.
    """
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        logger.error("Input y_true and y_pred must be pandas Series.")
        return

    if y_true.empty or y_pred.empty:
        logger.warning("One or both input series are empty. Cannot plot error by time group.")
        return

    if not isinstance(y_true.index, pd.DatetimeIndex) or not isinstance(y_pred.index, pd.DatetimeIndex):
        logger.error("Both actual and predicted series must have a DatetimeIndex. Cannot plot error by time group.")
        return

    if not y_true.index.equals(y_pred.index):
        logger.warning("Indices of actual and predicted series do not match. Attempting to align for error analysis.")
        common_index = y_true.index.intersection(y_pred.index)
        if common_index.empty:
            logger.error("No common index found after aligning for error analysis. Cannot plot.")
            return
        y_true = y_true.loc[common_index]
        y_pred = y_pred.loc[common_index]
        logger.info(f"Aligned series to common index of length: {len(common_index)}")

    errors = np.abs(y_true - y_pred) # Using Absolute Error for mean
    errors_df = pd.DataFrame({'error': errors, 'time_group': getattr(errors.index, group_by)})

    # Map dayofweek to names for better readability
    if group_by == 'dayofweek':
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        errors_df['time_group'] = errors_df['time_group'].map(lambda x: day_names[x])
        # Reorder categories for plotting
        errors_df['time_group'] = pd.Categorical(errors_df['time_group'], categories=day_names, ordered=True)
    elif group_by == 'month':
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        errors_df['time_group'] = errors_df['time_group'].map(lambda x: month_names[x-1])
        errors_df['time_group'] = pd.Categorical(errors_df['time_group'], categories=month_names, ordered=True)

    mean_errors = errors_df.groupby('time_group')['error'].mean()

    plt.figure(figsize=(10, 6))
    mean_errors.plot(kind='bar', color='skyblue')
    plt.title(f'Mean Absolute Error for {model_name} by {group_by.replace("dayofweek", "Day of Week").capitalize()}')
    plt.xlabel(group_by.replace("dayofweek", "Day of Week").capitalize())
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close() # Close plot to free memory
    logger.info(f"Plot: Mean Absolute Error by {group_by} for {model_name} generated.")


def plot_lag_depth_results(df_results: pd.DataFrame):
    """
    Generates and displays plots of RMSE and MAE vs. Lag Depth for different models,
    stacked one below the other on a single figure using Matplotlib.

    Parameters:
    -----------
    df_results : pd.DataFrame
        A DataFrame containing evaluation results with columns:
        'Lag Depth', 'Model', 'Metric Type', 'RMSE', 'MAE'.
    """
    if df_results.empty:
        logger.warning("No data provided for plotting lag depth results. Skipping plots.")
        return

    unique_lag_depths = sorted(df_results['Lag Depth'].unique())
    unique_models = df_results['Model'].unique()
    # unique_metric_types = df_results['Metric Type'].unique() # Not directly used for iteration here

    # Define a color palette manually if not using Seaborn's default
    colors = plt.cm.get_cmap('tab10', len(unique_models)) 
    
    # Create a single figure with two subplots, one above the other
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True) # sharex ensures x-axis limits are the same

    # --- Plotting RMSE on the top subplot ---
    ax0 = axes[0]
    for i, model_name in enumerate(unique_models):
        model_data = df_results[df_results['Model'] == model_name]
        
        # Plot Train RMSE
        train_rmse_data = model_data[model_data['Metric Type'] == 'Train'].dropna(subset=['RMSE'])
        if not train_rmse_data.empty:
            ax0.plot(train_rmse_data['Lag Depth'], train_rmse_data['RMSE'], 
                     marker='o', linestyle='--', color=colors(i), 
                     label=f'{model_name} (Train)')
        
        # Plot Test RMSE
        test_rmse_data = model_data[model_data['Metric Type'] == 'Test'].dropna(subset=['RMSE'])
        if not test_rmse_data.empty:
            ax0.plot(test_rmse_data['Lag Depth'], test_rmse_data['RMSE'], 
                     marker='o', linestyle='-', color=colors(i), 
                     label=f'{model_name} (Test)')

    ax0.set_title('Lag Depth vs. Model Performance (RMSE)')
    ax0.set_ylabel('RMSE')
    ax0.grid(True, linestyle='--', alpha=0.7)
    # Combine all unique labels for the legend
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels, title='Model & Metric Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- Plotting MAE on the bottom subplot ---
    ax1 = axes[1]
    for i, model_name in enumerate(unique_models):
        model_data = df_results[df_results['Model'] == model_name]
        
        # Plot Train MAE
        train_mae_data = model_data[model_data['Metric Type'] == 'Train'].dropna(subset=['MAE'])
        if not train_mae_data.empty:
            ax1.plot(train_mae_data['Lag Depth'], train_mae_data['MAE'], 
                     marker='o', linestyle='--', color=colors(i), 
                     label=f'{model_name} (Train)')
        
        # Plot Test MAE
        test_mae_data = model_data[model_data['Metric Type'] == 'Test'].dropna(subset=['MAE'])
        if not test_mae_data.empty:
            ax1.plot(test_mae_data['Lag Depth'], test_mae_data['MAE'], 
                     marker='o', linestyle='-', color=colors(i), 
                     label=f'{model_name} (Test)')

    ax1.set_title('Lag Depth vs. Model Performance (MAE)')
    ax1.set_xlabel('Lag Depth (k)')
    ax1.set_ylabel('MAE')
    ax1.set_xticks(unique_lag_depths)
    ax1.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for the legend
    
    plt.show() # Only show, no saving

    logger.info("Lag depth analysis plots generated.")