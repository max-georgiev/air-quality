import sys
import os
import pandas as pd
import numpy as np
import math
import logging
from typing import Dict, List, Any

project_parent_path = os.path.abspath(os.getcwd())
if project_parent_path not in sys.path:
    sys.path.insert(0, project_parent_path)

courselib_parent_path = os.path.abspath(os.path.join(os.getcwd(), "..", "AppliedML"))
if courselib_parent_path not in sys.path:
    sys.path.insert(0, courselib_parent_path)

# Project-specific imports
from src.data.data_processor import AirQualityProcessor
from src.features.feature_engineer import LagFeatureEngineer
from src.models.train_model import ModelEvaluator
from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE

# Model imports
from courselib.models.linear_models import LinearRegression
from courselib.optimizers import GDOptimizer
from sklearn.linear_model import Ridge

# Set up logging for this module
logger = logging.getLogger(__name__)

def evaluate_lag_depth_effect(
    lag_depths_to_test: List[int],
    model_configs: Dict[str, Dict[str, Any]],
    training_fraction: float = 0.8,
    target_pollutant: str = TARGET_POLLUTANT,
    start_date: str = START_DATE,
    end_date: str = END_DATE
    ) -> pd.DataFrame:
    """
    Evaluates the performance of multiple models across various lag depths.

    Parameters:
    -----------
    lag_depths_to_test : List[int]
        A list of integer lag depths to test (e.g., [1, 2, 24]).
    model_configs : Dict[str, Dict[str, Any]]
        A dictionary where keys are model names and values are dictionaries
        containing model configuration (model_class, is_courselib_model, etc.)
        as expected by ModelEvaluator.evaluate_model.
    training_fraction : float, optional
        The fraction of data to use for training the models. Defaults to 0.8.
    target_pollutant : str, optional
        The target pollutant to process. Defaults to TARGET_POLLUTANT from config.
    start_date : str, optional
        Start date for data processing. Defaults to START_DATE from config.
    end_date : str, optional
        End date for data processing. Defaults to END_DATE from config.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing 'Lag Depth', 'Model', 'MSE', 'MAE', and 'RMSE' for each
        model and lag depth tested.
    """
    logger.info(f"Starting lag depth effect evaluation for depths: {lag_depths_to_test}")
    
    results_list = []    # to store results

    processor = AirQualityProcessor(
        target_pollutant=target_pollutant,
        start_date=start_date,
        end_date=end_date
    )

    base_time_series = processor.get_target_time_series()


    for lag in lag_depths_to_test:
        feature_engineer = LagFeatureEngineer(lag_depth=lag) # initialize engineer
        df_lagged_features  = feature_engineer.prepare_supervised_data(base_time_series, return_separate=False) # get lagged matrix and vector

        # Initialize evaluator for the current lag depth.
        evaluator = ModelEvaluator(df_lagged_features, training_data_fraction=training_fraction) # initialize evaluator
        
        for model_name, config in model_configs.items():
            evaluator.evaluate_model(model_name, config)
        
            model_metrics = evaluator.get_metrics_results().get(model_name, {})
            
            if model_metrics:
                # Append both training and test metrics
                results_list.append({
                    'Lag Depth': lag,
                    'Model': model_name,
                    'Metric Type': 'Train',
                    'MSE': model_metrics.get('train_mse'),
                    'MAE': model_metrics.get('train_mae'),
                    'RMSE': model_metrics.get('train_rmse')
                })
                results_list.append({
                    'Lag Depth': lag,
                    'Model': model_name,
                    'Metric Type': 'Test',
                    'MSE': model_metrics.get('mse'),
                    'MAE': model_metrics.get('mae'),
                    'RMSE': model_metrics.get('rmse')
                })
                logger.info(f"    - {model_name} metrics recorded (Train RMSE: {model_metrics.get('train_rmse', np.nan):.4f}, Test RMSE: {model_metrics.get('rmse', np.nan):.4f}).")
            else:
                logger.warning(f"    - {model_name} failed or produced no metrics for lag depth {lag}.")
    
    results_df = pd.DataFrame(results_list)

    return results_df

if __name__ == "__main__":
    # Ensure logging is configured for standalone script execution
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Import matplotlib here, as it's only used in the __main__ plotting section
    from src.visualization.plotting import plot_lag_depth_results

    logger.info("--- Starting Model Selection Script for Lag Depth Analysis ---")

    # Define the lag depths to test
    test_lags = [1, 2] # A more comprehensive set of example lags

    # Define all model configurations to be tested
    model_configurations = {
        "Courselib LR (GD 0.0001)": {
            "model_class": LinearRegression,
            "is_courselib_model": True,
            "optimizer": GDOptimizer(learning_rate=0.0001), # Instantiate optimizer here
            "fit_params": {"num_epochs": 2000, "batch_size": 32}
        },
        "Courselib LR (GD 0.001)": { # Another Courselib LR with different learning rate
            "model_class": LinearRegression,
            "is_courselib_model": True,
            "optimizer": GDOptimizer(learning_rate=0.001),
            "fit_params": {"num_epochs": 2000, "batch_size": 32}
        },
        "Scikit-learn Ridge (Alpha 1.0)": {
            "model_class": Ridge,
            "is_courselib_model": False,
            "init_params": {"alpha": 1.0}
        },
        "Scikit-learn Ridge (Alpha 0.1)": {
            "model_class": Ridge,
            "is_courselib_model": False,
            "init_params": {"alpha": 0.1}
        }
    }

    # Call the evaluation function
    lag_evaluation_results = evaluate_lag_depth_effect(
        lag_depths_to_test=test_lags,
        model_configs=model_configurations,
        training_fraction=0.8
    )

    if not lag_evaluation_results.empty:
        print("\n--- Lag Depth Evaluation Results ---")
        print(lag_evaluation_results.to_string())

        # Call the new plotting function without the save_path argument
        plot_lag_depth_results(lag_evaluation_results)
    else:
        logger.warning("No lag depth evaluation results to display.")

    logger.info("--- Model Selection Script Finished ---")