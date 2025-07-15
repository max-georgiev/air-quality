import sys
import os

project_parent_path = os.path.abspath(os.getcwd())
if project_parent_path not in sys.path:
    sys.path.insert(0, project_parent_path)

courselib_parent_path = os.path.abspath(os.path.join(os.getcwd(), "..", "AppliedML"))
if courselib_parent_path not in sys.path:
    sys.path.insert(0, courselib_parent_path)

import pandas as pd
import numpy as np
import math
import logging
from typing import Dict, List, Any

# Project-specific imports
from src.data.data_processor import AirQualityProcessor
from src.features.feature_engineer import LagFeatureEngineer
from src.models.train_model import ModelEvaluator
from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE

# Model imports
from courselib.models.linear_models import LinearRegression
from courselib.optimizers import GDOptimizer
from courselib.utils.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge


def evaluate_lag_depth_effect(
    lag_depths_to_test: List[int],
    model_configs: Dict[str, Dict[str, Any]],
    training_fraction: float = 0.8,
    target_pollutant: str = TARGET_POLLUTANT,
    start_date: str = START_DATE,
    end_date: str = END_DATE
    ) -> pd.DataFrame:
    
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

        evaluator = ModelEvaluator(df_lagged_features, training_data_fraction=training_fraction) # initialize evaluator
        
        for model_name, config in model_configs.items():
            model_class = config['model_class']
            is_courselib_model = config.get('is_courselib_model', False)
            init_params = config.get('init_params', {})
            fit_params = config.get('fit_params', {})

            if is_courselib_model:
                # Specific handling for courselib LinearRegression (needs w, b, optimizer)
                if model_class == LinearRegression:
                    optimizer_instance = config.get('optimizer')
                    model_instance = model_class(w=np.zeros(lag), b=0.0, optimizer=optimizer_instance)
            else:
                # For scikit-learn models (or other non-courselib models)
                model_instance = model_class(**init_params)

            evaluator.evaluate_model(
                    model_name,
                    model_instance, 
                    is_courselib_model=is_courselib_model,
                    **fit_params # Pass specific fit parameters for this model
                )

        
            model_metrics = evaluator.get_metrics_results().get(model_name, {})
            if model_metrics:
                row = {
                    'Lag Depth': lag,
                    'Model': model_name,
                    'MSE': model_metrics.get('mse'),
                    'MAE': model_metrics.get('mae'),
                    'RMSE': model_metrics.get('rmse')
                    }
            results_list.append(row)

    results_df = pd.DataFrame(results_list)

    return results_df


def evaluate_lag_depth_effect_linregr(
    lag_depths_to_test: List[int],
    #model_configs: Dict[str, Dict[str, Any]],
    training_fraction: float = 0.8,
    target_pollutant: str = TARGET_POLLUTANT,
    start_date: str = START_DATE,
    end_date: str = END_DATE
    ) -> pd.DataFrame:
    
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

        courselib_lr_model = LinearRegression(w=np.zeros(lag), b=0.0, optimizer=GDOptimizer()) # assign model

        evaluator = ModelEvaluator(df_lagged_features, training_data_fraction=training_fraction) # initialize evaluator
        model_name = "Linear Regression (courselib)"
        evaluator.evaluate_model(
            model_name,
            courselib_lr_model, 
            is_courselib_model=True,
            num_epochs=2000, 
            batch_size=32, 
            #compute_metrics=True, 
            #metrics_dict={
            #    "RMSE": lambda y_true, y_pred: math.sqrt(mean_squared_error(y_pred, y_true)),
            #    "MAE": mean_absolute_error
            #}
        )

        model_results = evaluator.get_results().get(model_name, {})
        if model_results:
            row = {
                'Lag Depth': lag,
                'Model': model_name,
                'MSE': model_results.get('mse'),
                'MAE': model_results.get('mae'),
                'RMSE': model_results.get('rmse')
                }
        results_list.append(row)

    results_df = pd.DataFrame(results_list)

    return results_df

    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("Lag Depth (k)")
    plt.ylabel("Test RMSE")
    plt.title("Lag Depth vs. Model Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Ensure your config variables are set correctly or passed
    from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE
    import matplotlib.pyplot as plt
    

    test_lags = [2, 3, 4, 24] # Example lags to test

    # Define a single model config for courselib LR (not strictly needed if hardcoded, but for consistency)
    model_configurations = {
        "Courselib LR (GD 0.0001)": {
            "model_class": LinearRegression,
            "is_courselib_model": True,
            "optimizer": GDOptimizer(learning_rate=0.0001), # Instantiate optimizer here
            "fit_params": {"num_epochs": 2000, "batch_size": 32}
        },
        "Courselib LR (GD 0.001)": { # Another Courserlib LR with different learning rate
            "model_class": LinearRegression,
            "is_courselib_model": True,
            "optimizer": GDOptimizer(learning_rate=0.001), # Instantiate optimizer here
            "fit_params": {"num_epochs": 2000, "batch_size": 32}
        },
        "Scikit-learn Ridge (Alpha 1.0)": {
            "model_class": Ridge,
            "is_courselib_model": False, # Important for sklearn models
            "init_params": {"alpha": 1.0} # Constructor parameters for sklearn Ridge
        },
        "Scikit-learn Ridge (Alpha 0.1)": {
            "model_class": Ridge,
            "is_courselib_model": False,
            "init_params": {"alpha": 0.1}
        }
    }

    # Call the evaluation function with multiple model configurations
    lag_evaluation_results = evaluate_lag_depth_effect(
        lag_depths_to_test=test_lags,
        model_configs=model_configurations, # Pass the model_configurations dictionary
        training_fraction=0.8
    )

    '''
    # Call the function (note: model_configs param removed from func signature in this simplified version)
    lag_evaluation_results = evaluate_lag_depth_effect(
        lag_depths_to_test=test_lags,
        # model_configs=model_config_for_lr, # Not needed if hardcoded inside func
        training_fraction=0.8
    )
    '''

    print("\n--- Lag Depth Evaluation Results (courselib Linear Regression) ---")
    print(lag_evaluation_results.to_string())

    plt.figure(figsize=(12, 7))
        # Plot RMSE for each model
    for model_name in lag_evaluation_results['Model'].unique():
        model_data = lag_evaluation_results[lag_evaluation_results['Model'] == model_name].dropna(subset=['RMSE'])
        if not model_data.empty:
            plt.plot(model_data['Lag Depth'], model_data['RMSE'], marker='o', label=model_name)
        
    plt.xlabel("Lag Depth (k)")
    plt.ylabel("Test RMSE")
    plt.title("Lag Depth vs. Model Performance (RMSE) for Different Models")
    plt.xticks(sorted(lag_evaluation_results['Lag Depth'].unique())) # Ensure all lag depths are shown
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside
    plt.grid(True)
    plt.tight_layout()
    plt.show()