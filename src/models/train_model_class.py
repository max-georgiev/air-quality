import sys
import os

project_parent_path = os.path.abspath(os.getcwd())
if project_parent_path not in sys.path:
    sys.path.insert(0, project_parent_path)

courselib_parent_path = os.path.abspath(os.path.join(os.getcwd(), "..", "AppliedML"))
if courselib_parent_path not in sys.path:
    sys.path.insert(0, courselib_parent_path)

import numpy as np
import pandas as pd
from courselib.utils.metrics import mean_squared_error, mean_absolute_error
from courselib.utils.splits import train_test_split
from src.features.feature_engineer import LagFeatureEngineer
import logging
import math # For sqrt

# Assuming your custom_standard_scaler is available or part of a scaler class
def custom_standard_scaler(train_X, test_X):
    train_X_mean = np.mean(train_X, axis=0)
    train_X_std = np.std(train_X, axis=0)
    train_X_std[train_X_std == 0] = 1 # Avoid division by zero
    train_X_scaled = (train_X - train_X_mean) / train_X_std
    test_X_scaled = (test_X - train_X_mean) / train_X_std
    return train_X_scaled, test_X_scaled

class ModelEvaluator:
    def __init__(self, df):
        '''
        Initializes the ModelEvaluator

        Parameters:
        -----------
        df: pd.DataFrame
            A preprocessed dataframe containing the lagged feature matrix X and the target vector y
        '''

        self.df = df
        self.X = None
        self.Y = None
        self.train_X_scaled = None
        self.test_X_scaled = None
        self.train_Y = None
        self.test_Y = None
        self.naive_baseline_preds = None # To store the naive baseline

        self.models_results = {} # To store predictions and metrics for each trained model

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def train_model(self):
        '''
        Trains the model
        '''
        X, Y, train_X, train_Y, test_X, test_Y = train_test_split(
            self.df, 
            training_data_fraction=0.8, 
            class_column_name='Target', 
            shuffle=False, 
            return_numpy=True)

        lag_1 = test_X[:, 0].copy()


    def _prepare_data(self):
        """Internal method to load, process, feature engineer, and split data."""
        logging.info("Loading and processing data for evaluation...")
        air_quality_series = self.data_processor.get_target_time_series()
        self.df = self.feature_engineer.prepare_supervised_data(air_quality_series)
        logging.info(f"DataFrame shape after feature engineering: {self.df.shape}")

        # Capture the raw test_X before scaling for naive baseline
        X_raw, Y_raw, train_X_raw, train_Y, test_X_raw, test_Y = \
            train_test_split(self.df, self.training_fraction, 
                                class_column_name='Target', 
                                shuffle=False, return_numpy=True)
        
        # Capture the unscaled lag_1 for the naive baseline (assuming it's the first feature)
        # Note: If your train_test_split from courselib returns pd.DataFrame/Series,
        # you might need to adjust accessing the column, e.g., test_X_raw['lag_1'].values
        self.naive_baseline_preds = test_X_raw[:, 0].copy() 

        # Apply scaling
        self.train_X_scaled, self.test_X_scaled = custom_standard_scaler(train_X_raw, test_X_raw)
        self.train_Y = train_Y
        self.test_Y = test_Y

        logging.info(f"Training data shape: X={self.train_X_scaled.shape}, y={self.train_Y.shape}")
        logging.info(f"Testing data shape: X={self.test_X_scaled.shape}, y={self.test_Y.shape}")

    def evaluate_model(self, model_name, model_instance, is_courselib_model=False, **fit_params):
        """
        Trains and evaluates a single model.
        
        Parameters:
        - model_name (str): A unique name for this model (e.g., 'Courselib LR', 'Sklearn Ridge').
        - model_instance: An instantiated model object (e.g., LinearRegression(), Ridge(alpha=1.0)).
        - is_courselib_model (bool): True if it's a courselib model requiring specific fit params.
        - **fit_params: Additional parameters to pass to the model's fit method.
        """
        
        # Capture the raw test_X before scaling for naive baseline
        X_raw, Y_raw, train_X_raw, train_Y, test_X_raw, test_Y = \
            train_test_split(self.df, 0.8, 
                                class_column_name='Target', 
                                shuffle=False, return_numpy=True)
        
        # Capture the unscaled lag_1 for the naive baseline (assuming it's the first feature)
        # Note: If your train_test_split from courselib returns pd.DataFrame/Series,
        # you might need to adjust accessing the column, e.g., test_X_raw['lag_1'].values
        lag_1 = test_X_raw[:, 0].copy() 

        # Apply scaling
        train_X_scaled, test_X_scaled = custom_standard_scaler(train_X_raw, test_X_raw)
        train_Y = train_Y
        test_Y = test_Y

        logging.info(f"Training data shape: X={train_X_scaled.shape}, y={train_Y.shape}")
        logging.info(f"Testing data shape: X={test_X_scaled.shape}, y={test_Y.shape}")


        logging.info(f"Training {model_name} model...")
        
        if is_courselib_model:
            # Courserlib models might have different fit signatures (e.g., num_epochs, batch_size)
            model_instance.fit(train_X_scaled, train_Y, **fit_params)
            logging.info(f"{model_name} trained successfully.")

            logging.info(f"Evaluating {model_name} on the test set...")
            y_pred = model_instance.decision_function(test_X_scaled)
        else:
            model_instance.fit(train_X_scaled, train_Y)
            logging.info(f"{model_name} trained successfully.")

            logging.info(f"Evaluating {model_name} on the test set...")
            y_pred = model_instance.predict(test_X_scaled)
            
        logging.info(f"{model_name} trained successfully.")

        logging.info(f"Evaluating {model_name} on the test set...")
        

        # Store results
        self.models_results[model_name] = {
            "model_object": model_instance,
            "y_pred": y_pred,
            "mse": mean_squared_error(y_pred, test_Y),
            "mae": mean_absolute_error(y_pred, test_Y),
            "rmse": math.sqrt(mean_squared_error(y_pred, test_Y))
        }
        logging.info(f"{model_name} - Test MSE: {self.models_results[model_name]['mse']:.4f}, "
                     f"Test MAE: {self.models_results[model_name]['mae']:.4f}, "
                     f"Test RMSE: {self.models_results[model_name]['rmse']:.4f}")

    def evaluate_naive_baseline(self):
        """Calculates and stores metrics for the Naive Baseline (previous timepoint)."""
        
        X_raw, Y_raw, train_X_raw, train_Y, test_X_raw, test_Y = \
            train_test_split(self.df, 0.8, 
                                class_column_name='Target', 
                                shuffle=False, return_numpy=True)
        
        # Capture the unscaled lag_1 for the naive baseline (assuming it's the first feature)
        # Note: If your train_test_split from courselib returns pd.DataFrame/Series,
        # you might need to adjust accessing the column, e.g., test_X_raw['lag_1'].values
        naive_baseline_preds = test_X_raw[:, 0].copy() 

        # Apply scaling
        train_X_scaled, test_X_scaled = custom_standard_scaler(train_X_raw, test_X_raw)
        train_Y = train_Y
        test_Y = test_Y

        logging.info("Calculating Naive Baseline...")
        
        # Ensure that self.test_Y and self.naive_baseline_preds have compatible indices/lengths
        # when dealing with pandas Series, this is handled well. With numpy, ensure lengths match.
        if len(naive_baseline_preds) != len(test_Y):
             raise ValueError("Naive baseline predictions and test_Y have mismatched lengths.")

        mse_naive = mean_squared_error(naive_baseline_preds, test_Y)
        mae_naive = mean_absolute_error(naive_baseline_preds, test_Y)
        rmse_naive = math.sqrt(mse_naive)

        self.models_results["Naive Baseline"] = {
            "y_pred": naive_baseline_preds,
            "mse": mse_naive,
            "mae": mae_naive,
            "rmse": rmse_naive
        }
        logging.info(f"Naive Baseline - Test MSE: {mse_naive:.4f}, "
                     f"Test MAE: {mae_naive:.4f}, Test RMSE: {rmse_naive:.4f}")

    def get_results(self):
        """Returns the dictionary of all model results."""
        return self.models_results

    def get_metric_for_model(self, model_name, metric_name):
        """Returns a specific metric for a specific model."""
        return self.models_results.get(model_name, {}).get(metric_name)
    
'''
    def get_test_data(self):
        """Returns the true test values."""
        return self.test_Y

    def get_predictions_for_model(self, model_name):
        """Returns predictions for a specific model."""
        return self.models_results.get(model_name, {}).get("y_pred")
'''

        

# =============================================================================
# Updated Main Execution Block (if __name__ == "__main__":)
# =============================================================================
if __name__ == "__main__":
    from src.data.data_processor import AirQualityProcessor
    from src.features.feature_engineer import LagFeatureEngineer
    from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH
    from courselib.models.linear_models import LinearRegression
    from courselib.optimizers import GDOptimizer
    from sklearn.linear_model import LinearRegression as LR_SciKit
    from sklearn.linear_model import Ridge

    # Initialize your data and feature engineering components
    processor = AirQualityProcessor(
        target_pollutant=TARGET_POLLUTANT,
        start_date=START_DATE,
        end_date=END_DATE
    )

    time_series = processor.get_target_time_series()

    feature_engineer = LagFeatureEngineer(lag_depth=LAG_DEPTH) # Using LAG_DEPTH from config

    df_lagged = feature_engineer.prepare_supervised_data(time_series)

    # Run evaluations for each model
    # Note: prepare_data will be called automatically on first evaluate_model/evaluate_naive_baseline call
    
    optimizer = GDOptimizer()

    courselib_lr_model = LinearRegression(w=np.zeros(24), b=0.0, optimizer=optimizer)
    
    evaluator = ModelEvaluator(df_lagged)
    
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
    

    # Scikit-learn Linear Regression
    sklearn_lr_model = LR_SciKit()
    evaluator.evaluate_model("Scikit-learn Linear Regression", sklearn_lr_model)

    # Scikit-learn Ridge Regression
    ridge_model = Ridge(alpha=1.0) # Remember to tune alpha!
    evaluator.evaluate_model("Scikit-learn Ridge Regression", ridge_model)

    # Naive Baseline
    evaluator.evaluate_naive_baseline()

    # Accessing results
    all_results = evaluator.get_results()
    print("\n--- Summary of All Model Results ---")
    for model_name, metrics in all_results.items():
        print(f"{model_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    # Example of using results with plot_error_by_time_group from analysis.py
    from src.visualization.analysis import plot_error_by_time_group
    
    # You need the original dataframe's index for y_test and y_pred
    # This means retrieving the original y_test and making sure predictions have its index.
    # The original y_test from train_test_split (before being converted to np.array) would have the index.
    # If test_Y is already numpy array in evaluator, you might need to reconstruct a pd.Series with index.
    # Assuming the original 'Y' from data_processor/feature_engineer had datetime index,
    # and the split preserves order, you can infer index for test_Y.
    
    # For plotting, it's essential that y_true and y_pred are pandas Series with DatetimeIndex.
    # Let's adjust how test_Y is stored if it comes as numpy in _prepare_data
    # If Y_raw from train_test_split is a pd.Series:
    # evaluator.test_Y_series = Y_raw[-len(test_Y):] # Capture the Series with index
    
    # Assuming test_Y has correct length and we can get the index:
    # This part requires making sure the datetime index is available.
    # The `train_test_split` should ideally return pandas Series/DataFrames.
    # Currently your train_test_split has `return_numpy=True` which drops index.
    # You'd need to modify `train_test_split` to return pandas Series/DataFrames
    # or pass the original `df.index` down.
    
    # For demonstration, let's assume we can get the original test_Y with its datetime index.
    # If your original `Y` (from `df` before split) was a pandas Series with DatetimeIndex,
    # then `Y_raw` would be that series, and `test_Y_series = Y_raw.iloc[-len(evaluator.test_Y):]`
    # would retrieve the correct indexed series.
    # For now, I'll use a placeholder `get_original_test_Y_with_index()`
    
    # Placeholder: In a real scenario, ensure your train_test_split preserves DatetimeIndex
    # Or store the original df's index in the evaluator.
    original_df_index_for_test = evaluator.df.index[-len(evaluator.test_Y):] # Approximate if index lost
    
    # Convert numpy arrays back to pandas Series with the correct DatetimeIndex for plotting
    y_test_plot = pd.Series(evaluator.test_Y.flatten(), index=original_df_index_for_test) # .flatten() in case of (N,1)
    
    # Plot for Courserlib LR
    y_pred_courselib_plot = pd.Series(evaluator.get_predictions_for_model("Linear Regression (courselib)").flatten(), index=original_df_index_for_test)
    plot_error_by_time_group(y_test_plot, y_pred_courselib_plot, group_by='hour')
    
    # Plot for Scikit-learn Ridge
    y_pred_ridge_plot = pd.Series(evaluator.get_predictions_for_model("Scikit-learn Ridge Regression").flatten(), index=original_df_index_for_test)
    plot_error_by_time_group(y_test_plot, y_pred_ridge_plot, group_by='weekday')