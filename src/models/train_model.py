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
from typing import Dict, Callable, Tuple, Union, Any

# Set up logging for this module
logger = logging.getLogger(__name__)

def custom_standard_scaler(train_data: np.ndarray, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies standard scaling (mean 0, std 1) to features based on training data statistics.
    Avoids data leakage by using only training data's mean and std for both sets.

    Parameters:
    -----------
    train_data : np.ndarray
        Training feature data.
    test_data : np.ndarray
        Testing feature data.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Scaled training and testing feature data.
    """
    if train_data.ndim == 1: # Handle 1D arrays (e.g., single feature)
        train_data = train_data.reshape(-1, 1)
        test_data = test_data.reshape(-1, 1)

    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)

    # Avoid division by zero for features with zero standard deviation
    # Such features are constant and will remain constant (0 after scaling relative to their mean)
    train_std[train_std == 0] = 1

    train_scaled = (train_data - train_mean) / train_std
    test_scaled = (test_data - train_mean) / train_std
    
    return train_scaled, test_scaled

class ModelEvaluator:
    def __init__(self, df, training_data_fraction=0.8):
        '''
        Initializes the ModelEvaluator by performing train-test split and scaling.

        Parameters:
        -----------
        df : pd.DataFrame
            A preprocessed dataframe containing the lagged feature matrix X and the target vector y.
            It must have a 'Target' column and a DatetimeIndex.
        training_data_fraction : float
            The fraction of the data to use for training (e.g., 0.8 for 80%).
        '''

        if 'Target' not in df.columns:
            logger.error("Input DataFrame must contain a 'Target' column.")
            raise ValueError("Input DataFrame must contain a 'Target' column.")
        
        if not (0 < training_data_fraction < 1):
            logger.error(f"Invalid training_fraction: {training_data_fraction}. Must be between 0 and 1.")
            raise ValueError("training_fraction must be between 0 and 1.")
        
        self.df_original = df.copy() # Store a copy of the full preprocessed DF
        self.training_data_fraction = training_data_fraction

        # MODELS RESULTS INITIALIZATION

        #self.models_results: Dict[str, Dict[str, Union[float, np.ndarray]]] = {}

        self.metrics_results: Dict[str, Dict[str, Union[float, str]]] = {} # Stores only metrics and error strings
        self.predictions_results: Dict[str, pd.Series] = {} # Stores pd.Series of predictions
        self.trained_models: Dict[str, Any] = {} # Stores the actual trained model objects

        logger.info("Performing data split and scaling...")

        _, train_df, test_df = train_test_split(
            self.df_original, 
            training_data_fraction=self.training_data_fraction, 
            class_column_name='Target', 
            shuffle=False, 
            return_numpy=False
        )

        # Extract X and Y (as pandas DataFrames/Series)
        # Drop the 'Target' column from features
        self.train_X_raw = train_df.drop(columns=['Target'])
        self.train_Y = train_df['Target']
        self.test_X_raw = test_df.drop(columns=['Target'])
        self.test_Y = test_df['Target']         # Keep as Series to preserve index for plotting
        self.test_Y_index = self.test_Y.index   # Store the index for later use in plotting

        # Convert to numpy arrays for scaling and model training
        train_X_np = self.train_X_raw.to_numpy()
        test_X_np = self.test_X_raw.to_numpy()
        train_Y_np = self.train_Y.to_numpy()
        test_Y_np = self.test_Y.to_numpy() # This is the array for metrics calculation

        # Apply scaling using the custom_standard_scaler
        self.train_X_scaled, self.test_X_scaled = custom_standard_scaler(train_X_np, test_X_np)
        
        # Store the numpy versions of Y for consistent use with scaled X in models
        self.train_Y_np = train_Y_np
        self.test_Y_np = test_Y_np

        # Calculate naive baseline predictions (from the unscaled test_X_raw, specifically 'lag_1')
        # Assuming 'lag_1' is the first column after dropping 'Target'
        # Or, more robustly, access it by column name:
        if 'lag_1' not in self.test_X_raw.columns:
            logger.warning("No 'lag_1' column found for naive baseline. Using the first feature column.")
            # Fallback to second column if 'lag_1' is not present by name (the first is the 'Target')
            self.naive_baseline_preds_unscaled = self.test_X_raw.iloc[:, 1].to_numpy()
        else:
            self.naive_baseline_preds_unscaled = self.test_X_raw['lag_1'].to_numpy()

        logger.info(f"Training data shape: X={self.train_X_scaled.shape}, y={self.train_Y_np.shape}")
        logger.info(f"Testing data shape: X={self.test_X_scaled.shape}, y={self.test_Y_np.shape}")
        logger.info(f"Test Y index shape: {len(self.test_Y_index)}")


    def _fit_model(self, model_instance, is_courselib_model: bool, **fit_params):
        """
        Internal method to fit a given model instance to the training data.
        """
        logger.debug(f"Fitting model type: {type(model_instance).__name__}")
        train_X_use = self.train_X_scaled
        train_Y_use = self.train_Y_np

        if is_courselib_model:
            model_instance.fit(train_X_use, train_Y_use, **fit_params)
        else:
            model_instance.fit(train_X_use, train_Y_use)
        
        logger.info(f"Model {type(model_instance).__name__} trained successfully.")
        
        return model_instance

    def _predict_model(self, fitted_model, is_courselib_model: bool) -> np.ndarray:
        """
        Internal method to generate predictions on the test data using a fitted model.
        """
        logger.debug(f"Generating predictions for model type: {type(fitted_model).__name__}")
        test_X_use = self.test_X_scaled

        if is_courselib_model:
            y_pred = fitted_model.decision_function(test_X_use)
        else:
            y_pred = fitted_model.predict(test_X_use)
            
        logger.info(f"Predictions generated for {type(fitted_model).__name__}.")
        return y_pred

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Internal method to calculate performance metrics (MSE, MAE, RMSE) on the test set.
        """
        calculated_metrics = {}
        calculated_metrics["mse"] = mean_squared_error(y_pred, y_true)
        calculated_metrics["mae"] = mean_absolute_error(y_pred, y_true)
        calculated_metrics["rmse"] = math.sqrt(calculated_metrics["mse"])
        
        return calculated_metrics

    def evaluate_model(self, model_name: str, model_instance, is_courselib_model: bool = False, **fit_params):
        """
        Trains, predicts, and evaluates a single model using the pre-prepared train/test data. 
        Metrics computed are MSE, MAE and RMSE
        
        Parameters:
        - model_name (str): A unique name for this model (e.g., 'Courselib LR', 'Sklearn Ridge').
        - model_instance: An instantiated model object (e.g., LinearRegression(), Ridge(alpha=1.0)).
        - is_courselib_model (bool): True if it's a courselib model requiring specific fit params.
        - **fit_params: Additional parameters to pass to the model's fit method.
        """
        logger.info(f"--- Starting evaluation for {model_name} ---")
        try:
            # Fit the model and get training history
            fitted_model = self._fit_model(model_instance, is_courselib_model, **fit_params)

            # Generate predictions on the test set
            y_pred = self._predict_model(fitted_model, is_courselib_model)
            y_pred_series = pd.Series(y_pred, index=self.test_Y_index, name='Predictions')
            
            # Calculate test set metrics
            test_metrics = self._calculate_metrics(self.test_Y_np, y_pred)
            
            # Store results
            self.metrics_results[model_name] = test_metrics
            self.predictions_results[model_name] = y_pred_series
            self.trained_models[model_name] = fitted_model
            
            # Log metrics
            test_metric_log_str = ", ".join([f"Test {k.upper()}: {v:.4f}" 
                                              for k, v in test_metrics.items() if k in ['mse', 'mae', 'rmse']])
            logger.info(f"Model {model_name} - {test_metric_log_str}")

            logger.info(f"--- Finished evaluation for {model_name} ---")

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}", exc_info=True)

    def evaluate_naive_baseline(self):
        """
        Calculates and stores metrics for the Naive Baseline (previous timepoint). 
        Metrics computed are MSE, MAE, and RMSE.
        """
        logger.info("--- Calculating Naive Baseline ---")
        
        naive_baseline_preds = self.naive_baseline_preds_unscaled
        test_Y_use = self.test_Y_np

        if len(naive_baseline_preds) != len(test_Y_use):
             logger.error("Naive baseline predictions and test_Y have mismatched lengths.")
             raise ValueError("Naive baseline predictions and test_Y have mismatched lengths.")

        calculated_metrics = self._calculate_metrics(test_Y_use, naive_baseline_preds)
        naive_baseline_series = pd.Series(naive_baseline_preds, index=self.test_Y_index, name='Naive Baseline Predictions')

        self.metrics_results["Naive Baseline"] = calculated_metrics
        self.predictions_results["Naive Baseline"] = naive_baseline_series

        metric_log_str = ", ".join([f"{k.upper()}: {v:.4f}" for k, v in calculated_metrics.items()])
        logger.info(f"Naive Baseline - Test {metric_log_str}")
        logger.info("--- Finished Naive Baseline Calculation ---")

    def get_metrics_results(self) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Returns the dictionary of all model evaluation metrics (MSE, MAE, RMSE).
        """
        return self.metrics_results

    def get_metric_for_model(self, model_name: str, metric_name: str):
        """
        Returns a specific metric for a specific model.
        """
        return self.metrics_results.get(model_name, {}).get(metric_name)

    def get_test_Y_series(self) -> pd.Series:
        """Returns the true test values as a pandas Series with its original DateTimeIndex."""
        return self.test_Y
        #return pd.Series(self.test_Y_np.flatten(), index=self.test_Y_index)

    def get_train_test_split(self):
        """
        Returns the scaled training and testing features, and the numpy target arrays.
        These are the arrays ready for model consumption.

        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
          (X_train_scaled, X_test_scaled, y_train_np, y_test_np)
        """
        return self.train_X_scaled, self.test_X_scaled, self.train_Y_np, self.test_Y_np

    def get_predictions_series_for_model(self, model_name: str) -> pd.Series:
        """
        Returns predictions for a specific model as a pandas Series with the correct DateTimeIndex.
        """
        return self.predictions_results.get(model_name)
    
    def get_trained_model(self, model_name: str):
        """
        Returns a trained model instance by its name.
        """
        return self.trained_models.get(model_name)
        

# =============================================================================
# Updated Main Execution Block (if __name__ == "__main__":)
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Import necessary modules for the standalone script execution
    from src.data.data_processor import AirQualityProcessor
    from src.features.feature_engineer import LagFeatureEngineer
    from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH
    from src.visualization.analysis import plot_error_by_time_group, plot_predictions_vs_actual, plot_acf_pacf

    from courselib.models.linear_models import LinearRegression
    from courselib.optimizers import GDOptimizer
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression
    from sklearn.linear_model import Ridge

    logger.info("--- Starting Model Evaluation Script ---")

    # Create dummy config for direct execution if not available
    if 'TARGET_POLLUTANT' not in locals():
        TARGET_POLLUTANT = "PM2.5"
        START_DATE = "2023-01-01"
        END_DATE = "2023-01-31"
        LAG_DEPTH = 5
        logger.warning("Using dummy config for standalone script execution.")


    processor = AirQualityProcessor(
        target_pollutant=TARGET_POLLUTANT,
        start_date=START_DATE,
        end_date=END_DATE
    )
    time_series = processor.get_target_time_series()
    logger.info("AirQualityProcessor completed.")

    logger.info("\n--- Plotting Autocorrelation (ACF) and Partial Autocorrelation (PACF) ---")
    plot_acf_pacf(time_series, lags=LAG_DEPTH * 2, title="ACF and PACF of Target Time Series")
    logger.info(f"ACF/PACF plot generated to help justify LAG_DEPTH={LAG_DEPTH}.")

    feature_engineer = LagFeatureEngineer(lag_depth=LAG_DEPTH)
    df_lagged_features = feature_engineer.prepare_supervised_data(time_series, return_separate=False)
    logger.info("LagFeatureEngineer completed.")
    logger.info(f"Prepared DataFrame for ModelEvaluator shape: {df_lagged_features.shape}")

    evaluator = ModelEvaluator(df_lagged_features, training_data_fraction=0.8)
    logger.info("ModelEvaluator initialized with data splits.")
    
    # Courserlib Linear Regression
    optimizer = GDOptimizer(learning_rate=0.0001)
    courselib_lr_model = LinearRegression(w=np.zeros(LAG_DEPTH), b=0.0, optimizer=optimizer)
    evaluator.evaluate_model(
        "Linear Regression (courselib)", 
        courselib_lr_model, 
        is_courselib_model=True,
        num_epochs=2000, 
        batch_size=32
    )
    
    # Scikit-learn Linear Regression
    sklearn_lr_model = SklearnLinearRegression()
    evaluator.evaluate_model("Scikit-learn Linear Regression", sklearn_lr_model)

    # Scikit-learn Ridge Regression
    ridge_model = Ridge(alpha=1.0) 
    evaluator.evaluate_model("Scikit-learn Ridge Regression", ridge_model)

    # Naive Baseline
    evaluator.evaluate_naive_baseline()

    # --- REVISED: Call get_metrics_results ---
    all_metrics_results = evaluator.get_metrics_results()
    logger.info("\n--- Summary of All Model Metrics Results ---")
    for model_name, metrics_data in all_metrics_results.items():
        test_metric_summary = ", ".join([f"Test {k.upper()}={v:.4f}" 
                                         for k, v in metrics_data.items() 
                                         if k in ["mse", "mae", "rmse"] and isinstance(v, (int, float))])
        if "error" in metrics_data:
            test_metric_summary += f", Error: {metrics_data['error']}"
        
        logger.info(f"{model_name}: {test_metric_summary}")
    # --- END REVISED ---

    # --- Plot Error by Time Group ---
    y_test_series_for_plot = evaluator.get_test_Y_series()
    
    y_pred_courselib_series = evaluator.get_predictions_series_for_model("Linear Regression (courselib)")
    if not y_pred_courselib_series.empty:
        logger.info("\nPlotting error by hour for Linear Regression (courselib)...")
        plot_error_by_time_group(y_test_series_for_plot, y_pred_courselib_series, group_by='hour')
    else:
        logger.warning("No predictions available for Linear Regression (courselib) for plotting.")
    
    y_pred_ridge_series = evaluator.get_predictions_series_for_model("Scikit-learn Ridge Regression")
    if not y_pred_ridge_series.empty:
        logger.info("\nPlotting error by weekday for Scikit-learn Ridge Regression...")
        plot_error_by_time_group(y_test_series_for_plot, y_pred_ridge_series, group_by='weekday')
    else:
        logger.warning("No predictions available for Scikit-learn Ridge Regression for plotting.")
    
    # --- Plot Predictions vs. Actual ---
    logger.info("\n--- Plotting Predictions vs. Actual Values ---")
    if not y_pred_ridge_series.empty:
        logger.info(f"\nPlotting Actual vs. Predicted for Scikit-learn Ridge Regression...")
        plot_predictions_vs_actual(y_test_series_for_plot, y_pred_ridge_series, model_name="Scikit-learn Ridge Regression")
    else:
        logger.warning("No predictions available for Scikit-learn Ridge Regression for plotting Actual vs. Predicted.")

    logger.info("\n--- Retrieving Trained Models (Example) ---")
    retrieved_lr = evaluator.get_trained_model("Linear Regression (courselib)")
    if retrieved_lr:
        logger.info(f"Retrieved Courselib LR model: {type(retrieved_lr).__name__}")
    
    retrieved_ridge = evaluator.get_trained_model("Scikit-learn Ridge Regression")
    if retrieved_ridge:
        logger.info(f"Retrieved Scikit-learn Ridge model: {type(retrieved_ridge).__name__}")
        if hasattr(retrieved_ridge, 'coef_'):
            logger.info(f"Ridge Coefficients (first 5): {retrieved_ridge.coef_[:5]}")


    logger.info("--- Model Evaluation Script Finished ---")