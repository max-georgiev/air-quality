import sys
import os
import numpy as np
import pandas as pd
import logging
import math # For sqrt
import argparse # For command-line arguments in __main__
from typing import Dict, Callable, Tuple, Union, Any


project_parent_path = os.path.abspath(os.getcwd())
if project_parent_path not in sys.path:
    sys.path.insert(0, project_parent_path)

courselib_parent_path = os.path.abspath(os.path.join(os.getcwd(), "..", "AppliedML"))
if courselib_parent_path not in sys.path:
    sys.path.insert(0, courselib_parent_path)

# Import necessary modules (now that sys.path is correctly set)
from courselib.utils.metrics import mean_squared_error, mean_absolute_error
from courselib.utils.splits import train_test_split
from courselib.models.linear_models import LinearRegression
from courselib.optimizers import GDOptimizer
from sklearn.linear_model import Ridge, LinearRegression as SklearnLinearRegression

# Specific imports for the main block, now correctly resolved:
from src.data.data_processor import AirQualityProcessor
from src.features.feature_engineer import LagFeatureEngineer
from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH

# Set up logging for this module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    if test_data.ndim == 1: # Ensure test data is also 2D for consistent scaling
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
            # Fallback to first column if 'lag_1' is not present by name (the first is the 'Target')
            self.naive_baseline_preds_unscaled = self.test_X_raw.iloc[:, 0].to_numpy()
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

        train_Y_fit = self.train_Y_np.reshape(-1, 1) if not is_courselib_model else self.train_Y_np
        train_metrics: Dict[str, float] = {}

        if is_courselib_model:
            courselib_metrics_dict = {
                    "mse": mean_squared_error,
                    "mae": mean_absolute_error
                }
            
            metrics_history = model_instance.fit(
                    X=self.train_X_scaled, 
                    y=train_Y_fit, 
                    compute_metrics=True, # <-- Enable metric computation during fit
                    metrics_dict=courselib_metrics_dict, # <-- Pass the metrics functions
                    **fit_params
                )
            
            for metric_name, values in metrics_history.items():
                train_metrics[metric_name] = values[-1]
            train_metrics["rmse"] = math.sqrt(train_metrics["mse"]) # Calculate RMSE

        else:
            model_instance.fit(self.train_X_scaled, train_Y_fit)
            # Manually compute training metrics for scikit-learn models
            train_y_pred = model_instance.predict(self.train_X_scaled)
            if train_y_pred.ndim > 1:
                train_y_pred = train_y_pred.flatten()
            train_metrics = self._calculate_metrics(self.train_Y_np, train_y_pred)
        
        logger.info(f"Model {type(model_instance).__name__} trained successfully.")
        
        return model_instance, train_metrics

    def _predict_model(self, fitted_model, is_courselib_model: bool) -> np.ndarray:
        """
        Internal method to generate predictions on the test data using a fitted model.
        """
        logger.debug(f"Generating predictions for model type: {type(fitted_model).__name__}")
        

        if is_courselib_model:
            y_pred = fitted_model.decision_function(self.test_X_scaled)
        else:
            y_pred = fitted_model.predict(self.test_X_scaled)
            
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

    def evaluate_model(self, model_name: str, model_config: Dict[str, Any]):
        """
        Trains, predicts, and evaluates a single model given its configuration.
        Metrics computed are MSE, MAE and RMSE.
        
        Parameters:
        - model_name (str): A unique name for the model.
        - model_config (Dict[str, Any]): A dictionary containing model configuration,
                                         including 'model_class', 'is_courselib_model',
                                         'init_params', 'fit_params', and potentially 'optimizer'
                                         for courselib LinRegr model.
        """
        logger.info(f"--- Starting evaluation for {model_name} ---")

        
        model_class = model_config['model_class']
        is_courselib_model = model_config.get('is_courselib_model', False)
        init_params = model_config.get('init_params', {})
        fit_params = model_config.get('fit_params', {})

        if is_courselib_model:
            num_features = self.train_X_scaled.shape[1]
            optimizer_instance = model_config.get('optimizer') # Get pre-instantiated optimizer
            model_instance = model_class(w=np.zeros(num_features), b=0.0, optimizer=optimizer_instance)
        else:
            # For scikit-learn models
            model_instance = model_class(**init_params)
        
        try:
            # Fit the model and get training history
            fitted_model, train_metrics = self._fit_model(model_instance, is_courselib_model, **fit_params)

            # Generate predictions on the test set
            y_pred = self._predict_model(fitted_model, is_courselib_model)
            y_pred_series = pd.Series(y_pred, index=self.test_Y_index, name='Predictions')
            
            # Calculate test set metrics
            test_metrics = self._calculate_metrics(self.test_Y_np, y_pred)
            
            # Store results: Combine train and test metrics
            full_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
            full_metrics.update(test_metrics)
            
            self.metrics_results[model_name] = full_metrics # Store combined metrics
            self.predictions_results[model_name] = y_pred_series
            self.trained_models[model_name] = fitted_model
            
            # Log metrics (adjusting for both train and test)
            train_metric_log_str = ", ".join([f"Train {k.upper()}: {v:.4f}" 
                                               for k, v in train_metrics.items() if k in ['mse', 'mae', 'rmse']])
            test_metric_log_str = ", ".join([f"Test {k.upper()}: {v:.4f}" 
                                             for k, v in test_metrics.items() if k in ['mse', 'mae', 'rmse']])
            logger.info(f"Model {model_name} - {train_metric_log_str} | {test_metric_log_str}")

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

    def get_metric_for_model(self, model_name: str, metric_name: str, is_train_metric=False):
        """
        Returns a specific metric for a specific model.
        """
        key = f"train_{metric_name}" if is_train_metric else metric_name
        return self.metrics_results.get(model_name, {}).get(key)

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
    # Logging configuration for standalone script execution
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("--- Starting Model Training Script ---")

    # Define available model configurations with default hyperparams
    ALL_MODEL_CONFIGURATIONS = {
        "courselib_lr": {
            "model_class": LinearRegression,
            "is_courselib_model": True,
            "optimizer": GDOptimizer(learning_rate=0.001), # Default learning rate
            "fit_params": {"num_epochs": 1000, "batch_size": 32} # Default fit params
        },
        "sklearn_lr": {
            "model_class": SklearnLinearRegression,
            "is_courselib_model": False,
            "init_params": {}
        },
        "sklearn_ridge": {
            "model_class": Ridge,
            "is_courselib_model": False,
            "init_params": {"alpha": 1.0} # Default alpha
        }
    }

    # Setup argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Train a specified regression model using lagged air quality data.")
    parser.add_argument("--model", type=str, required=True, choices=ALL_MODEL_CONFIGURATIONS.keys(),
                        help=f"Name of the model to train. Choose from: {', '.join(ALL_MODEL_CONFIGURATIONS.keys())}")
    parser.add_argument("--lag_depth", type=int, default=LAG_DEPTH,
                        help=f"Lag depth for feature engineering. Default: {LAG_DEPTH}")
    # Arguments for courselib_lr
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for courselib_lr model. Default: 0.001")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs for courselib_lr model. Default: 1000")
    # Argument for sklearn_ridge
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha (regularization strength) for sklearn_ridge model. Default: 1.0")

    args = parser.parse_args()

    # --- Data Preparation ---
    # This minimal setup ensures the script is runnable as a standalone.
    # In a full pipeline, data might be loaded from pre-processed files.
    try:
        processor = AirQualityProcessor(
            target_pollutant=TARGET_POLLUTANT,
            start_date=START_DATE,
            end_date=END_DATE
        )
        time_series = processor.get_target_time_series()
        logger.info("AirQualityProcessor completed for data generation.")

        feature_engineer = LagFeatureEngineer(lag_depth=args.lag_depth)
        df_lagged_features = feature_engineer.prepare_supervised_data(time_series, return_separate=False)
        logger.info(f"LagFeatureEngineer completed with lag_depth={args.lag_depth}. DataFrame shape: {df_lagged_features.shape}")

        # Initialize ModelEvaluator with the prepared data
        evaluator = ModelEvaluator(df_lagged_features, training_data_fraction=0.8)
        logger.info("ModelEvaluator initialized with data splits.")
    except Exception as e:
        logger.error(f"Error during data preparation or ModelEvaluator initialization: {e}", exc_info=True)
        sys.exit(1) # Exit if data setup fails

    # --- Model Training and Evaluation ---
    model_name_to_train = args.model
    model_config = ALL_MODEL_CONFIGURATIONS[model_name_to_train]
    
    # Override hyperparameters based on command-line arguments for the selected model
    if model_name_to_train == "courselib_lr":
        model_config["optimizer"] = GDOptimizer(learning_rate=args.learning_rate)
        model_config["fit_params"]["num_epochs"] = args.num_epochs
    elif model_name_to_train == "sklearn_ridge":
        model_config["init_params"]["alpha"] = args.alpha
    
    # Train and evaluate the specified model
    evaluator.evaluate_model(model_name_to_train, model_config)

    # --- Print Final Metrics for the Trained Model ---
    trained_model_metrics = evaluator.get_metrics_results().get(model_name_to_train)
    if trained_model_metrics and 'error' not in trained_model_metrics:
        logger.info(f"\n--- Final Metrics for Trained Model: {model_name_to_train} ---")
        train_mse = trained_model_metrics.get('train_mse')
        train_mae = trained_model_metrics.get('train_mae')
        train_rmse = trained_model_metrics.get('train_rmse')
        test_mse = trained_model_metrics.get('mse')
        test_mae = trained_model_metrics.get('mae')
        test_rmse = trained_model_metrics.get('rmse')

        if train_rmse is not None: # Check if training metrics were recorded
            logger.info(f"Train RMSE: {train_rmse:.4f}")
            logger.info(f"Train MAE: {train_mae:.4f}")
            logger.info(f"Train MSE: {train_mse:.4f}")
        if test_rmse is not None:
            logger.info(f"Test RMSE: {test_rmse:.4f}")
            logger.info(f"Test MAE: {test_mae:.4f}")
            logger.info(f"Test MSE: {test_mse:.4f}")
    elif trained_model_metrics and 'error' in trained_model_metrics:
        logger.error(f"Training for {model_name_to_train} failed: {trained_model_metrics['error']}")
    else:
        logger.error(f"No metrics available for model: {model_name_to_train}. Training might have failed or model name is incorrect.")

    logger.info("--- Model Training Script Finished ---")