'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt # For visualization

# Assuming you have these ready from your previous steps
from src.data.data_processor import AirQualityProcessor
from src.features.feature_engineer import LagFeatureEngineer
from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH

# For this example, let's regenerate X and y if they aren't loaded from disk
# If X and y are saved in a previous step, you would load them here instead:
# X = pd.read_csv("data/processed/X_features.csv")
# y = pd.read_csv("data/processed/y_target.csv")

print("--- Preparing X and y data ---")
processor = AirQualityProcessor(
    target_pollutant=TARGET_POLLUTANT,
    start_date=START_DATE,
    end_date=END_DATE
)
air_quality_series = processor.get_target_time_series()

feature_engineer = LagFeatureEngineer(lag_depth=LAG_DEPTH)
X, y = feature_engineer.prepare_supervised_data(air_quality_series)
print("X and y successfully prepared.")
print(f"X shape: {X.shape}, y shape: {y.shape}")

def  train_test_split(df, training_data_fraction, shuffle=True):
    """
    Split all available data into a training and test data set.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame of available data.

    training_data_fraction : float
        The fraction of the data that should be used for training.
        E.g., `training_data_fraction=0.6` means 60%.

    shuffle : bool , optional
        If True, randomly reshuffles the data before the split.

    Returns:
    --------

    train_df: DataFrame
        Train split
        
    test_df: DataFrame
        Test split
    """
    
    train_df = df.iloc[:round(training_data_fraction*len(df)),:]
    test_df = df.iloc[round(training_data_fraction*len(df)):,:]
    
    return df, train_df, test_df

'''


import sys
import os
import logging
import math

# Configure basic logging for a standalone script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

'''
# Add the repo root (three levels up from this file) to sys.path
repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "AppliedML"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print(f"{repo_root} added to sys.path.")
else:
    print("Repo root already in sys.path.")
'''
    
# Get the absolute path of the current script's directory (src/models/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Add the current project's root directory to sys.path
#    We need to go up two levels from 'src/models/' to reach 'appliedML-air-quality/'
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # print(f"'{project_root}' added to sys.path for project imports.") # Optional print

# 2. Add the courselib parent directory to sys.path
#    We need to go up three levels from 'src/models/' to 'D:\Studium\Excercises\Applied ML\'
#    then enter the 'AppliedML' folder.
courselib_parent_path = os.path.abspath(os.path.join(current_script_dir, "..", "..", "..", "AppliedML"))
if courselib_parent_path not in sys.path:
    sys.path.insert(0, courselib_parent_path) # Insert at the beginning so it's searched first
    # print(f"'{courselib_parent_path}' added to sys.path for courselib access.") # Optional print

from courselib.models.linear_models import LinearRegression # type: ignore
from courselib.optimizers import GDOptimizer # type: ignore

from courselib.utils.metrics import mean_squared_error, mean_absolute_error # type: ignore
from courselib.utils.splits import train_test_split # type: ignore

from sklearn.linear_model import LinearRegression as LR_SciKit
from sklearn.linear_model import Ridge

import pandas as pd
import numpy as np


# Assuming you have these ready from your previous steps
from src.data.data_processor import AirQualityProcessor
from src.features.feature_engineer import LagFeatureEngineer
from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH

# For this example, let's regenerate X and y if they aren't loaded from disk
# If X and y are saved in a previous step, you would load them here instead:
# X = pd.read_csv("data/processed/X_features.csv")
# y = pd.read_csv("data/processed/y_target.csv")

def train_model():



    logging.info("Starting model training process...")

    # --- Data Loading and Preprocessing ---
    logging.info("Loading and processing data...")
    print("--- Preparing X and y data ---")
    processor = AirQualityProcessor(
        target_pollutant=TARGET_POLLUTANT,
        start_date=START_DATE,
        end_date=END_DATE
    )
    air_quality_series = processor.get_target_time_series()

    #feature_engineer = LagFeatureEngineer(lag_depth=LAG_DEPTH)
    feature_engineer = LagFeatureEngineer(lag_depth=12)
    df = feature_engineer.prepare_supervised_data(air_quality_series)
    print("DataFrame successfully prepared.")
    print(f"DataFrame shape: {df.shape}")

    logging.info(f"DataFrame shape after feature engineering and cleaning: {df.shape}")

    # --- Train/Test Split using courselib's train_test_split ---
    logging.info("Splitting data into training and testing sets using courselib.utils.splits...")

    X, Y, train_X, train_Y, test_X, test_Y = train_test_split(df, training_data_fraction=0.8, class_column_name='Target', shuffle=False, return_numpy=True)

    def custom_standard_scaler(train_X, test_X):
        """
        Applies standardization to features, fitting parameters on train_X
        and transforming both train_X and test_X using those parameters.

        Parameters:
        - train_X (np.ndarray): Training features.
        - test_X (np.ndarray): Testing features.

        Returns:
        - tuple: (train_X_scaled, test_X_scaled)
        """
        # Calculate mean and standard deviation ONLY from the TRAINING data
        train_X_mean = np.mean(train_X, axis=0)
        train_X_std = np.std(train_X, axis=0)

        # Handle cases where std might be zero (for features with constant values)
        train_X_std[train_X_std == 0] = 1

        # Apply standardization using the calculated training stats
        train_X_scaled = (train_X - train_X_mean) / train_X_std
        test_X_scaled = (test_X - train_X_mean) / train_X_std

        return train_X_scaled, test_X_scaled

    logging.info("Applying feature scaling to training and testing data...")
    train_X, test_X = custom_standard_scaler(train_X, test_X)
    logging.info("Feature scaling complete.")

    logging.info(f"Training data shape: X_train={train_X.shape}, y_train={train_Y.shape}")
    logging.info(f"Testing data shape: X_test={test_X.shape}, y_test={test_Y.shape}")

    # --- Model Training: courselib LinearRegression ---
    logging.info("Training Linear Regression model using courselib...")

    # Get number of features for model initialization
    n_features = train_X.shape[1]

    # Initialize weights 'w' and bias 'b'
    initial_w = np.zeros(n_features)
    initial_b = np.array(0.0)

    # Initialize model
    optimizer = GDOptimizer(learning_rate=0.0001)
    #model = LinearRegression(w=np.zeros(train_X.shape[1]), b=0.0, optimizer=optimizer)
    model = LinearRegression(w=initial_w, b=initial_b, optimizer=optimizer)
    

    def root_mean_squared_error(y_true, y_pred):
        return math.sqrt(mean_squared_error(y_pred, y_true))

    metrics_for_fit = {
        "RMSE": root_mean_squared_error,
        "MAE": mean_absolute_error # courselib's mean_absolute_error directly
    }

    # Train model
    metrics = model.fit(train_X, train_Y, num_epochs=2000, batch_size=32, 
                        compute_metrics=True, metrics_dict=metrics_for_fit)
        
    logging.info("Linear Regression (courselib) trained successfully.")
    #logging.info(f"Metrics history during training (last epoch):\n{metrics}")


    # --- SciKit Learn ---
    sklearn_lr_model = LR_SciKit()

    # The .fit() method will internally choose an efficient solver,
    # which might be the Normal Equation or SVD for this size of data.
    sklearn_lr_model.fit(train_X, train_Y)
    y_pred_sklearn = sklearn_lr_model.predict(test_X)


    # --- SciKit Ridge ---
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(train_X, train_Y)
    logging.info("Scikit-learn Ridge Regression trained successfully.")
    logging.info("Evaluating scikit-learn Ridge model on the test set...")
    y_pred_ridge = ridge_model.predict(test_X)
    test_mse_ridge = mean_squared_error(y_pred_ridge, test_Y)
    test_mae_ridge = mean_absolute_error(y_pred_ridge, test_Y)
    test_rmse_ridge = root_mean_squared_error(y_pred_ridge, test_Y)

    # --- Naive Baseline ---
    logging.info("Calculating Naive Baseline...")
    # For time series, a common naive baseline is simply the mean of the training target
    y_pred_naive = np.full_like(test_Y, train_Y.mean())
    logging.info("Naive Baseline calculated.")

    # --- Model Evaluation on Test Set ---
    logging.info("Evaluating models on the test set...")

    # Evaluate Linear Regression (courselib)
    y_pred_lr = model.decision_function(test_X) # Get predictions from the trained model
    final_mse_lr = mean_squared_error(y_pred_lr, test_Y) # Using courselib's MSE
    final_mae_lr = mean_absolute_error(y_pred_lr, test_Y) # Using courselib's MAE
    final_rmse_lr = root_mean_squared_error(y_pred_lr, test_Y) # Calculated from courselib's MSE
    logging.info(f"Linear Regression (courselib) - Test MSE: {final_mse_lr:.4f}, Test MAE: {final_mae_lr:.4f}, Test RMSE: {final_rmse_lr:.4f}")

    test_mse_sklearn = mean_squared_error(y_pred_sklearn, test_Y)
    test_mae_sklearn = mean_absolute_error(y_pred_sklearn, test_Y)
    test_rmse_sklearn = root_mean_squared_error(y_pred_sklearn, test_Y)

    logging.info(f"Scikit-learn Linear Regression - Test MSE: {test_mse_sklearn:.4f}, Test MAE: {test_mae_sklearn:.4f}, Test RMSE: {test_rmse_sklearn:.4f}")

    logging.info(f"Scikit-learn Ridge Regression - Test MSE: {test_mse_ridge:.4f}, Test MAE: {test_mae_ridge:.4f}, Test RMSE: {test_rmse_ridge:.4f}")

    # Evaluate Naive Baseline
    final_mse_naive = mean_squared_error(y_pred_naive, test_Y) # Using courselib's MSE
    final_mae_naive = mean_absolute_error(y_pred_naive, test_Y) # Using courselib's MAE
    final_rmse_naive = root_mean_squared_error(y_pred_naive, test_Y) # Calculated from courselib's MSE
    logging.info(f"Naive Baseline - Test MSE: {final_mse_naive:.4f}, Test MAE: {final_mae_naive:.4f}, Test RMSE: {final_rmse_naive:.4f}")

    logging.info("Model training and evaluation complete.")


# =============================================================================
# 6. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    train_model()