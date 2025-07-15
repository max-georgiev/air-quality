import sys
import os
import logging

project_parent_path = os.path.abspath(os.getcwd())
if project_parent_path not in sys.path:
    sys.path.insert(0, project_parent_path)

# Setup logging for the script
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a logger instance

import pandas as pd
import numpy as np
from src.utils.config import LAG_DEPTH
import logging

class LagFeatureEngineer:
    """
    A class to create lagged features matrix (X) and a corresponding target vector (y)
    from a given time series for supervised learning.
    """
    def __init__(self, lag_depth: int):
        """
        Initializes the LagFeatureEngineer.

        Parameters:
        -----------
        lag_depth : int 
            The number (k) of past observations to use as features.
                E.g., if lag_depth=24, it will create lag_1, ..., lag_24 features.
        """
        if not isinstance(lag_depth, int) or lag_depth <= 0:
            raise ValueError("lag_depth must be a positive integer.")
        self.lag_depth = lag_depth

    def prepare_supervised_data(self, series: pd.Series, return_separate=False):
        """
        Generates lagged features matrix (X) and the corresponding target vector (y)
        from the input time series.

        X will have columns 'lag_1', 'lag_2', ..., 'lag_k' (newest to oldest lag).
        y will be the original series value at the current timestamp (from k+1 to T).

        Parameters:
        -----------
        series : pd.Series
            The input time series (i.e., target pollutant values).
                It must have a DatetimeIndex.

        return_separate : bool
            If True, returns the lagged features matrix X and the target vector y as separate DataFrames

        Returns:
        --------
        X, y : tuple[pd.DataFrame, pd.Series]
            A tuple containing:
            - X (pd.DataFrame): The feature matrix (n x k).
            - y (pd.Series): The target vector (n).
            The indices of X and y will be aligned and correspond to the prediction timestamp.
        """

        if len(series) <= self.lag_depth:
            raise ValueError(f"Series length ({len(series)}) must be greater than lag_depth ({self.lag_depth}) to create features.")

        df = pd.DataFrame(series)       # create a dataframe from the time series
        df.columns = ['Target']         # rename the column with the time series values

        print(f"Generating {self.lag_depth} lag features and aligning target vector...")
        
        for i in range(1, self.lag_depth + 1):
            # Add a new column 'lag_i' which is the original series shifted by i values downwards,
            # i.e., remove the i last values of the series, corresponding to i lags
            df[f'lag_{i}'] = df['Target'].shift(i)

        # Remove all rows without values (resulting from shifting).
        # This way, in the dataframe only the lagged values remain
        df_features = df.dropna(axis=0)

        if return_separate:
            y = df_features['Target']
            X = df_features.drop(columns=['Target'])
        
            if X.empty or y.empty: # Check after all processing
                raise ValueError("After generating lags and dropping NaNs, no valid data points remain. "
                                    "Check series length and lag_depth.")
        
            print(f"Feature matrix (X) shape: {X.shape}")
            print(f"Target vector (y) shape: {y.shape}")
        
            return X, y
        else:
            return df_features
    
if __name__ == '__main__':
    from src.data.data_processor import AirQualityProcessor
    from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH # Import global config
    
    logger.info(f"--- Demonstrating LagFeatureEngineer functionality ---")
    logger.info(f"Using default LAG_DEPTH: {LAG_DEPTH}")
    # Step 1: Get a sample time series using AirQualityProcessor
    logger.info("Step 1: Retrieving sample time series for feature engineering...")
    processor = AirQualityProcessor(
        target_pollutant=TARGET_POLLUTANT,
        start_date=START_DATE,
        end_date=END_DATE
    )
    try:
        air_quality_series = processor.get_target_time_series()
        logger.info(f"Successfully retrieved time series of length {len(air_quality_series)}.")
        logger.info("Time Series Head:\n" + str(air_quality_series.head()))

        # Step 2: Initialize and use LagFeatureEngineer
        logger.info(f"\nStep 2: Initializing LagFeatureEngineer with lag_depth={LAG_DEPTH}...")
        feature_engineer = LagFeatureEngineer(lag_depth=LAG_DEPTH)
        
        # Get the lagged features and target as separate DataFrames/Series
        X_features, y_target = feature_engineer.prepare_supervised_data(air_quality_series, return_separate=True)

        # Step 3: Display results
        logger.info("\nStep 3: Displaying generated lagged features and target...")
        logger.info(f"Generated Feature Matrix (X) shape: {X_features.shape}")
        logger.info(f"Generated Target Vector (y) shape: {y_target.shape}")
        
        logger.info("\nFirst 5 rows of Feature Matrix (X):")
        print(X_features.head())
        logger.info("\nFirst 5 values of Target Vector (y):")
        print(y_target.head())
        
        logger.info("\nFeature Matrix (X) Info:")
        X_features.info()
        logger.info("\nTarget Vector (y) Info:")
        y_target.info()

        # Optional: Verify alignment
        if not X_features.empty and not y_target.empty and X_features.index.equals(y_target.index):
            logger.info("X and y indices are perfectly aligned, as expected.")
        else:
            logger.warning("WARNING: X and y indices are NOT aligned or data is empty!")

    except Exception as e:
        logger.error(f"An error occurred during feature engineering demonstration: {e}", exc_info=True)

    logger.info("--- LagFeatureEngineer Demonstration Finished ---")