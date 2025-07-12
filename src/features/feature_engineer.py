import pandas as pd
import numpy as np
from src.utils.config import LAG_DEPTH

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
            lag_depth (int): The number (k) of past observations to use as features.
                             E.g., if lag_depth=24, it will create lag_1, ..., lag_24 features.
        """
        if not isinstance(lag_depth, int) or lag_depth <= 0:
            raise ValueError("lag_depth must be a positive integer.")
        self.lag_depth = lag_depth

    def prepare_supervised_data(self, series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Generates lagged features matrix (X) and the corresponding target vector (y)
        from the input time series.

        X will have columns 'lag_1', 'lag_2', ..., 'lag_k' (newest to oldest lag).
        y will be the original series value at the current timestamp (from k+1 to T).

        Parameters:
        -----------
        series (pd.Series): The input time series (i.e., target pollutant values).
                            It must have a DatetimeIndex.

        Returns:
        -----------
        tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): The feature matrix (n x k).
            - y (pd.Series): The target vector (n).
            The indices of X and y will be aligned and correspond to the prediction timestamp.
        """

        if len(series) <= self.lag_depth:
            raise ValueError(f"Series length ({len(series)}) must be greater than lag_depth ({self.lag_depth}) to create features.")


        df = pd.DataFrame(series)           # create a dataframe from the time series
        df.columns = ['original_value']     # rename the column with the time series values

        print(f"Generating {self.lag_depth} lag features and aligning target vector...")
        
        for i in range(1, self.lag_depth + 1):
            col_name = f'lag_{i}'
            # Add a new column which is the original series shifted by i values downwards,
            # i.e., remove the i last values of the series, corresponding to i lags
            df[col_name] = df['original_value'].shift(i)

        # Remove all rows without values (resulting from shifting).
        # This way, in the dataframe only the lagged values remain
        df_features = df.dropna(axis=0)
        y = df_features['original_value']
        X = df_features.drop(columns=['original_value'])
        
        if X.empty or y.empty: # Check after all processing
            raise ValueError("After generating lags and dropping NaNs, no valid data points remain. "
                             "Check series length and lag_depth.")
        
        print(f"Feature matrix (X) shape: {X.shape}")
        print(f"Target vector (y) shape: {y.shape}")
        
        return X, y
    
if __name__ == '__main__':
        from src.data.data_processor import AirQualityProcessor
        from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH

        print("\n--- Testing LagFeatureEngineer with AirQualityProcessor output ---")

        # Get processed time series
        processor = AirQualityProcessor(
            target_pollutant=TARGET_POLLUTANT,
            start_date=START_DATE,
            end_date=END_DATE
        )
        try:
            air_quality_series = processor.get_target_time_series()
            print("\nSuccessfully retrieved target time series. Head:")
            print(air_quality_series.head())

            # Generate lagged feature matrix (X) and target (y)
            feature_engineer = LagFeatureEngineer(lag_depth=LAG_DEPTH)
            X, y = feature_engineer.prepare_supervised_data(air_quality_series)

            print("\nGenerated Feature Matrix (X) head:")
            print(X.head())
            print("\nGenerated Feature Matrix (X) info:")
            X.info()
            print("\nGenerated Feature Matrix (X) shape:", X.shape)
            print("\nGenerated Target Vector (y) head:")
            print(y.head())
            print("\nGenerated Target Vector (y) info:")
            y.info()
            print("\nGenerated Target Vector (y) shape:", y.shape)

            # Verify alignment of X and y (e.g., last values)
            if not X.empty and not y.empty:
                print(f"\nLast X index: {X.index[-1]}")
                print(f"Last y index: {y.index[-1]}")
                if X.index.equals(y.index):
                    print("X and y indices are perfectly aligned.")
                else:
                    print("WARNING: X and y indices are NOT aligned!")

        except Exception as e:
            print(f"Feature Engineering test failed: {e}")
    