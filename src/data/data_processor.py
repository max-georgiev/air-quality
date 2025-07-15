import sys
import os
import logging

# Setup logging for the script
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a logger instance

project_parent_path = os.path.abspath(os.getcwd())
if project_parent_path not in sys.path:
    sys.path.insert(0, project_parent_path)

import pandas as pd
from src.utils.config import RAW_DATA_PATH

class AirQualityProcessor:
    def __init__(self, target_pollutant, start_date, end_date):
        """
        Initializes the AirQualityProcessor

        Parameters:
        -----------
        target_pollutant : str
            The name of the pollutant column to extract (e.g., 'NO2(GT)').
        start_date : str
            The start date for the continuous subset (format 'DD/MM/YYYY').
        end_date : str
            The end date for the continuous subset (format 'DD/MM/YYYY').
        """

        self.file_path = RAW_DATA_PATH
        self.target_pollutant = target_pollutant
        self.start_date = start_date
        self.end_date = end_date

        self._raw_df = None # Initialize a cached raw DataFrame

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads the raw Air Quality UCI dataset with correct parsing for Date/Time,
        missing values, and decimal separators.
        Caches the loaded DataFrame to avoid repeated loading.
        """

        # Check if data is already cached
        if self._raw_df is not None:
            print("Using cached raw data.")
            return self._raw_df

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Raw data file not found at: {self.file_path}. "
                                    "Please download it and place it there.")

        df = pd.read_csv(
            self.file_path,
            sep=';',                # deliimter for the dataset
            na_values=['-200'],     # missing values indicator
            decimal=','             # decimal separator for the dataset
        )

        # Combine Date and Time columns into a single column to use it for indexing
        df['DateTime'] = df['Date'] + ' ' + df['Time']

        # Convert the date and time columns to actual datetime objects
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Time'] = pd.to_datetime(df['Time'], format='%H.%M.%S')
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H.%M.%S')

        df.set_index('DateTime', inplace=True)      # set the DateTime column to be the index column
        df.index.name = 'DateTime'                  # ensure the index has a name
        df.dropna(axis=1, how='all', inplace=True)  # drop all empty unnamed columns
        df.sort_index(inplace=True)                 # ensure index column is sorted

        logger.info("Raw data loaded and initially parsed.")
        
        # Cache the loaded DataFrame
        self._raw_df = df
        return self._raw_df
    
    def get_target_time_series(self) -> pd.Series:
        """
        Loads raw data (or uses cached), performs initial cleaning, and extracts 
        a specific pollutant time series, filtered by date range.

        Returns:
        -----------
        time_series : pd.Series
            A pandas Series containing the cleaned and selected pollutant time series, with a datetime index.
        """
        df = self.load_raw_data()

        if self.target_pollutant not in df.columns:
            raise ValueError(f"Target pollutant '{self.target_pollutant}' not found in the dataset columns: {df.columns.tolist()}")
        
        # Filter the dataframe only for selected time period
        start_dt = pd.to_datetime(self.start_date, format='%d/%m/%Y')
        end_dt = pd.to_datetime(self.end_date, format='%d/%m/%Y')
        df_filtered_time = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].copy()

        # Convert the target pollutant column to numeric
        df_filtered_time[self.target_pollutant] = pd.to_numeric(
            df_filtered_time[self.target_pollutant], errors='coerce') # invalid parsing set as NaN
        
        # Drop rows with NaN's in the target pollutant column, overriding the dataframe
        df_filtered_time.dropna(subset=[self.target_pollutant], inplace=True)

        time_series = df_filtered_time[self.target_pollutant]

        if time_series.empty:
            raise ValueError(f"No data found for {self.target_pollutant} in the specified date range "
                             f"({self.start_date} to {self.end_date}) after cleaning.")
        
        return time_series

# For demonstrating its purpose when run as a main script:
if __name__ == '__main__':
    from src.utils.config import TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH # Import global config
    # Use config variables to define the purpose
    logger.info(f"--- Demonstrating AirQualityProcessor functionality ---")
    logger.info(f"Target Pollutant: {TARGET_POLLUTANT}")
    logger.info(f"Date Range: {START_DATE} to {END_DATE}")

    processor = AirQualityProcessor(
        target_pollutant=TARGET_POLLUTANT,
        start_date=START_DATE,
        end_date=END_DATE
    )

    try:
        time_series = processor.get_target_time_series()
        logger.info(f"Successfully processed time series for {TARGET_POLLUTANT}.")
        logger.info(f"Time Series Length: {len(time_series)}")
        logger.info(f"Time Series Start Date: {time_series.index.min()}")
        logger.info(f"Time Series End Date: {time_series.index.max()}")
        logger.info("\nFirst 5 entries of the processed time series:")
        print(time_series.head()) # Use print for the actual data output for clarity
        logger.info("\nBasic statistics of the processed time series:")
        print(time_series.describe()) # Use print for the actual data output for clarity

    except FileNotFoundError as e:
        logger.error(f"Error: {e}. Please ensure '{RAW_DATA_PATH}' exists.")
    except ValueError as e:
        logger.error(f"Data processing error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during data processing: {e}")

    logger.info("--- AirQualityProcessor Demonstration Finished ---")