import pandas as pd
import os
from src.utils.config import RAW_DATA_PATH

class AirQualityProcessor:
    def __init__(self, target_pollutant, start_date, end_date):
        """
        Initializes the AirQualityProcessor

        Parameters:
        -----------
        file_path (str): Path to the raw AirQualityUCI.csv file.
        target_pollutant (str): The name of the pollutant column to extract (e.g., 'NO2(GT)').
        start_date (str): The start date for the continuous subset (format 'DD/MM/YYYY').
        end_date (str): The end date for the continuous subset (format 'DD/MM/YYYY').
        """

        self.file_path = RAW_DATA_PATH
        self.target_pollutant = target_pollutant
        self.start_date = start_date
        self.end_date = end_date

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads the raw Air Quality UCI dataset with correct parsing for Date/Time,
        missing values, and decimal separators.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Raw data file not found at: {self.file_path}. "
                                    "Please download it and place it there.")
        

        df = pd.read_csv(
            self.file_path,
            sep=';',                                        # deliimter for the dataset
            na_values=['-200'],                             # missing values indicator
            decimal=','                                     # decimal separator for the dataset
        )

        # combine Date and Time columns into a new temporary column
        df['CombinedDateTime'] = df['Date'] + ' ' + df['Time']

        # add DateTime column with a datetime custom format
        df['DateTime'] = pd.to_datetime(df['CombinedDateTime'], format='%d/%m/%Y %H.%M.%S')

        df.set_index('DateTime', inplace=True)      # set the DateTime column to be the index column
        df.index.name = 'DateTime'                  # ensure the index has a name
        df.dropna(axis=1, how='all', inplace=True)  # drop all empty unnamed columns
        df.sort_index(inplace=True)                 # ensure index column is sorted

        # drop the original Date and Time columns and the temporary column CombinedDateTime
        df.drop(columns=['Date', 'Time', 'CombinedDateTime'], errors='ignore', inplace=True)

        print("Raw data loaded and initially parsed.")
        return df