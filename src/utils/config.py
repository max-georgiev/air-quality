import os

# Data Configuration
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DATA_DIR = os.path.join(_project_root, 'data', 'raw')
RAW_DATA_FILE = 'AirQualityUCI.csv'
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)

# Other config variables like TARGET_POLLUTANT, START_DATE, END_DATE, LAG_DEPTH etc.
TARGET_POLLUTANT = 'NO2(GT)'
START_DATE = '10/03/2004'
END_DATE = '08/06/2004'
LAG_DEPTH = 24
PROCESSED_DATA_DIR = os.path.join('data', 'processed')