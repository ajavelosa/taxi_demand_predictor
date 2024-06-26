import os
from dotenv import load_dotenv

from src.paths import PARENT_DIR

# load key-value pairs from .env file in the parent directory
load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'ny_taxi_demand_predictor'
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except KeyError:
    raise KeyError('Create a .env file in the project root with the HOPSWORKS_API_KEY')

MODEL_NAME = 'ny_taxi_demand_predictor_next_hour'
MODEL_VERSION = 1

# Added prediction feature group for monitoring purposes
FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'
FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view'
FEATURE_VIEW_MONITORING = 'predictions_vs_actuals_for_monitoring_feature_view'

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = 'time_series_hourly_feature_group'
FEATURE_VIEW_VERSION = 1

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28
