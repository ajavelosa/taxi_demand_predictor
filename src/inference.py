from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config

def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Get DataFrame with model predictions for rides in the next hour
    """
    predictions = model.predict(features)
    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)

    return results

def load_batch_of_features_from_store(
        current_date: datetime,
) -> pd.DataFrame:
    """
    Load batch of features from hopsworks feature store
    """
    feature_store = get_feature_store()

    n_features = config.N_FEATURES

    # read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours = 1) # one hour before
    fetch_data_from = current_date - timedelta(days  = 28) # four weeks ago
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to - timedelta(days=1),
    )
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    # validate that we are not missing data
    location_ids = ts_data['location_id'].unique()
    assert len(ts_data) == n_features*len(location_ids), 'Time-series data is incomplete'

    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    print(f'{ts_data=}')
