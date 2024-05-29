import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline

import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column with the average rides from
    - 1 week ago
    - 2 weeks ago
    - 3 weeks ago
    - 4 weeks ago
    """
    X['average_rides_last_4_weeks'] = 0.25*(
        X[f'rides_previous_{1*7*24}_hour'] + \
        X[f'rides_previous_{2*7*24}_hour'] + \
        X[f'rides_previous_{3*7*24}_hour'] + \
        X[f'rides_previous_{4*7*24}_hour']
    )
    return X

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds two columns
    in the form of numerical features:
    - hour of day
    - day of week
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        X_ = X.copy()

        # Generate numeric columns from datetime
        X_['hour'] = X_['pickup_hour'].dt.hour
        X_['day_of_week'] = X_['pickup_hour'].dt.dayofweek

        return X_.drop(columns=['pickup_hour'])

def get_pipeline(**hyperparams) -> Pipeline:
    # sklearn transform
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks,
        validate=False
    )

    # sklearn transform
    add_temporal_features = TemporalFeatureEngineer()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )
