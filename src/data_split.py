from datetime import datetime
from typing import Tuple

import pandas as pd

def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits data into train and test sets.
    """

    mask = df['pickup_hour'] < cutoff_date

    X_train, y_train = df[mask].drop(target_column_name, axis=1), df[mask][target_column_name]
    X_test, y_test = df[~mask].drop(target_column_name, axis=1), df[~mask][target_column_name]

    return X_train, y_train, X_test, y_test
