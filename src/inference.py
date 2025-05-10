from datetime import datetime, timedelta, timezone

import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore

import sys
from pathlib import Path

# Add the src folder to the Python module search path
sys.path.append(str(Path(__file__).resolve().parent))

import config as config
from data_utils import transform_ts_data_info_features_and_target


def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
    )


def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    features_2 = features.copy()
    features = features.drop(columns=['pickup_hour', 'station_id'])
    predictions = model.predict(features, predict_disable_shape_check=True)

    results = pd.DataFrame()
    results["station_id"] = features_2["station_id"].values
    results["predicted_demand"] = predictions.round(0)

    return results


def load_model_from_registry(version=None):
    from pathlib import Path
    import joblib
    import lightgbm as lgb  # <-- Add this line

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    # Get the latest version if not specified
    models = model_registry.get_models(name=config.MODEL_NAME)
    model = max(models, key=lambda model: model.version)
    model_dir = model.download()

    # Load the LightGBM model
    model = joblib.load(Path(model_dir) / "lgb_model.pkl")

    return model



def load_metrics_from_registry(version=None):

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    models = model_registry.get_models(name=config.MODEL_NAME)
    model = max(models, key=lambda model: model.version)

    return model.training_metrics 


def fetch_next_hour_predictions():
    # Get current UTC time and round up to next hour
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    df = fg.read()
    # Then filter for next hour in the DataFrame
    df = df[df["pickup_hour"] == next_hour]

    print(f"Current UTC time: {now}")
    print(f"Next hour: {next_hour}")
    print(f"Found {len(df)} records")
    return df


def fetch_predictions(hours):
    current_hour = (pd.Timestamp.now() - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)

    df = fg.filter((fg.pickup_hour >= current_hour)).read()

    return df

