from datetime import datetime, timedelta

import pandas as pd

import src.config as config
from src.inference import (
    get_feature_store,
    get_model_predictions,
    load_model_from_registry,
)

current_date = pd.Timestamp.now()
feature_store = get_feature_store()

fetch_data_to = current_date
fetch_data_from = current_date - timedelta(days=29)
fetch_data_from = pd.to_datetime(fetch_data_from).floor('h').tz_localize("Etc/UTC")
fetch_data_to = pd.to_datetime(fetch_data_to).floor('h').tz_localize("Etc/UTC")


print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
)

ts_data = feature_view.get_batch_data(
    start_time=(fetch_data_from - timedelta(days=1)),
    end_time=(fetch_data_to + timedelta(days=1)),
)
ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
ts_data.sort_values(["station_id", "pickup_hour"]).reset_index(drop=True)
ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)

from src.data_utils import transform_ts_data_info_features_and_target

features, _ = transform_ts_data_info_features_and_target(ts_data, window_size=28, step_size=1)

model = load_model_from_registry()

predictions = get_model_predictions(model, features)
predictions["pickup_hour"] = current_date.ceil("h")
print(predictions)

feature_group = get_feature_store().get_or_create_feature_group(
    name=config.FEATURE_GROUP_MODEL_PREDICTION,
    version=1,
    description="Predictions from LGBM Model",
    primary_key=["station_id", "pickup_hour"],
    event_time="pickup_hour",
)

feature_group.insert(predictions, write_options={"wait_for_job": False})
