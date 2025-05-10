import pandas as pd

import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.data_utils import transform_ts_data_info_features_and_target, fetch_batch_raw_data
from src.inference import (
    fetch_days_data,
    get_hopsworks_project,
    load_metrics_from_registry,
    load_model_from_registry,
)
from src.pipeline_utils import get_pipeline
import time

print(f"Fetching data from group store ...")
date_from = pd.Timestamp.now()
date_to = date_from - pd.Timedelta(days=180)
ts_data = fetch_batch_raw_data(date_from, date_to)

print(f"Transforming to ts_data ...")

try:
    features, targets = transform_ts_data_info_features_and_target(
        ts_data, window_size=24 * 28, step_size=1
    )
    pipeline = get_pipeline()
    print(f"Training model ...")
    print("Waiting for data to load")
    pipeline.fit(features, targets)

    predictions = pipeline.predict(features)

    test_mae = mean_absolute_error(targets, predictions)
    metric = load_metrics_from_registry()

    print(f"The new MAE is {test_mae:.4f}")
    print(f"The previous MAE is {metric['test_mae']:.4f}")

    if test_mae < metric.get("test_mae"):
        print(f"Registering new model")
        model_path = config.MODELS_DIR / "lgb_model.pkl"
        joblib.dump(pipeline, model_path)

        input_schema = Schema(features)
        output_schema = Schema(targets)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        project = get_hopsworks_project()
        model_registry = project.get_model_registry()

        model = model_registry.sklearn.create_model(
            name="taxi_demand_predictor_next_hour",
            metrics={"test_mae": test_mae},
            input_example=features.sample(),
            model_schema=model_schema,
        )
        model.save(model_path)
    else:
        print(f"Skipping model registration because new model is not better!")
except ValueError as e:
    print(f"Skipping model training: {str(e)}")