import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "Dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"


HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

FEATURE_GROUP_NAME = "time_series_6_hour_feature_group"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "time_series_6_hour_feature_view"
FEATURE_VIEW_VERSION = 1


MODEL_NAME = "bike_demand_predictor_next_hour"
MODEL_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTION = "bike_6_hour_model_prediction"