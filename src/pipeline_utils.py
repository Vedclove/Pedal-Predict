import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# Function to return the pipeline
def get_pipeline(**hyper_params):
    """
    Returns a pipeline with optional parameters for LGBMRegressor.

    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with feature engineering and LGBMRegressor.
    """
    pipeline = make_pipeline(
        lgb.LGBMRegressor(**hyper_params),  # Pass optional parameters here
    )
    return pipeline