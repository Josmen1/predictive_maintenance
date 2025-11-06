import sys
import numpy as np
import pandas as pd
import os


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)

from predictive_maintenance.entity.artifact_entity import ModelEvaluationArtifact

# calculate regression metrics
def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelEvaluationArtifact:
    try:
        log.info("Calculating regression metrics.")
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        log.info(
            f"Regression metrics calculated: MAE={mae}, MSE={mse}, RMSE={rmse}, MAPE={mape}"
        )

        return ModelEvaluationArtifact(
            mean_absolute_error=mae,
            mean_squared_error=mse,
            root_mean_squared_error=rmse,
            mean_absolute_percentage_error=mape,
        )
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)