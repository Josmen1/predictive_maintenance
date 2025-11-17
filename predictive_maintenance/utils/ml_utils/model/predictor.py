from predictive_maintenance.exception.exception import PredictiveMaintenanceException
import sys
import numpy as np
import pandas as pd
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)


class ModelPredictor:
    def __init__(self, preprocessor, model):
        try:
            log.info("Initializing ModelPredictor.")
            self.preprocessor = preprocessor
            self.model = model
            # Common ID/display cols (do NOT assume they are never used by the preprocessor)
            # self.id_cols = ["unit_number", "time_in_cycles", "subset", "split"]
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            log.info("Transforming input data using preprocessor")
            return self.preprocessor.transform(X)
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            log.info("Generating predictions using the model")
            return self.model.predict(X)
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
    