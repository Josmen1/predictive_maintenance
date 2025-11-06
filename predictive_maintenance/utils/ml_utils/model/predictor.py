from predictive_maintenance.exception.exception import PredictiveMaintenanceException
import sys
import os
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)
from predictive_maintenance.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME


class ModelPredictor:
    def __init__(self, preprocessor, model):
        try:
            log.info("Initializing ModelPredictor.")
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
        
    def predict(self, X):
        try:
            log.info("Starting prediction process.")
            X_processed = self.preprocessor.transform(X)
            predictions = self.model.predict(X_processed)
            log.info("Prediction process completed successfully.")
            return predictions
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
