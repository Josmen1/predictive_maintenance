import os
import sys
import pandas as pd
import numpy as np
from typing import Optional

"""
Initial constants for C-MAPSS dataset preprocessing. 
These constants will be used throughout the initial stages 
of extracting and preparing the data into a combined dataframe.
Both for training and testing datasets.
"""
TARGET_COLUMN = "RUL"
ID_COLUMNS = ["unit_number", "time_in_cycles", "subset", "split"]
OPS_COLUMNS = ["ops_1", "ops_2", "ops_3"]
SENSOR_COLUMN_NAMES = [f"sensor_{i}" for i in range(1, 22)]
BASE_FEATURE_COLUMNS = OPS_COLUMNS + SENSOR_COLUMN_NAMES
ALL_COLUMN_NAMES = ID_COLUMNS + BASE_FEATURE_COLUMNS + ["subset"] + ["split"]
SUBSET_IDS = [f"FD00{i}" for i in range(1, 5)]
GROUP_COLUMN = "unit_number"
ZN_SUFFIX = "_zn"
PROTECT_COLUMNS = ID_COLUMNS + [TARGET_COLUMN, "subset"]


PREPROCESSING_OBJECT_FILE_NAME = "preprocessing_object.joblib"
"""
Defining constants for training pipeline
"""
TARGET_COLUMN: str = "RUL"
PIPELINE_NAME: str = "predictive_maintenance"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
RUL_FILE_NAME = "RUL.csv"

FINISHED_TRAIN_FILE_NAME: str = "train_finished.csv"
FINISHED_TEST_FILE_NAME: str = "test_finished.csv"

TRAIN_SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")
TEST_SCHEMA_FILE_PATH: str = os.path.join("data_schema", "test_schema.yaml")

SAVED_MODEL_DIR: str = "saved_models"
MODEL_FILE_NAME: str = "model.joblib"

"""
Data ingestion related constants
"""
COLLECTION_NAME: str = "okioma_predictor"
DATA_INGESTION_DATABASE_NAME: str = "Menge_predictor"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data validation related constants
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid_data"
DATA_VALIDATION_INVALID_DIR: str = "invalid_data"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"
# TRAIN_DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "train_drift_report.yaml"
# TEST_DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "test_drift_report.yaml"
# RUL_DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "rul_drift_report.yaml"

"""
Data transformation related constants
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
TRANSFORMED_DATA_DIR: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_CORRELATION_THRESHOLD: float = 0.995

"""Model trainer related constants"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_FILE_NAME: str = "trained_model.joblib"
MODEL_TRAINER_CV_RESULTS_DIR: str = "cv_results"
MODEL_TRAINER_CV_FILE_NAME: str = "cv_results.csv"
TEST_PREDICTIONS_DIR: str = "test_predictions"
TEST_PREDICTIONS_FILE_NAME: str = "test_predictions.csv"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.1
MODEL_TRAINER_METRICS_DIR: str = "metrics"
MODEL_TRAINER_METRICS_FILE_NAME: str = "metrics.json"
COMBINED_OBJECT_FILE_NAME: str = "combined_model_predictor.joblib"

# --- Model search options ---
MODEL_SEARCH_OPTIONS = {
    "n_splits": 5,
    "random_state": 12,
    "n_jobs": 1,
    "enable_xgboost": True,
    "enable_lightgbm": True,
    "enable_random_forest": True,
    "enable_adaboost": True,
}
"""MLflow related constants"""
MLFLOW_TRACKING_URI: Optional[str] = (
    None  # Set to None for default local file-based tracking
)
MLFLOW_EXPERIMENT_NAME: str = "predictive_maintenance"
MLFLOW_AUTOLOG: bool = True
MLFLOW_LOG_MODELS: bool = True
MLFLOW_TAGS: dict[str, str] = {"project": "Predictive Maintenance", "owner": "Menge"}
