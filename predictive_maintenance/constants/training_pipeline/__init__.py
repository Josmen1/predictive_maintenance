import os
import sys
import pandas as pd
import numpy as np

"""
Initial constants for C-MAPSS dataset preprocessing. 
These constants will be used throughout the initial stages 
of extracting and preparing the data into a combined dataframe.
Both for training and testing datasets.
"""
TARGET_COLUMN = "RUL"
BASE_COLUMN_NAMES = ["unit_number", "time_in_cycles", "ops_1", "ops_2", "ops_3"]
SENSOR_COLUMN_NAMES = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMN_NAMES = BASE_COLUMN_NAMES + SENSOR_COLUMN_NAMES + ["subset"] + ["split"]
SUBSET_IDS = [f"FD00{i}" for i in range(1, 5)]
OPS_COLUMNS = ["ops_1", "ops_2", "ops_3"]
DRIFT_COLUMNS = OPS_COLUMNS + SENSOR_COLUMN_NAMES

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
