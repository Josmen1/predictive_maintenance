import os
import sys
import pandas as pd
import numpy as np

"""
Defining constants for training pipeline
"""
TARGET_COLUMN: str = "RUL"
PIPELINE_NAME: str = "predictive_maintenance"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

FINISHED_TRAIN_FILE_NAME: str = "train_finished.csv"
FINISHED_TEST_FILE_NAME: str = "test_finished.csv"

"""
Data ingestion related constants
"""
TRAIN_DATA_INGESTION_COLLECTION_NAME: str = "train_predictor"
TEST_DATA_INGESTION_COLLECTION_NAME: str = "test_predictor"
DATA_INGESTION_DATABASE_NAME: str = "Menge_predictor"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
