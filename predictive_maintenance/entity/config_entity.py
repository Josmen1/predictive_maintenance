import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from predictive_maintenance.constants import training_pipeline
from predictive_maintenance.exception.exception import PredictiveMaintenanceException


# Create a training pileline config class
class TrainingPipelineConfig:
    """Configuration for the training pipeline, including artifact directory setup.
    This class initializes the artifact directory based on the current timestamp.
    Attributes:
        pipeline_name (str): Name of the training pipeline.
        artifact_name (str): Name of the artifact directory.
        artifact_dir (str): Full path to the artifact directory with timestamp.
        timestamp (str): Timestamp string for the artifact directory.
    Here an artifact refers to any output or result produced during the training process,
    such as trained models, evaluation reports, or processed data.
    """

    def __init__(self, timestamp: str = datetime.now()):
        try:
            timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
            self.pipeline_name = training_pipeline.PIPELINE_NAME
            self.artifact_name = training_pipeline.ARTIFACT_DIR
            self.artifact_dir = os.path.join(self.artifact_name, timestamp)
            self.timestamp = timestamp
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)


class DataIngestionConfig:
    """Configuration for data ingestion process.
    This class sets up the necessary parameters for ingesting data,
    including database and collection names, as well as directory paths
    for storing ingested data.
    Attributes:
        data_ingestion_collection_name (str): Name of the data collection.
        data_ingestion_database_name (str): Name of the database.
        data_ingestion_dir (str): Directory for data ingestion artifacts.
        feature_store_dir (str): Directory for storing feature store.
        ingested_dir (str): Directory for storing ingested data.
        train_test_split_ratio (float): Ratio for splitting train and test data.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.train_collection_name = training_pipeline.TRAIN_DATA_INGESTION_COLLECTION_NAME
            self.test_collection_name = training_pipeline.TEST_DATA_INGESTION_COLLECTION_NAME
            self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
            self.data_ingestion_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                training_pipeline.DATA_INGESTION_DIR_NAME,
            )
            self.train_feature_store_file_path = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
                training_pipeline.TRAIN_FILE_NAME,
            )
            self.test_feature_store_file_path = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
                training_pipeline.TEST_FILE_NAME,
            )
            self.training_file_path = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_INGESTED_DIR,
                training_pipeline.FINISHED_TRAIN_FILE_NAME,
            )
            self.testing_file_path = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_INGESTED_DIR,
                training_pipeline.FINISHED_TEST_FILE_NAME,
            )
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
