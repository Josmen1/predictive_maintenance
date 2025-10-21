# Import necessary libraries
import pandas as pd
import numpy as np
import os
import sys
import pymongo

from typing import List

# import exception file
from predictive_maintenance.exception.exception import PredictiveMaintenanceException

# import logger
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)
# import config entity
from predictive_maintenance.entity.config_entity import DataIngestionConfig

# import artifact entity
from predictive_maintenance.entity.artifact_entity import DataIngestionArtifact

# import dotenv to read env variables
from dotenv import load_dotenv

load_dotenv()

# Read the MONGO_DB_URL from environment variables
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    """Class for data ingestion from MongoDB.
    This class connects to a MongoDB database and provides methods to fetch data
    from specified collections. The fetched data can be used for further processing in the
    predictive maintenance pipeline.
    Attributes:
        data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
        client (pymongo.MongoClient): MongoDB client for database connection.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.client = pymongo.MongoClient(MONGO_DB_URL)
            log.info("Connected to MongoDB successfully for data ingestion.")
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def fetch_data_from_collection(self) -> pd.DataFrame:
        """Fetches data from the specified MongoDB collection and returns it as a DataFrame.
        Args:
            collection_name (str): Name of the MongoDB collection to fetch data from.
        Returns:
            pd.DataFrame: DataFrame containing the data fetched from the collection.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            train_collection_name = self.data_ingestion_config.train_collection_name
            test_collection_name = self.data_ingestion_config.test_collection_name
            train_collection = self.client[database_name][train_collection_name]
            test_collection = self.client[database_name][test_collection_name]
            # Fetch data from train collection
            train_data = list(train_collection.find())
            test_data = list(test_collection.find())
            # Convert to DataFrames
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            # Check if data is empty
            if len(train_df) == 0:
                log.warning(f"No data found in collection: {train_collection_name}")
                return pd.DataFrame()
            if len(test_df) == 0:
                log.warning(f"No data found in collection: {test_collection_name}")
                return pd.DataFrame()
            # Remove the MongoDB specific '_id' field if it exists
            if "_id" in train_df.columns.to_list():
                train_df.drop("_id", axis=1, inplace=True)
            log.info(
                f"Fetched {len(train_df)} records from collection: {train_collection_name}"
            )
            if "_id" in test_df.columns.to_list():
                test_df.drop("_id", axis=1, inplace=True)
            return train_df, test_df
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def export_data_to_feature_store(self, df: pd.DataFrame, file_path: str):
        """Exports the given DataFrame to a CSV file at the specified file path.
        Args:
            df (pd.DataFrame): DataFrame to be exported.
            file_path (str): File path where the CSV will be saved.
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(file_path, index=False, header=True)
            log.info(f"Data exported to feature store at: {file_path}")
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    # A function to save a df to train and test files
    def save_data_to_ingested_files(self, df: pd.DataFrame, file_path: str):
        """Saves the given DataFrame to the specified file path.
        Args:
            df (pd.DataFrame): DataFrame to be saved.
            file_path (str): File path where the DataFrame will be saved.
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(file_path, index=False, header=True)
            log.info(f"Data saved to ingested files at: {file_path}")
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def initiate_data_ingestion(self) -> List[pd.DataFrame]:
        """Initiates the data ingestion process by fetching data from train and test collections.
        Returns:
            List[pd.DataFrame]: A list containing DataFrames for training and testing data.
        """
        try:
            train_df, test_df = self.fetch_data_from_collection()
            log.info("Data ingestion completed successfully.")
            # Export data to feature store
            self.export_data_to_feature_store(
                train_df, self.data_ingestion_config.train_feature_store_file_path
            )
            self.export_data_to_feature_store(
                test_df, self.data_ingestion_config.test_feature_store_file_path
            )
            # We are not splitting the data here as we are fetching already split data
            # from collections, so we'll straightaway save the dataframes (similar to those sent to the feature store)
            # to the ingested files.
            self.save_data_to_ingested_files(
                train_df, self.data_ingestion_config.training_file_path
            )
            self.save_data_to_ingested_files(
                test_df, self.data_ingestion_config.testing_file_path
            )
            # return DataIngestionArtifact
            data_ingestion_artifact = DataIngestionArtifact(
                training_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path,
            )
            return data_ingestion_artifact
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
