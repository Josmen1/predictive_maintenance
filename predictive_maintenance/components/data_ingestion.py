# Import necessary libraries
import pandas as pd
import numpy as np
import os
import sys
import pymongo

from typing import List, Tuple, Optional

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

    """
    """

    def fetch_data_from_collection(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Minimal-change version:
        - single Mongo find
        - split with 'split' field
        - for RUL: drop NaNs (columns that are all-NaN) and, if possible,
            reduce to exactly [subset_col, value_col]; otherwise keep the
            remaining non-all-NaN columns.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            collection = self.client[database_name][collection_name]

            # One fetch; skip _id to avoid later drops
            df = pd.DataFrame(list(collection.find({}, {"_id": 0})))

            if df.empty:
                log.warning(f"No data found in collection: {collection_name}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # Train/Test masks
            if "split" in df.columns:
                split_norm = df["split"].astype(str).str.strip().str.lower()
                train_mask = split_norm.eq("train")
                test_mask = split_norm.eq("test")
            else:
                train_mask = pd.Series(False, index=df.index)
                test_mask = pd.Series(False, index=df.index)

            # Train/Test frames (drop routing field)
            train_df = df.loc[train_mask].reset_index(drop=True)
            test_df = df.loc[test_mask].reset_index(drop=True)

            # RUL = anything not train/test
            rul_raw = df.loc[~train_mask & ~test_mask].drop(
                columns=["split"], errors="ignore"
            )

            # If nothing falls in RUL, return empty
            if rul_raw.empty:
                rul_df = pd.DataFrame()
            else:
                # First, drop columns that are entirely NaN within RUL slice
                rul_trim = rul_raw.dropna(axis=1, how="all")

                # Try to keep only the desired two columns if we can detect them
                value_col = getattr(self.data_ingestion_config, "rul_value_col", None)
                subset_col = getattr(self.data_ingestion_config, "rul_subset_col", None)

                # Lightweight autodetect only if not configured (no heavy heuristics)

                if value_col is None:
                    for cand in ["RUL", "rul", "remaining_useful_life", "value"]:
                        if cand in rul_trim.columns:
                            value_col = cand
                            break
                if subset_col is None:
                    for cand in ["subset", "subset_id", "unit_id", "id"]:
                        if cand in rul_trim.columns:
                            subset_col = cand
                            break
                # ...
                if value_col in rul_trim.columns and subset_col in rul_trim.columns:
                    # value first (e.g., "RUL"), then subset
                    rul_df = (
                        rul_trim[[value_col, subset_col]]
                        .dropna(subset=[value_col, subset_col])
                        .reset_index(drop=True)
                    )
                else:
                    # Fall back: return trimmed RUL with NaN-only columns removed
                    if value_col or subset_col:
                        log.warning(
                            f"Could not find both RUL columns (subset={subset_col}, value={value_col}). "
                            f"Returning RUL with NaN-only columns removed."
                        )
                    rul_df = rul_trim.reset_index(drop=True)
                # ...

            log.info(
                f"Fetched {len(df)} records from '{collection_name}' -> "
                f"train={len(train_df)}, test={len(test_df)}, RUL={len(rul_df)}"
            )
            if train_df.empty:
                log.warning("Train split is empty.")
            if test_df.empty:
                log.warning("Test split is empty.")
            if rul_df.empty:
                log.warning("RUL set is empty.")

            return train_df, test_df, rul_df

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
            train_df, test_df, rul_df = self.fetch_data_from_collection()
            log.info("Data ingestion completed successfully.")
            # Export data to feature store
            self.export_data_to_feature_store(
                train_df, self.data_ingestion_config.train_feature_store_file_path
            )
            self.export_data_to_feature_store(
                test_df, self.data_ingestion_config.test_feature_store_file_path
            )
            self.export_data_to_feature_store(
                rul_df, self.data_ingestion_config.rul_feature_store_file_path
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
            self.save_data_to_ingested_files(
                rul_df, self.data_ingestion_config.rul_file_path
            )
            # return DataIngestionArtifact
            data_ingestion_artifact = DataIngestionArtifact(
                training_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path,
                rul_file_path=self.data_ingestion_config.rul_file_path,
            )
            return data_ingestion_artifact
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
