import os
import sys
import json
import pymongo
import pandas as pd
from dotenv import load_dotenv
from bson import json_util

load_dotenv()

from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.utils.file_functions import (
    load_data,
    add_column_names_and_drop_constant_columns,
    save_combined_data_to_csv,
)
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)


MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if MONGO_DB_URL is None:
    sys.exit("Error: MONGO_DB_URL is not set in the environment variables.")
print(f"MONGO_DB_URL: {MONGO_DB_URL}")

import certifi

ca = certifi.where()


class PredictiveMaintenanceDataExtractor:
    def __init__(
        self,
        records,
        database_name: str,
        collection_name: str,
        mongo_db_url=MONGO_DB_URL,
    ):
        self.records = records
        self.database_name = database_name
        self.collection_name = collection_name
        try:
            self.mongo_client = pymongo.MongoClient(mongo_db_url)
            log.info("Successfully connected to MongoDB.")
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def get_collection_as_dataframe(self, database_name: str, collection_name: str):
        try:
            log.info(
                f"Fetching data from database: {database_name}, collection: {collection_name}"
            )
            database = self.mongo_client[database_name]
            collection = database[collection_name]
            data = list(collection.find())
            if len(data) == 0:
                log.warning("No data found in the collection.")
                return None

            df = pd.DataFrame(data)
            log.info(f"Data fetched successfully with shape: {df.shape}")
            return df
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def csv_to_json(self, file_path: str):
        try:
            log.info(f"Converting CSV data to JSON and saving to {file_path}")
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = json.loads(data.to_json(orient="records"))
            log.info("Data successfully saved to JSON.")
            return records
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def push_data_to_mongo(self, records):
        try:
            if not records:
                log.warning("No records to push to MongoDB.")
                return
            database = self.mongo_client[self.database_name]
            collection = database[self.collection_name]
            log.info(
                f"Pushing data to database: {self.database_name}, collection: {self.collection_name}"
            )
            collection.insert_many(records)
            log.info("Data successfully pushed to MongoDB.")
            return len(records)
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)


if __name__ == "__main__":
    DATABASE_NAME = "Menge_predictor"
    COLLECTION_NAME = "okioma_predictor"
    try:
        log.info("Starting data extraction and processing...")
        extractor = PredictiveMaintenanceDataExtractor(
            records=None,
            database_name=DATABASE_NAME,
            collection_name=None,
        )
        df_train = load_data(
            "Prediction_Dataset/Turbofan_Engine_data_Set", "train_FD00"
        )
        if df_train is not None:
            log.info(f"Loaded data shape: {df_train.shape}")
            train_df = add_column_names_and_drop_constant_columns(df_train)
            train_output_path = save_combined_data_to_csv(
                train_df,
                subfolder_name="train",
                output_file_name="train.csv",
            )
            log.info(f"Processed data saved at: {train_output_path}")

        df_test = load_data("Prediction_Dataset/Turbofan_Engine_data_Set", "test_FD00")
        if df_test is not None:
            log.info(f"Loaded data shape: {df_test.shape}")
            test_df = add_column_names_and_drop_constant_columns(df_test)
            test_output_path = save_combined_data_to_csv(
                test_df,
                subfolder_name="test",
                output_file_name="test.csv",
            )
            log.info(f"Processed data saved at: {test_output_path}")

        log.info("Dataset initial preparation and processing completed successfully.")
        train_records = extractor.csv_to_json(file_path=train_output_path)
        test_records = extractor.csv_to_json(file_path=test_output_path)
        log.info("Data conversion to JSON completed successfully.")
        # Push data to MongoDB
        # Since we are pushing data to separate collections for train and test,
        # we need to dynamically set the collection name before each push
        extractor.collection_name = "train_predictor"
        no_of_train_records = extractor.push_data_to_mongo(records=train_records)
        extractor.collection_name = "test_predictor"
        no_of_test_records = extractor.push_data_to_mongo(records=test_records)
        # print the first five records
        # Using json_util.dumps for better formatting
        # Json_util helps in serializing MongoDB specific data types removing the mongo_db id field

        print("First five training records:")
        print(json_util.dumps(train_records[:5], indent=4))
        print("First five testing records:")
        print(json_util.dumps(test_records[:5], indent=4))
        # Print the number of records inserted
        print(
            f"Number of training records inserted: {no_of_train_records}, Number of testing records inserted: {no_of_test_records}"
        )

    except Exception as e:
        log.error(f"An error occurred: {e}")
        raise PredictiveMaintenanceException(e, sys)
