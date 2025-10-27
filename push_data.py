import os
import sys
import json
import pymongo
import pandas as pd
from dotenv import load_dotenv
from bson import json_util

from pymongo.write_concern import WriteConcern
from pymongo.errors import BulkWriteError, AutoReconnect, NetworkTimeout
import random
import time
from time import sleep

load_dotenv()

from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.utils.main_utils.pre_ingestion_fns import (
    load_data,
    add_column_names,
    save_combined_data_to_csv,
)
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)

from predictive_maintenance.constants import training_pipeline


MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if MONGO_DB_URL is None:
    sys.exit("Error: MONGO_DB_URL is not set in the environment variables.")
print(f"MONGO_DB_URL: {MONGO_DB_URL}")

import certifi

ca = certifi.where()


# Helper function to chunk a list into smaller lists of a specified size
# Used for batch insertion into MongoDB
def _chunk(lst, n):
    """
    Yield successive n-sized chunks from lst.
    Parameters
    ----------
    lst : list
        The list to chunk.
    n : int
        The size of each chunk.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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
            # If you’re on Atlas and your URI lacks tls=true, uncomment tls args.
            self.mongo_client = pymongo.MongoClient(
                mongo_db_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=120000,
                retryWrites=True,
                retryReads=True,
                maxPoolSize=20,
                # tls=True, tlsCAFile=ca,
            )
            # Fail fast if unreachable
            self.mongo_client.admin.command("ping")
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

    def push_data_to_mongo(
        self, records, batch_size=1000, max_retries=5, base_backoff=0.5
    ):
        """Push data records to MongoDB in batches with retry logic for transient errors.
        This function handles AutoReconnect and NetworkTimeout exceptions by retrying
        the insertion with exponential backoff. It also handles duplicate key errors.
        Parameters
        ----------
        records : list
            The list of records to insert into MongoDB.
        batch_size : int
            The number of records to insert in each batch.
        max_retries : int
            The maximum number of retry attempts for transient errors.
        base_backoff : float
            The base backoff time (in seconds) for retrying failed inserts.
        """
        try:
            # Check if records is empty
            if not records:
                log.warning("No records to push to MongoDB.")
                return 0
            # Get the collection
            db = self.mongo_client[self.database_name]
            # If you need durability later, bump to majority; for bulk loads w=1 is faster.
            coll = db.get_collection(
                self.collection_name, write_concern=WriteConcern(w=1)
            )
            # Insert records in batches with retry logic
            # The following loop will handle transient errors and retry the insertion.
            # Keep track of total inserted records
            inserted_total = 0
            for batch in _chunk(records, batch_size):
                attempt = 0
                while True:
                    try:
                        res = coll.insert_many(batch, ordered=False)
                        inserted_total += len(res.inserted_ids)
                        break
                    except (AutoReconnect, NetworkTimeout) as e:
                        if attempt >= max_retries:
                            raise
                        sleep_for = (base_backoff * (2**attempt)) + random.uniform(
                            0, 0.25
                        )
                        log.warning(
                            f"Transient Mongo error ({type(e).__name__}). Retry {attempt+1}/{max_retries} in {sleep_for:.2f}s."
                        )
                        time.sleep(sleep_for)
                        attempt += 1
                    except BulkWriteError as bwe:
                        details = bwe.details or {}
                        n_ok = details.get("nInserted", 0)
                        inserted_total += n_ok
                        # If all errors are duplicates, proceed; else re-raise.
                        dupes = [
                            w
                            for w in details.get("writeErrors", [])
                            if w.get("code") == 11000
                        ]
                        if dupes and len(dupes) == len(details.get("writeErrors", [])):
                            log.info(f"Skipped {len(dupes)} duplicates in batch.")
                            break
                        raise
            log.info(f"Data successfully pushed: {inserted_total} docs.")
            return inserted_total
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)


if __name__ == "__main__":
    DATABASE_NAME = training_pipeline.DATA_INGESTION_DATABASE_NAME
    COLLECTION_NAME = training_pipeline.COLLECTION_NAME
    try:
        log.info("Starting data extraction and processing...")
        extractor = PredictiveMaintenanceDataExtractor(
            records=None,
            database_name=DATABASE_NAME,
            collection_name=COLLECTION_NAME,
        )
        df_train = load_data(
            "Prediction_Dataset/Turbofan_Engine_data_Set", "train_FD00", "train"
        )
        if df_train is not None:
            log.info(f"Loaded data shape: {df_train.shape}")
            train_df = add_column_names(df_train)
            train_output_path = save_combined_data_to_csv(
                train_df,
                subfolder_name="train",
                output_file_name="train.csv",
            )
            log.info(f"Processed data saved at: {train_output_path}")

        df_test = load_data(
            "Prediction_Dataset/Turbofan_Engine_data_Set", "test_FD00", "test"
        )
        if df_test is not None:
            log.info(f"Loaded data shape: {df_test.shape}")
            test_df = add_column_names(df_test)
            test_output_path = save_combined_data_to_csv(
                test_df,
                subfolder_name="test",
                output_file_name="test.csv",
            )
            log.info(f"Processed data saved at: {test_output_path}")

        df_rul = load_data("Prediction_Dataset/Turbofan_Engine_data_Set", "RUL_FD00")
        if df_rul is not None:
            log.info(f"Loaded RUL data shape: {df_rul.shape}")
            rul_output_path = save_combined_data_to_csv(
                df_rul,
                subfolder_name="RUL",
                output_file_name="RUL.csv",
            )

        log.info("Dataset initial preparation and processing completed successfully.")
        train_records = extractor.csv_to_json(file_path=train_output_path)
        test_records = extractor.csv_to_json(file_path=test_output_path)
        rul_records = extractor.csv_to_json(file_path=rul_output_path)
        log.info("Data conversion to JSON completed successfully.")
        # Push data to MongoDB
        # Since we have both train, test and RUL records identifiable through a 'split' column,
        # we can push them into the same collection but for clarity, we will use separate collections

        no_of_train_records = extractor.push_data_to_mongo(records=train_records)
        no_of_test_records = extractor.push_data_to_mongo(records=test_records)
        no_of_rul_records = extractor.push_data_to_mongo(records=rul_records)
        # print the first five records
        # Using json_util.dumps for better formatting
        # Json_util helps in serializing MongoDB specific data types removing the mongo_db id field

        print("First five training records:")
        print(json_util.dumps(train_records[:5], indent=4))
        print("First five testing records:")
        print(json_util.dumps(test_records[:5], indent=4))
        print("First five RUL records:")
        print(json_util.dumps(rul_records[:5], indent=4))
        # Print the number of records inserted
        print(
            f"Number of training records inserted: {no_of_train_records}, Number of testing records inserted: {no_of_test_records}, Number of RUL records inserted: {no_of_rul_records}"
        )

    except Exception as e:
        log.error(f"An error occurred: {e}")
        raise PredictiveMaintenanceException(e, sys)
