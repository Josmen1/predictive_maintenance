import os
import sys
import pandas as pd
import numpy as np
import yaml

# from scipy import ks_2samp


from predictive_maintenance.entity.config_entity import TrainingPipelineConfig
from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.entity.config_entity import DataValidationConfig
from predictive_maintenance.entity.artifact_entity import (
    DataValidationArtifact,
    DataIngestionArtifact,
)

from predictive_maintenance.utils.main_utils.validation_fn import (
    validate_cmapss,
    run_drift_check_and_save,
    read_data_schema_yaml,
    ensure_parent_dir,
    safe_write_csv,
    safe_write_yaml,
)
from predictive_maintenance.constants import training_pipeline

from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)


class DataValidation:
    """Data Validation component for Predictive Maintenance Pipeline.
    This component is responsible for validating the ingested data against
    a predefined schema, checking for missing values, and detecting data drift.
    Attributes
    ----------
    data_validation_config : DataValidationConfig
        Configuration for data validation.
    data_ingestion_artifact : DataIngestionArtifact
        Artifact from data ingestion containing paths to ingested data.
    schema_file_path : str
        Path to the data schema YAML file.
    schema_info : dict
        Dictionary containing schema information read from the YAML file.
    Methods
    -------
    __init__(self, data_validation_config: DataValidationConfig,
             data_ingestion_artifact: DataIngestionArtifact)
        Initializes the DataValidation component with configuration and artifacts.
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            # call function to create schema file path
            self.train_schema_file_path = training_pipeline.TRAIN_SCHEMA_FILE_PATH
            self.train_schema_info = read_data_schema_yaml(self.train_schema_file_path)
            self.test_schema_file_path = training_pipeline.TEST_SCHEMA_FILE_PATH
            self.test_schema_info = read_data_schema_yaml(self.test_schema_file_path)

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys) from e

    # create a static method to read data
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads data from a CSV file and returns a pandas DataFrame.
        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        Returns
        -------
        pd.DataFrame
            DataFrame containing the data read from the CSV file.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    # function to detect data drift
    def confirm_cols(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, RUL_df: pd.DataFrame
    ) -> bool:
        res = validate_cmapss(train_df, test_df, RUL_df)
        print(res["status"])
        print(res["errors"])
        print(res["warnings"])
        if res["status"] != "PASS":
            log.info(f"Data validation errors: {res['errors']}")
            log.info(f"Data validation warnings: {res['warnings']}")
            return False
        else:
            return True
        # return res["status"] == "PASS"

    # create a method to initiate data validation

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Initiates the data validation process.
        This method performs data validation by checking the ingested data
        against the predefined schema, checking for missing values, and
        detecting data drift. It generates a DataValidationArtifact
        containing the results of the validation.
        Returns
        -------
        DataValidationArtifact
            Artifact containing the results of data validation.
        """
        try:
            log.info("Starting data validation process.")
            train_file_path = self.data_ingestion_artifact.training_file_path
            test_file_path = self.data_ingestion_artifact.testing_file_path
            rul_file_path = self.data_ingestion_artifact.rul_file_path
            # Load the data post ingestion
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)
            rul_df = DataValidation.read_data(rul_file_path)
            # Implement data validation logic here
            # For example: validate schema, check missing values, detect drift
            status = self.confirm_cols(train_df, test_df, rul_df)
            if not status:
                error_message = (
                    "Data validation failed. Please check the logs for details."
                )
                log.error(error_message)
                # To hard stop the pipeline if validation fails
                # raise PredictiveMaintenanceException(error_message, sys)
            # If validation passes, generate drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            ensure_parent_dir(drift_report_file_path)

            drift_report = run_drift_check_and_save(
                reference_df=train_df,
                current_df=test_df,
                output_path=drift_report_file_path,
            )
            log.info("Data drift report generated successfully.")
            # Persist the valid datasets
            valid_train_path = self.data_validation_config.valid_train_file_path
            valid_test_path = self.data_validation_config.valid_test_file_path
            valid_rul_path = self.data_validation_config.valid_rul_file_path

            # ensure_parent_dir(valid_train_path)
            # ensure_parent_dir(valid_test_path)
            # ensure_parent_dir(valid_rul_path)
            # Create directories if they don't exist and write atomically
            safe_write_csv(train_df, valid_train_path)
            safe_write_csv(test_df, valid_test_path)
            safe_write_csv(rul_df, valid_rul_path)
            log.info("Valid datasets saved successfully.")

            # After validation, create and return DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=valid_train_path,
                valid_test_file_path=valid_test_path,
                valid_rul_file_path=valid_rul_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                invalid_rul_file_path=None,
                drift_report_file_path=drift_report_file_path,
                # train_drift_report_file_path=self.data_validation_config.train_drift_report_file_path,
                # test_drift_report_file_path=self.data_validation_config.test_drift_report_file_path,
                # rul_drift_report_file_path=self.data_validation_config.rul_drift_report_file_path,
            )

            log.info("Data validation process completed successfully.")
            return data_validation_artifact

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys) from e
