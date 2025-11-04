import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import List, Dict, Tuple, Optional

from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.constants.training_pipeline import (
    TARGET_COLUMN,
    ID_COLUMNS,
    SENSOR_COLUMN_NAMES,
    BASE_FEATURE_COLUMNS,
    ZN_SUFFIX,
    DATA_TRANSFORMATION_CORRELATION_THRESHOLD,
)
from predictive_maintenance.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from predictive_maintenance.entity.config_entity import DataTransformationConfig

from predictive_maintenance.logging import logger
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)
from predictive_maintenance.utils.main_utils.data_transformation_fns import (
    save_object,
    save_numpy_array_data,
    add_train_rul,
    add_test_rul,
    save_transformed_data_as_csv,
)
from predictive_maintenance.utils.transformer import (
    ZeroVarianceDropper,
    SubsetNormalizer,
    TemporalFeatureEngineer,
    DropOriginalWithZN,
    CorrelationFilter,
    FinalFrameAssembler,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            log.info(
                "Initializing Data Transformation Component with configuration and validation artifact."
            )
            self.data_validation_artifact: DataValidationArtifact = (
                data_validation_artifact
            )
            self.data_transformation_config: DataTransformationConfig = (
                data_transformation_config
            )

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            log.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    # preprocessor_sklearn.py (continued)

    def build_preprocessor_pipeline(
        self,
        id_cols: List[str],
        sensor_cols: List[str],
        base_feature_cols: List[str],
        target_col: str,
        zn_suffix: str,
        corr_threshold: float,
    ):
        return Pipeline(
            steps=[
                (
                    "zv",
                    ZeroVarianceDropper(cols=base_feature_cols, subset_col="subset"),
                ),
                (
                    "norm",
                    SubsetNormalizer(
                        feature_cols=base_feature_cols,
                        subset_col="subset",
                        suffix=zn_suffix,
                    ),
                ),
                (
                    "temporal",
                    TemporalFeatureEngineer(
                        sensor_cols=sensor_cols, windows=(5, 20), suffix_in=zn_suffix
                    ),
                ),
                (
                    "drop_orig",
                    DropOriginalWithZN(
                        base_cols=base_feature_cols, suffix_in=zn_suffix
                    ),
                ),
                (
                    "corr",
                    CorrelationFilter(
                        threshold=corr_threshold,
                        protect=id_cols + ["subset", target_col],
                        verbose=True,
                    ),
                ),
                ("final", FinalFrameAssembler(id_cols=id_cols, target_col=target_col)),
            ]
        )

    # drop duplicate columns if any
    def drop_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            log.info("Dropping duplicate columns if any.")
            return df.loc[:, ~df.columns.duplicated(keep="first")]
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def assert_no_duplicate_columns(self, df: pd.DataFrame, where: str):
        dups = df.columns[df.columns.duplicated()].tolist()
        if dups:
            raise RuntimeError(f"Duplicate columns at {where}: {dups}")

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            log.info("Starting data transformation process.")
            # Read valid train and test data
            valid_train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            valid_test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )
            valid_rul_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_rul_file_path
            )
            # Reverse column order
            valid_rul_df = valid_rul_df[valid_rul_df.columns[::-1]]

            # assert no duplicate columns
            self.assert_no_duplicate_columns(valid_train_df, "valid_train_df")
            self.assert_no_duplicate_columns(valid_test_df, "valid_test_df")
            self.assert_no_duplicate_columns(valid_rul_df, "valid_rul_df")

            log.info("Successfully read validated train, test, and RUL data.")
            if TARGET_COLUMN not in valid_train_df.columns:
                log.info(
                    "Target column missing in training data. Adding RUL to training data."
                )
                valid_train_df = add_train_rul(valid_train_df)
                self.assert_no_duplicate_columns(
                    valid_train_df, "valid_train_df after adding RUL"
                )
            log.info("Adding RUL to test data.")
            if TARGET_COLUMN not in valid_test_df.columns:
                valid_test_df = add_test_rul(valid_test_df, valid_rul_df)
                self.assert_no_duplicate_columns(
                    valid_test_df, "valid_test_df after adding RUL"
                )
            # Build preprocessing pipeline
            log.info("Building preprocessing pipeline.")
            preprocessor_pipeline = self.build_preprocessor_pipeline(
                id_cols=ID_COLUMNS,
                sensor_cols=SENSOR_COLUMN_NAMES,
                base_feature_cols=BASE_FEATURE_COLUMNS,
                target_col=TARGET_COLUMN,
                zn_suffix=ZN_SUFFIX,
                corr_threshold=DATA_TRANSFORMATION_CORRELATION_THRESHOLD,
            )
            # fit the preprocessor pipeline
            log.info("Fitting preprocessing pipeline on training data.")
            preprocessor_pipeline.fit(valid_train_df)
            # Transform train data
            log.info("Transforming training data.")
            transformed_train_df = preprocessor_pipeline.transform(valid_train_df)
            # Transform test data
            log.info("Transforming testing data.")
            transformed_test_df = preprocessor_pipeline.transform(valid_test_df)
            # Save transformed data csv files
            log.info("Saving transformed training and testing data to CSV files.")
            # drop duplicate columns if any
            transformed_train_df = self.drop_duplicate_columns(transformed_train_df)
            transformed_test_df = self.drop_duplicate_columns(transformed_test_df)
            save_transformed_data_as_csv(
                file_path=self.data_transformation_config.transformed_train_file_path,
                data=transformed_train_df,
            )
            save_transformed_data_as_csv(
                file_path=self.data_transformation_config.transformed_test_file_path,
                data=transformed_test_df,
            )
            # Save preprocessing object
            log.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.transformation_object_file_path,
                obj=preprocessor_pipeline,
            )
            # Prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformation_object_file_path=self.data_transformation_config.transformation_object_file_path,
            )
            return data_transformation_artifact

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
