from predictive_maintenance.components.data_ingestion import DataIngestion
from predictive_maintenance.components.data_validation import DataValidation
from predictive_maintenance.components.data_transformation import DataTransformation
from predictive_maintenance.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)

from predictive_maintenance.exception.exception import PredictiveMaintenanceException
import sys

from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        log.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
        print(f"Data Ingestion Artifact: {data_ingestion_artifact}")

        log.info("Data Ingestion Completed")

        log.info("Starting Data Validation")
        data_validation_config = DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config,
        )
        log.info("Initiating Data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        log.info(f"Data Validation Artifact: {data_validation_artifact}")
        print(f"Data Validation Artifact: {data_validation_artifact}")
        log.info("Data Validation Completed")

        log.info("Starting Data Transformation")
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config,
        )
        log.info("Initiating Data Transformation")
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        log.info(f"Data Transformation Artifact: {data_transformation_artifact}")
        print(f"Data Transformation Artifact: {data_transformation_artifact}")
        log.info("Data Transformation Completed")
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
