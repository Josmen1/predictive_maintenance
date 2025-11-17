'''We will create several classes for the training pipeline:

'''
import sys
import os

from predictive_maintenance.logging.logger import get_logger
from predictive_maintenance.exception.exception import PredictiveMaintenanceException
log = get_logger(__name__)

from predictive_maintenance.components.data_ingestion import DataIngestion
from predictive_maintenance.components.data_validation import DataValidation
from predictive_maintenance.components.data_transformation import DataTransformation
from predictive_maintenance.components.model_trainer import ModelTrainer

from predictive_maintenance.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from predictive_maintenance.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            log.info("Starting Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            log.info(f"Data ingestion completed; Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            log.info("Starting Data Validation")
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            log.info(f"Data Validation completed; Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys) from e
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            log.info("Starting Data Transformation")
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config,
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            log.info(f"Data Transformation completed; Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys) from e
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            log.info("Starting Model Training")
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            log.info(f"Model Training completed; Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            log.info("Training Pipeline completed successfully")
            return model_trainer_artifact
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys) from e