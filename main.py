from predictive_maintenance.components.data_ingestion import DataIngestion
from predictive_maintenance.entity.config_entity import DataIngestionConfig
from predictive_maintenance.entity.config_entity import TrainingPipelineConfig

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
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
