import sys
from prefect import flow, get_run_logger
from predictive_maintenance.pipeline.training_pipeline import TrainingPipeline
from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.logging.logger import get_logger

logger = get_run_logger()
log = get_logger(__name__)


@flow(name="predictive-maintenance-training")
def training_flow():
    try:
        logger.info("Starting training pipeline...")
        log.info("Starting training pipeline...")
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        logger.info(f"Training pipeline completed successfully.{artifact}")
        log.info(f"Training pipeline completed successfully.{artifact}")
        return str(artifact)
    except PredictiveMaintenanceException as e:
        logger.error(f"Error occurred: {e}")
        log.error(f"Error occurred: {e}")
        raise Exception(e, sys) from e
