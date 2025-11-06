# run_model_trainer_only.py
import argparse
from pathlib import Path
import sys

from predictive_maintenance.components.model_trainer import ModelTrainer
from predictive_maintenance.entity.config_entity import (
    TrainingPipelineConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from predictive_maintenance.entity.artifact_entity import DataTransformationArtifact
from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)


def find_latest_run_dir(root: Path) -> Path:
    # choose the newest subfolder under artifact/
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Path to the EXISTING run folder under artifact/, e.g. artifact/11_06_2025_07_18_05",
    )
    args = parser.parse_args()

    try:
        artifact_root = Path("artifact")
        run_dir = (
            Path(args.run_dir) if args.run_dir else find_latest_run_dir(artifact_root)
        )

        # Build expected paths from the chosen run_dir
        train_path = run_dir / "data_transformation" / "transformed_data" / "train.csv"
        test_path = run_dir / "data_transformation" / "transformed_data" / "test.csv"
        transformer_path = (
            run_dir
            / "data_transformation"
            / "transformed_object"
            / "preprocessing_object.joblib"
        )

        # Sanity checks
        missing = [
            p for p in [train_path, test_path, transformer_path] if not p.exists()
        ]
        if missing:
            missing_str = "\n  - " + "\n  - ".join(str(p) for p in missing)
            raise FileNotFoundError(f"Expected artifact files not found:{missing_str}")

        log.info(f"Using existing run dir: {run_dir}")

        # If your TrainingPipelineConfig accepts artifact_dir, pin it to the SAME run,
        # so the trainer also writes under this run. If not, it's fine to let it
        # create a new output run; the important part is reading the right inputs.
        try:
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=str(run_dir))
        except TypeError:
            # Fallback if your class doesn't accept artifact_dir in __init__
            training_pipeline_config = TrainingPipelineConfig()

        # You don’t need to construct DataTransformationConfig; just rebuild the artifact
        data_transformation_artifact = DataTransformationArtifact(
            transformed_train_file_path=str(train_path),
            transformed_test_file_path=str(test_path),
            transformation_object_file_path=str(transformer_path),
        )

        model_trainer_config = ModelTrainerConfig(
            training_pipeline_config=training_pipeline_config
        )

        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )

        log.info("Initiating Model Training (using existing transformation artifacts)")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        log.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        print(f"Model Trainer Artifact: {model_trainer_artifact}")

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
