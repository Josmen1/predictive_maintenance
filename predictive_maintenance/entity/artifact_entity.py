from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DataIngestionArtifact:
    """Data class for storing data ingestion artifact information.
    In this case, it holds the file path where the ingested data is stored.

    Attributes:
        feature_store_file_path (str): Path to the feature store file.
        training_file_path (str): Path to the training dataset file.
        testing_file_path (str): Path to the testing dataset file.
    """

    # feature_store_file_path: str
    training_file_path: str
    testing_file_path: str
    rul_file_path: str


@dataclass
class DataValidationArtifact:
    """Data class for storing data validation artifact information.
    In this case, it holds the file paths for valid and invalid datasets,
    as well as the drift report file path.

    Attributes:
        valid_train_file_path (str): Path to the valid training dataset file.
        valid_test_file_path (str): Path to the valid testing dataset file.
        invalid_train_file_path (str): Path to the invalid training dataset file.
        invalid_test_file_path (str): Path to the invalid testing dataset file.
        drift_report_file_path (str): Path to the drift report file.
    """

    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    valid_rul_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    invalid_rul_file_path: str
    drift_report_file_path: str
    # train_drift_report_file_path: str
    # test_drift_report_file_path: str
    # rul_drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """Data class for storing data transformation artifact information.

    Attributes:
        transformed_train_file_path (str): Path to the transformed training dataset file.
        transformed_test_file_path (str): Path to the transformed testing dataset file.
        transformed_rul_file_path (str): Path to the transformed RUL dataset file.
    """

    transformation_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    # transformed_rul_file_path: str


@dataclass
class ModelEvaluationArtifact:
    """Data class for storing model evaluation artifact information.

    Attributes:
        test_rmse (float): RMSE score on the testing dataset.
    """

    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float


@dataclass
class ModelTrainerArtifact:
    """Data class for storing model trainer artifact information.

    Attributes:
        trained_model_file_path (str): Path to the trained model file.
        train_rmse (float): RMSE score on the training dataset.
        test_rmse (float): RMSE score on the testing dataset.
    """

    trained_model_file_path: str
    combined_object_file_path: str
    cv_results_file_path: str
    test_predictions_file_path: str
    train_model_artifact: ModelEvaluationArtifact
    test_model_artifact: ModelEvaluationArtifact
