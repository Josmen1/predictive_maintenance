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
