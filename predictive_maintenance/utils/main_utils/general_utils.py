# save object as joblib file
import os
import sys
import pandas as pd
import numpy as np  
import joblib

from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)

def save_object(file_path: str, obj: object) -> None:
    """Save a Python object to a file using joblib.

    Args:
        file_path (str): The file path where the object will be saved.
        obj (object): The Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        log.info(f"Saved object to: {file_path}")

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
    
# load object from joblib file
def load_object(file_path: str) -> object:
    """Load a Python object from a joblib file.

    Args:
        file_path (str): The file path from which the object will be loaded.

    Returns:
        object: The loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = joblib.load(file_obj)
        log.info(f"Loaded object from: {file_path}")
        return obj

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
    
def read_csv_as_dataframe(file_path: str) -> pd.DataFrame:
    """Read a CSV file and return it as a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        log.info(f"Loaded DataFrame from: {file_path}")
        return df

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)

def write_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """Write a pandas DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be written.
        file_path (str): The path where the CSV file will be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        dataframe.to_csv(file_path, index=False)
        log.info(f"Wrote DataFrame to: {file_path}")

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
    
# make directory if not exists
def make_directory(dir_path: str) -> None:
    """Create a directory if it does not exist.

    Args:
        dir_path (str): The path of the directory to be created.
    """
    try:
        
        os.makedirs(dir_path, exist_ok=True)
        log.info(f"Directory created or already exists: {dir_path}")

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)