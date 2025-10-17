import os
import sys
import json
import pandas as pd

from predictive_maintenance import logging
from predictive_maintenance.exception.exception import PredictiveMaintenanceException

# from predictive_maintenance.logging.logger import logging
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)


def load_data(file_path: str, prefix: str) -> pd.DataFrame:
    """
    Load and combine CMPASS training and testing subsets into a single DataFrame for both train and test datasets.
    Each subset is read, lightly cleaned, tagged with its subset id, and appended to a list.
    The list is then concatenated into a single DataFrame.
    """
    try:
        log.info(f"Loading data files with prefix: {prefix} from path: {file_path}")
        all_data = []

        for i in range(1, 5):
            fname = f"{prefix}{i}.txt"
            fpath = os.path.join(file_path, fname)
            if not os.path.exists(fpath):
                print(
                    f"File {fpath} does not exist. Please check the path and filename."
                )
                continue
            log.info(f"Loading data from {fpath}")
            df = pd.read_csv(fpath, sep="\s+", header=None)
            # Light cleaning: drop any empty space columns (if any)
            df.dropna(axis=1, how="all", inplace=True)
            # Tag with subset id
            df["subset"] = f"FD00{i}"
            # Append to list
            all_data.append(df)
            if not all_data:
                raise FileNotFoundError(
                    f"No data files found for prefix: {prefix} in path: {file_path}"
                )
        return pd.concat(all_data, ignore_index=True)
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)


# Defining column names based on community conventions since they are not provided in the dataset
# or treated as proprietary information.
def add_column_names_and_drop_constant_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add column names to the DataFrame based on community conventions.
    """
    base_cols = [
        "unit_number",
        "time_in_cycles",
        "operating_setting_1",
        "operating_setting_2",
        "operating_setting_3",
    ]
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    full_cols = base_cols + sensor_cols + ["subset"]
    data.columns = full_cols
    # Drop columns that are not useful for analysis
    drop_cols = [
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
        "operating_setting_3",
    ]
    data.drop(columns=drop_cols, inplace=True, errors="ignore")
    # Check for any columns that remain constant across the combined dataset
    # zero_var_cols_after = [col for col in data.columns if data[col].nunique(dropna=False) == 1 and col != 'subset']
    # if zero_var_cols_after:
    #    print(f"Warning! constant columns still exist: {zero_var_cols_after}")
    return data


def save_combined_data_to_csv(
    data: pd.DataFrame,
    base_dir="Prediction_Dataset",
    subfolder_name: str = None,
    output_file_name: str = None,
):
    """
    Save a combined DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_dir : str
        Directory to save the file.
    output_filename : str
        Name of the CSV file (e.g., 'train_combined.csv').

    Returns
    -------
    str
        Full path of the saved CSV file..
    """
    try:
        log.info(f"Saving combined data to CSV at: {output_file_name}")
        if subfolder_name is None:
            subfolder_name = "_combined_data_"
        output_dir = os.path.join(base_dir, subfolder_name)
        os.makedirs(output_dir, exist_ok=True)
        """if output_file_name is None:
            output_file_name = f"{prefix}_combined_data.csv"
            """
        output_path = os.path.join(output_dir, output_file_name)
        data.to_csv(output_path, index=False)
        log.info("Data successfully saved to CSV.")
        return output_path
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
