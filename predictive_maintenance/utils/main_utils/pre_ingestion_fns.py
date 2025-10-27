import os
import sys
import json
import pandas as pd
import yaml

from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)
from typing import Dict, List, Optional


from predictive_maintenance import logging
from predictive_maintenance.exception.exception import PredictiveMaintenanceException

# from predictive_maintenance.logging.logger import logging
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)

from predictive_maintenance.constants import training_pipeline


def load_data(file_path: str, prefix: str, split: str = None) -> pd.DataFrame:
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
            # create a new column for the split type. If split is None, Do not set it
            if split is not None:
                df["split"] = split
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
def add_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add column names to the DataFrame based on community conventions.
    """

    full_cols = training_pipeline.ALL_COLUMN_NAMES
    data.columns = full_cols
    # Dropping columns that are not useful for analysis is moved to data preprocessing step
    """
    drop_cols = [
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
        "ops_3",
    ]
    data.drop(columns=drop_cols, inplace=True, errors="ignore")
    # Check for any columns that remain constant across the combined dataset
    # zero_var_cols_after = [col for col in data.columns if data[col].nunique(dropna=False) == 1 and col != 'subset']
    # if zero_var_cols_after:
    #    print(f"Warning! constant columns still exist: {zero_var_cols_after}")
    """
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


def _norm_dtype(s: pd.Series) -> str:
    """Map pandas dtype to a compact schema dtype.
    The returned dtype strings are simplified versions suitable for schema representation.
    Parameters
    ----------
    s : pd.Series
        The pandas Series to map.

    Returns
    -------
    str
        The mapped schema dtype as a string.
    """
    if is_integer_dtype(s):
        return "int"
    if is_float_dtype(s):
        return "float"
    if is_bool_dtype(s):
        return "bool"
    if is_datetime64_any_dtype(s):
        return "datetime"
    if is_categorical_dtype(s):
        return "category"
    if is_string_dtype(s) or s.dtype == object:
        return "string"
    # Fallback: show the raw dtype string (e.g., Int64, Float32)
    return str(s.dtype)

# Create a function to separate train, test, and RUL data from combined records obtained
# from mongodb through data ingestion component data = list(collection.find()).
# based on the 'split' column. 
# The 'split' column is expected to have values 'train' and 'test' for train and test data
# respectively. However, RUL data does not have a 'split' column.
def separate_train_test_rul_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separate combined data into train, test, and RUL DataFrames based on the 'split' column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Combined DataFrame containing train, test, and RUL data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three DataFrames: (train_df, test_df, rul_df).
    """
    try:
        log.info("Separating combined data into train, test, and RUL datasets.")
        
        # Separate train data
        train_df = data[data['split'] == 'train'].copy()
        train_df.drop(columns=['split'], inplace=True)

        # Separate test data
        test_df = data[data['split'] == 'test'].copy()
        test_df.drop(columns=['split'], inplace=True)

        # Separate RUL data (rows without 'split' column)
        rul_df = data[~data.index.isin(train_df.index) & ~data.index.isin(test_df.index)].copy()

        log.info(
            f"Separated datasets - Train: {len(train_df)} records, Test: {len(test_df)} records, RUL: {len(rul_df)} records."
        )
        
        return train_df, test_df, rul_df

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
    
# Create a function to extract a data_schema from a dataframe and save it to a yaml file
def generate_data_schema_yaml(data: pd.DataFrame, schema_file_path: str) -> str:
    """
    YAML format:
      <col1>: <dtype>
      <col2>: <dtype>
      ...
      numeric_columns:
        - ...
      categorical_columns:
        - ...
    """
    try:
        log.info(f"Generating data schema and saving to: {schema_file_path}")

        # 1) Flat column -> dtype mapping (regular dict preserves order)
        columns_map: dict[str, str] = {}
        for col in data.columns:
            columns_map[col] = _norm_dtype(data[col])

        # 2) Build lists
        numeric_cols = [c for c, t in columns_map.items() if t in {"int", "float"}]
        categorical_cols = [
            c for c, t in columns_map.items() if t in {"bool", "category", "string"}
        ]

        # 3) Final payload: columns first, then lists (order preserved)
        schema: dict = {}
        schema.update(columns_map)
        schema["numeric_columns"] = numeric_cols
        schema["categorical_columns"] = categorical_cols

        # 4) Write YAML
        os.makedirs(os.path.dirname(schema_file_path), exist_ok=True)
        with open(schema_file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                schema, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

        log.info("Data schema successfully saved to YAML.")
        return schema_file_path

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)
