import pandas as pd
import os

from predictive_maintenance.utils.main_utils.pre_ingestion_fns import (
    generate_data_schema_yaml,
)

from predictive_maintenance.constants import training_pipeline as tp

train_schema_file_path = tp.TRAIN_SCHEMA_FILE_PATH
test_schema_file_path = tp.TEST_SCHEMA_FILE_PATH


# A function reads a csv file, creates a data schema yaml file
# from the dataframe and saves it to a given path
def generate_yml_from_df(
    data: pd.DataFrame,
    schema_file_path: str,
) -> None:
    """
    Generate a data schema YAML file from a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to extract schema from.
    schema_file_path : str
        Path to save the YAML schema file.

    Returns
    -------
    None
    """
    generate_data_schema_yaml(data, schema_file_path)


if __name__ == "__main__":
    # generate a schema yaml file from a dataframe
    df_train = pd.read_csv("Prediction_Dataset/train/train.csv")
    generate_yml_from_df(df_train, train_schema_file_path)
    print(f"Schema YAML file generated at: {train_schema_file_path}")

    df_test = pd.read_csv("Prediction_Dataset/test/test.csv")
    generate_yml_from_df(df_test, test_schema_file_path)
    print(f"Schema YAML file generated at: {test_schema_file_path}")
