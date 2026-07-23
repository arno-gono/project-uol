import json
import os

from config import KAGGLE_DATASET_NAME, DB_DIR, DB_NAME, MAX_CARDINALITY_NB
from data.sqlite_connector import connecting_to_sqlite
import pandas as pd
import numpy as np
import sqlite3


# remapping numpy / pandas datatypes to Python built-ins
type_map = {
    "int64": "int",
    "float64": "float",
    "str": "str",
    "bool": "bool",
}


def _get_metadata_from_table(table_name: str, co: sqlite3.Connection):
    # Main dict of metadata, containing nested dictionaries.
    dict_metadata = {}

    # Correlations
    dict_correlations = {}
    numerical_cols = []  # listing all numerical columns to calculate correlations between them

    # Now getting the actual table to gather data
    df = pd.read_sql(f"SELECT * FROM {table_name}", co)

    # df.describe() returns distribution data already that we will save as well
    df_dist = df.describe()

    # Main data
    nb_entries = len(df)
    nb_columns = len(df.columns)

    dict_metadata["nb_entries"] = nb_entries
    dict_metadata["nb_columns"] = nb_columns
    dict_metadata["columns_details"] = {}

    # First step: getting the column names and types
    for col_name, dtype in df.dtypes.items():
        dict_column = {}

        # Checking the unique values
        unique_values = list(df[col_name].unique())

        if len(unique_values) <= MAX_CARDINALITY_NB:
            categorical_field = True
            dict_categories = {}

            # looping through the value_counts method, listing each category and the absolute number of times they
            # appear in the table
            for i, value in df[col_name].value_counts(normalize=True).items():
                dict_categories[i] = value # Storing the relative value

            dict_column["cardinality_distribution"] = dict_categories

        # Add a check for Nulls
        if df[col_name].isna().any():
            dict_column["null_values"] = True
        else:
            dict_column["null_values"] = False

        # Data dependent on the column type
        temp_type = str(dtype)

        # Checking if this data type is referenced in type_map
        if temp_type not in type_map:
            # Add a log rather than printing on console only
            e = f"{temp_type} is not referenced in type_map"
            print(e)
            continue

        # Y/N columns and 0/1 columns are remapped to bool.
        # Using issubset as some data actually have only one value that seems trivial to be marked as True or False.
        if set(df[col_name].unique()).issubset({"Y", "N"}) or set(df[col_name].unique()).issubset({0, 1}):
            temp_type = "bool"

        dict_column["datatype"] = type_map[temp_type]

        # Now checking the distribution if the field is numerical
        if type_map[temp_type] in ["float", "int"]:
            numerical_cols.append(col_name)
            dict_column["distributions"] = df_dist[col_name].to_dict()

        # Adding column's metadata to the table metadata
        dict_metadata["columns_details"][col_name] = dict_column


    return dict_metadata


def _aggregate_metadata(kaggle_dataset: str = KAGGLE_DATASET_NAME):
    # This function loops through all tables and views available in the dataset passed as argument.
    # For each table, we save the metadata as a dictionary, and aggregate them in a list of dictionaries.
    # Once all the data is collected, it is written as a json file.

    conn = connecting_to_sqlite(kaggle_dataset)

    # Getting the scope of tables / views
    df = pd.read_sql("SELECT * FROM sqlite_master", conn)

    dict_metadata = {}

    for n, t in zip(df["name"], df["type"]):
        dict_table_metadata = _get_metadata_from_table(table_name=n, co=conn)
        dict_metadata[n] = {"type": t} | dict_table_metadata

    conn.close()

    return dict_metadata


def _save_metadata(dict_data: dict, dataset_name: str = DB_NAME):
    # Saving data under the name of the dataset name
    path_json_file = os.path.join(DB_DIR, f"{dataset_name}.json")

    with open(path_json_file, "w") as f:
        json.dump(dict_data, f, indent=4, default=str)

    return None


def dataset_calibration():
    # Full analysis of each table of the dataset
    dict_metadata = _aggregate_metadata()

    # Saving the data as a json file
    _save_metadata(dict_data=dict_metadata, dataset_name=DB_NAME)

    return None

if __name__ == "__main__":
    dataset_calibration()