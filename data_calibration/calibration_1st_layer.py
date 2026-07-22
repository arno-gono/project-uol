import numpy as np

from config import KAGGLE_DATASET_NAME, DB_DIR, DB_NAME
from data.sqlite_connector import connecting_to_sqlite
import pandas as pd
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

    # Nested dictionaries.
    # Column names and column types
    dict_col_types = {}

    # Correlations
    dict_correlations = {}

    # Distributions
    dict_distributions = {}

    # Now getting the actual table to gather data
    df = pd.read_sql(f"SELECT * FROM {table_name}", co)

    # df.describe() returns distribution data already that we will save as well
    df_dist = df.describe()

    # First step: getting the column names and types
    for col_name, dtype in df.dtypes.items():
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

        dict_col_types[col_name] = type_map[temp_type]

        # Add a check for Nulls

        # Now checking the distribution if the field is numerical
        if type_map[temp_type] in ["float", "int"]:
            dict_distributions[col_name] = df_dist[col_name].to_dict()




    return dict_metadata


def dataset_calibration(kaggle_dataset: str = KAGGLE_DATASET_NAME):
    # This function loops through all tables and views available in the dataset passed as argument.
    # For each table, we save the metadata as a dictionary, and aggregate them in a list of dictionaries.
    # Once all the data is collected, it is written as a json file.

    conn = connecting_to_sqlite(kaggle_dataset)

    # Getting the scope of tables / views
    df = pd.read_sql("SELECT * FROM sqlite_master", conn)

    for n, t in zip(df["name"], df["type"]):
        mt_data = _get_metadata_from_table(table_name=n, co=conn)





    conn.close()
    return None




# if __name__ == "__main__":
#     conn = connecting_to_sqlite(KAGGLE_DATASET_NAME)
#
#
#     conn.close()
