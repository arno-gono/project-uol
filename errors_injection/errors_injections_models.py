import random
import numpy as np
from typing import Any
from pandas import DataFrame
from data.utils import get_calibration_file_as_dict
import pandas as pd


# TODO
# Listing corrupted rows: this needs to align with what we have in SQLite. The index register is not the same

def inject_wrong_datatype(df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, dict]:
    # Inserting another datatype in one column of the test data
    available_datatypes = ["int", "float", "bool", "str", "datetime"]

    # Choosing a threshold for the proportion of data that will be corrupted
    threshold_datatype = random.random()

    # Starting with picking a random column
    col_error = random.choice(df.columns)

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    # Finding the datatype of the column that was picked up to inject errors in
    current_datatype = d_calibration["columns_details"][col_error]["datatype"]

    # Picking up a random new datatype for this column
    available_datatypes.remove(current_datatype)
    new_datatype = random.choice(available_datatypes)

    # Creating a mask determining which rows get corrupted
    mask = np.random.random(len(df)) < threshold_datatype
    nb_data_corrupted = int(mask.sum())
    corrupted_rows = list(df.loc[mask].index)

    # Generating replacement values only for the rows being corrupted
    n = nb_data_corrupted

    if current_datatype in ["int", "float"] and new_datatype == "str":
        # This case is more subtle than the others: the values stay the same,
        # only the datatype changes, making it harder for an agent to detect
        new_values = df.loc[mask, col_error].astype(str).tolist()
    elif new_datatype == "datetime":
        new_values = pd.date_range("2000-01-01", "2050-12-31", periods=max(n, 1)).tolist()[:n]
    elif new_datatype == "int":
        new_values = [random.randint(0, 100) for _ in range(n)]
    elif new_datatype == "float":
        new_values = [round(random.random(), 5) for _ in range(n)]
    elif new_datatype == "bool":
        new_values = [random.randint(0, 1) for _ in range(n)]
    elif new_datatype == "str":
        new_values = [str(random.randint(0, 100)) for _ in range(n)]

    # The column needs to accept mixed types, so it is cast to object first
    # There might be an issue when uploading back to SQLite as str / int might be cast back to original datatype
    df[col_error] = df[col_error].astype(object)
    df.loc[mask, col_error] = new_values

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "former_datatype": current_datatype,
        "new_datatype": new_datatype,
        "threshold_datatype": round(threshold_datatype, 4),
        "nb_data_corrupted": nb_data_corrupted,
        "total_nb_rows": len(df),
        "index_row_corrupted": corrupted_rows
    }
    return df, params


def inject_nulls(df: pd.DataFrame, table_name: str) -> tuple[DataFrame, dict[str, float | Any]]:

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    # Listing all columns accepting NULLs
    null_columns = [col for col in d_calibration["columns_details"]
                    if d_calibration["columns_details"][col]["null_values"] == False]

    # Choosing a random column that does not accept NULLs
    col_error = random.choice(null_columns)

    # Choosing a random threshold of NULL values that will be inserted
    threshold_nulls = random.random()

    # Creating a mask determining which rows get corrupted
    mask = np.random.random(len(df)) < threshold_nulls

    # Inserting NULL values for the selected mask
    df.loc[mask, col_error] = None

    # Count number of NULLs inserted and listing row numbers that were corrupted
    nb_nulls_injected = int(df[col_error].isna().sum())
    corrupted_rows = list(df.loc[mask].index)

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "threshold_nulls": threshold_nulls,
        "nb_nulls_injected": nb_nulls_injected,
        "total_nb_rows": len(df),
        "index_row_corrupted": corrupted_rows
    }

    return df, params


def inject_duplicate_rows(df: pd.DataFrame) -> tuple[DataFrame, dict[str, list[Any] | float]]:
    # Choosing a random number of rows that will be duplicated
    threshold_duplicate = random.random()

    # Picking up index from the dataframe to duplicate, according to a threshold
    index_to_duplicate = [i for i in df.index if random.random() < threshold_duplicate]

    # Selecting all rows that will be duplicated
    df_dups = df[df.index.isin(index_to_duplicate)]

    # Adding rows to the initial df and sorting by index
    df = pd.concat([df, df_dups])
    df = df.sort_index()

    # Keeping params used in injection logs
    params = {
        "threshold_duplicate": threshold_duplicate,
        "nb_duplicated_rows_inserted": len(df_dups),
        "total_nb_rows_before_dups": len(df) - len(df_dups),
        "index_rows_duplicated": index_to_duplicate,
    }

    return df, params


def inject_new_column(df: pd.DataFrame) -> tuple[DataFrame, dict[str, str | Any]]:
    # Choosing a random column
    col_error = random.choice(df.columns)
    name_new_column = f"NEW_{col_error}"

    # Inserting a new column, taking the exact same data as the column that was picked up
    df[name_new_column] = df[col_error]

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "name_new_column": name_new_column,
        "total_nb_rows": len(df)
    }

    return df, params


if __name__ == "__main__":
    from data.sqlite_connector import connecting_to_sqlite
    from config import KAGGLE_DATASET_NAME

    conn_test = connecting_to_sqlite(KAGGLE_DATASET_NAME, database_type="test")

    table_name = "application_record"
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn_test)
    df_raw = df.copy()

    ERROR_TYPES_DICT = {
        "wrong_datatype": inject_wrong_datatype,
        "insert_null": inject_nulls,
        "duplicate_primary_key": 1,
        "duplicate_rows": inject_duplicate_rows,
        "insert_column": inject_new_column,
        "referential_drift": 1,
    }

