from config import DB_DIR, DB_NAME
import json
import sqlite3
import pandas as pd

def get_clean_or_test_csv_path(table_name: str, is_clean_table: bool = True) -> str:
    if is_clean_table:
        table_type = "clean"
    else:
        table_type = "test"

    return f"{DB_DIR}/{table_name}_{table_type}.csv"


def get_calibration_file_as_dict() -> dict:
    path_calibration_file = f"{DB_DIR}/{DB_NAME}.json"
    return json.load(open(path_calibration_file))


def get_calibration_file_path() -> str:
    return f"{DB_DIR}/{DB_NAME}.json"


def read_column_from_whole_dataset(table_name: str, col_name: str, co_clean: sqlite3.Connection,
                                    co_test: sqlite3.Connection) -> pd.Series:
    # Some primary keys might be split from the clean dataset when migrating Kaggle data to SQLite,
    # preventing from mapping the Foreign Keys accurately. Reading both clean and test data together.
    df_clean = pd.read_sql(f"SELECT {col_name} FROM {table_name}", co_clean)
    df_test = pd.read_sql(f"SELECT {col_name} FROM {table_name}", co_test)

    return pd.concat([df_clean, df_test])[col_name]