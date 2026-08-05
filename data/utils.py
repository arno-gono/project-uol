from config import DB_DIR, DB_NAME, KAGGLE_DATASET_NAME
import json


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