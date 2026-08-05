import random
import pandas as pd
import os
from data.sqlite_connector import connecting_to_sqlite
from config import KAGGLE_DATASET_NAME, DB_NAME, DB_DIR_AGENT
from data.utils import get_calibration_file_path, get_calibration_file_as_dict
import shutil
from errors_injection.errors_injections_models import inject_wrong_datatype

ERROR_TYPES_DICT = {
    "wrong_datatype": inject_wrong_datatype,
    "insert_null": 1,
    "duplicate_primary_key": 1,
    "duplicate_rows": 1,
    "referential_drift": 1,
}


def _clean_db_agent() -> None:
    for file in os.listdir(DB_DIR_AGENT):
        os.remove(f"{DB_DIR_AGENT}/{file}")
    return


def _copy_calibration_files() -> None:
    # Getting the path of the JSON calibration file
    path_calibration_file = get_calibration_file_path()
    file_name = path_calibration_file.split("/")[-1]

    # Pasting in the db_agent folder
    shutil.copy(path_calibration_file, f"{DB_DIR_AGENT}/{file_name}")

    return None


def _get_all_tables_from_database(kaggle_dataset: str = KAGGLE_DATASET_NAME) -> list[str]:
    conn = connecting_to_sqlite(kaggle_dataset, is_clean=True)

    # Getting the scope of tables / views
    df = pd.read_sql("SELECT * FROM sqlite_master", conn)
    conn.close()

    # Only adding data to tables / removing views from the scope
    df = df[df["type"] == "table"]

    return list(df["name"])


def _pick_tables_to_inject_errors_in(list_tables: list[str]) -> list[str]:
    nb_tables_to_pick = random.randint(0, len(list_tables))
    if nb_tables_to_pick == 0:
        return []

    tables_picked = []
    counter = 0

    while counter < nb_tables_to_pick:
        # Choosing a random index in the list of tables
        random_index = random.randint(0, len(list_tables) - 1)

        if list_tables[random_index] not in tables_picked:
            tables_picked.append(list_tables[random_index])
            counter += 1

    return tables_picked


def save_corrupted_data_for_agent(kaggle_dataset: str = KAGGLE_DATASET_NAME) -> None:
    # Clean the folder where the agent finds the data to analyse
    _clean_db_agent()

    # Get the scope of all the available tables
    all_tables = _get_all_tables_from_database(kaggle_dataset)

    # Randomly choose how many tables will have errors injected
    tables_with_errors = _pick_tables_to_inject_errors_in(all_tables)

    for table in all_tables:
        conn_clean = connecting_to_sqlite(kaggle_dataset, is_clean=True)
        conn_test = connecting_to_sqlite(kaggle_dataset, is_clean=False)

        df_clean = pd.read_sql(f"SELECT * FROM {table}", conn_clean)
        df_test = pd.read_sql(f"SELECT * FROM {table}", conn_test)

        if table in tables_with_errors:

            # Randomly choose the type of errors that will be injected

            # Inject errors in the _test database
            df_test = df_test

        # Add the error data to the _clean dataset and save for the agent
        df_final = pd.concat([df_clean, df_test])
        df_final.to_csv(f"{DB_DIR_AGENT}/{DB_NAME}_{table}.csv", index=False)

    # The JSON for the calibration also needs to be available to the agent.
    _copy_calibration_files()

    return


if __name__ == "__main__":
    kaggle_dataset = KAGGLE_DATASET_NAME