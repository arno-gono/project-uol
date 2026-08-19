import random
import pandas as pd
import os
from data.sqlite_connector import connecting_to_sqlite
from config import KAGGLE_DATASET_NAME, DB_DIR_AGENT
from data.utils import get_calibration_file_path
import shutil
from errors_injection.injection_logs import clean_injection_logs
from errors_injection.errors_injections_models import ErrorInjectionsModels



def _clean_db_agent() -> None:
    for file in os.listdir(DB_DIR_AGENT):
        # Cleaning only the db files which are newly generated each round. Leaving the JSON file (Calibration)
        if "db" in file:
            os.remove(f"{DB_DIR_AGENT}/{file}")
    return


def _get_all_tables_from_database(kaggle_dataset: str = KAGGLE_DATASET_NAME) -> list[str]:
    conn = connecting_to_sqlite(kaggle_dataset, database_type="clean")

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


def save_corrupted_data_for_agent(run_number: int, kaggle_dataset: str = KAGGLE_DATASET_NAME) -> None:
    # Clean the folder where the agent finds the data to analyse and the injection log
    _clean_db_agent()

    # Get the scope of all the available tables
    all_tables = _get_all_tables_from_database(kaggle_dataset)

    # Randomly choose how many tables will have errors injected
    tables_with_errors = _pick_tables_to_inject_errors_in(all_tables)
    print(f"Starting injecting errors in {tables_with_errors} tables")

    conn_clean = connecting_to_sqlite(kaggle_dataset, database_type="clean")
    conn_test = connecting_to_sqlite(kaggle_dataset, database_type="test")
    conn_agent = connecting_to_sqlite(kaggle_dataset, database_type="agent")

    # class handling running the error functions
    error_inject_runner = ErrorInjectionsModels()

    for table in all_tables:

        df_clean = pd.read_sql(f"SELECT * FROM {table}", conn_clean)
        df_test = pd.read_sql(f"SELECT * FROM {table}", conn_test)

        if table in tables_with_errors:
            print(f"\nStarting injecting errors in {table}")
            # Inject errors in the _test df
            df_test = error_inject_runner.run_errors(df_test=df_test, table_name=table, run_number=run_number)

        # Saving both the clean and test data for the agent under 2 separate tables
        df_clean.to_sql(table, conn_agent, if_exists="replace", index=False)
        df_test.to_sql(f"{table}_new_data", conn_agent, if_exists="replace", index=False)

    conn_clean.close()
    conn_test.close()
    conn_agent.close()

    return


def run_multiple_rounds(run_number: int, kaggle_dataset: str = KAGGLE_DATASET_NAME) -> None:
    """
        Function that runs several rounds of injection. Each round selects a random number of tables
        which will have errors injected into.
    """

    # Cleaning the injection log file for a fresh start
    clean_injection_logs(kaggle_dataset=kaggle_dataset)

    # Randomly injecting errors into the Test dataset
    save_corrupted_data_for_agent(run_number, kaggle_dataset)

    return None


if __name__ == "__main__":
    kaggle_dataset = KAGGLE_DATASET_NAME
    run_number = 1
    run_multiple_rounds(run_number)

    conn_agent = connecting_to_sqlite(kaggle_dataset, database_type="agent")
    # print(pd.read_sql("SELECT name, type FROM sqlite_master", conn_agent))
