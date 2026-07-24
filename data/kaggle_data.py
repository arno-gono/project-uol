import kagglehub
import pandas as pd
import os
from data.sqlite_connector import connecting_to_sqlite
from config import KAGGLE_DATASET_NAME, DB_DIR, DB_NAME



def _download_data_from_kaggle(kaggle_dataset: str):
    # kaggle_dataset has the format "owner_data/dataset_name"
    path = kagglehub.dataset_download(kaggle_dataset)
    return path


def _split_raw_table_to_clean_and_test(df: pd.DataFrame):
    len_df = len(df)

    # only splitting tables that have over X rows to avoid splitting a mapping that might be used in another table
    min_nb_rows = 100

    # Split choice: Y% of the data is kept in the clean vs the test table
    split_percent = 0.95
    nb_row_split = int(len_df * split_percent)

    if len(df) < min_nb_rows:
        return {"df_raw": df}

    # Creating 2 separate tables
    df_clean = df[:nb_row_split]
    df_test = df[nb_row_split:]

    return {"df_raw": df, "df_clean": df_clean, "df_test": df_test}


def _clean_database():
    for f in os.listdir(DB_DIR):
        if DB_NAME in f:
            os.remove(f"{DB_DIR}/{f}")


def _upload_files_to_sqlite(path_kaggle_data: str, kaggle_dataset: str):
    # getting a connection to sqlite database (or creating one if it does not exist)
    conn = connecting_to_sqlite(kaggle_dataset)

    # looping through the csv files from Kaggle and uploading to sqlite
    for file in os.listdir(path_kaggle_data):
        # reading csv file as a dataframe
        df = pd.read_csv(f"{path_kaggle_data}/{file}")

        # Each table is saved in 3 different versions. the _raw table is the full data available.
        # the table is then split in 2: a _clean table, which will be used for calibration, and a _test table where
        # errors will be injected
        split_set = _split_raw_table_to_clean_and_test(df=df)

        # creating / replacing table as sqlite
        table_name = file.split(".")[0]

        # if there are a clean and a test table, they are also uploaded to SQLite
        if "df_clean" in split_set:
            # Uploading the clean data to SQLite
            split_set["df_clean"].to_sql(table_name, conn, if_exists="replace", index=False)

            # Also saving the clean and test data as Dataframe in order not to pollute the SQL database
            split_set["df_clean"].to_csv(f"{DB_DIR}/{table_name}_clean.csv", index=False)
            split_set["df_test"].to_csv(f"{DB_DIR}/{table_name}_test.csv", index=False)

        else:
            # Uploading the raw data to SQLite
            df.to_sql(table_name, conn, if_exists="replace", index=False)

    # closing connection
    conn.close()

    return None


def _create_sqlite_view(kaggle_dataset: str):
    conn = connecting_to_sqlite(kaggle_dataset)

    # declare the view as a string

    view_query = """
                    CREATE VIEW v_test_table AS
                    SELECT 
                        CR.ID, 
                        CR.MONTHS_BALANCE,
                        CR.STATUS,
                        AR.NAME_INCOME_TYPE,
                        AR.NAME_FAMILY_STATUS
                    FROM 
                        CREDIT_RECORD CR
                    LEFT JOIN
                        APPLICATION_RECORD AR
                    ON 
                        CR.ID = AR.ID
                    WHERE
                        AR.NAME_FAMILY_STATUS = 'Married'
                    LIMIT 50
                """

    # dropping the current view if it already exists
    conn.execute("DROP VIEW IF EXISTS V_TEST_TABLE")

    # now creating the view
    conn.execute(view_query)

    conn.close()
    return None

def main_data_to_sqlite(kaggle_dataset: str):

    # downloading data into a default folder used by Kaggle as csv files,
    # and getting the path where the files are located.
    path_kaggle_data = _download_data_from_kaggle(kaggle_dataset)

    # cleaning the database if it already exists
    _clean_database()

    # loading all csv files as db files - recreating a production environment
    _upload_files_to_sqlite(path_kaggle_data, kaggle_dataset)

    # Run the following function create_sqlite_view(kaggle_dataset) to create a view. Make sure
    # it is actually a view that works with the selected dataset

    return None


if __name__ == "__main__":
    # open a connection
    conn = connecting_to_sqlite(KAGGLE_DATASET_NAME)
    kaggle_dataset = KAGGLE_DATASET_NAME
    print("Hello")

    # main_data_to_sqlite(KAGGLE_DATASET_NAME)

    # check existing tables
    print(pd.read_sql("SELECT name, type FROM sqlite_master", conn))

    # _create_sqlite_view(KAGGLE_DATASET_NAME)

    # open a table / view as df
    # table_name = "v_test_table"
    # df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10", conn)

    conn.close()
    print("END")


