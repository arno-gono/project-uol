import kagglehub
import pandas as pd
import os
from data.sqlite_connector import connecting_to_sqlite
from config import KAGGLE_DATASET_NAME



def download_data_from_kaggle(kaggle_dataset: str):
    # kaggle_dataset has the format "owner_data/dataset_name"
    path = kagglehub.dataset_download(kaggle_dataset)

    print(f"{len(os.listdir(path))} files from {kaggle_dataset} downloaded to: \n{path}")
    return path


def upload_files_to_sqlite(path_kaggle_data: str, kaggle_dataset: str):
    # getting a connection to sqlite database (or creating one if it does not exist)
    conn = connecting_to_sqlite(kaggle_dataset)

    # looping through the csv files from Kaggle and uploading to sqlite
    for file in os.listdir(path_kaggle_data):
        # reading csv file as a dataframe
        df = pd.read_csv(f"{path_kaggle_data}/{file}")

        # creating / replacing table as sqlite
        table_name = file.split(".")[0]
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    # closing connection
    conn.close()

    return None


def create_sqlite_view(kaggle_dataset: str):
    conn = connecting_to_sqlite(kaggle_dataset)

    # declare the view as a string

    view_query = """
                    CREATE VIEW V_TEST_TABLE AS
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
    path_kaggle_data = download_data_from_kaggle(kaggle_dataset)

    # loading all csv files as db files - recreating a production environment
    upload_files_to_sqlite(path_kaggle_data, kaggle_dataset)

    # Run the following function create_sqlite_view(kaggle_dataset) to create a view. Make sure
    # it is actually a view that works with the selected dataset

    return None


if __name__ == "__main__":
    # open a connection
    conn = connecting_to_sqlite(KAGGLE_DATASET_NAME)

    print("Hello")

    main_data_to_sqlite(KAGGLE_DATASET_NAME)

    # check existing tables
    print(pd.read_sql("SELECT name, type FROM sqlite_master", conn))

    create_sqlite_view(KAGGLE_DATASET_NAME)

    # open a table / view as df
    table_name = "v_test_table"
    # df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10", conn)

    conn.close()
    print("END")


