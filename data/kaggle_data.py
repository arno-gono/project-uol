import kagglehub
import pandas as pd
import os
import sqlite3

# other dataset to try:
# olistbr/brazilian-ecommerce
# shrinivasv/retail-store-star-schema-dataset
# rikdifos/credit-card-approval-prediction

kaggle_dataset_name = "rikdifos/credit-card-approval-prediction"


def download_data_from_kaggle(kaggle_dataset_name: str):
    # kaggle_dataset_name has the format "owner_data/dataset_name"
    path = kagglehub.dataset_download(kaggle_dataset_name)

    print(f"{len(os.listdir(path))} files from {kaggle_dataset_name} downloaded to: \n{path}")
    return path


def connecting_to_sqlite(kaggle_dataset_name: str):
    # we need a name for the db files. we will use the second part of the kaggle_dataset_name
    db_name = kaggle_dataset_name.split("/")[-1]
    db_name = db_name.replace("-", "_")  # using database conventions
    db_name = f"{db_name}.db"

    # all db files are stored within a specific folder
    rel_path_db_file = f"db_files/{db_name}"

    # sqlite connector
    conn = sqlite3.connect(rel_path_db_file)
    return conn


def upload_files_to_sqlite(path_kaggle_data: str, kaggle_dataset_name: str):
    # getting a connection to sqlite database (or creating one if it does not exist)
    conn = connecting_to_sqlite(kaggle_dataset_name)

    # looping through the csv files from Kaggle and uploading to sqlite
    for file in os.listdir(path_kaggle_data):
        # reading csv file as a dataframe
        df = pd.read_csv(f"{path_kaggle_data}/{file}")

        # creating / replacing table as sqlite
        table_name = file.split(".")[0]
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"Created tables in SQlite")
    print(pd.read_sql("SELECT name, type FROM sqlite_master", conn))

    # closing connection
    conn.close()

    return None


def create_sqlite_view(kaggle_dataset_name: str):
    conn = connecting_to_sqlite(kaggle_dataset_name)
    

    conn.close()
    return None

def main_data_to_sqlite(kaggle_dataset_name: str):

    # downloading data into a default folder used by Kaggle as csv files,
    # and getting the path where the files are located.
    path_kaggle_data = download_data_from_kaggle(kaggle_dataset_name)

    # loading all csv files as db files - recreating a production environment
    upload_files_to_sqlite(path_kaggle_data, kaggle_dataset_name)

    # creating a view - one-off and specific to the kaggle dataset that was chosen
    create_sqlite_view(kaggle_dataset_name)

    return None

