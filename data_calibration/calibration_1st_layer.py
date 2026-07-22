from config import KAGGLE_DATASET_NAME, DB_DIR
from data.sqlite_connector import connecting_to_sqlite
import pandas as pd


def dataset_calibration(kaggle_dataset: str):
    conn = connecting_to_sqlite(KAGGLE_DATASET_NAME)
    


    conn.close()
    return None




if __name__ == "__main__":
    conn = connecting_to_sqlite(KAGGLE_DATASET_NAME)


    conn.close()
