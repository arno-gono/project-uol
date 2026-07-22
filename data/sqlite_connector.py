import sqlite3
from config import DB_DIR, DB_NAME


def connecting_to_sqlite(kaggle_dataset: str):
    # all db files are stored within a specific folder
    path_db_file = DB_DIR / f"{DB_NAME}.db"

    # sqlite connector
    conn = sqlite3.connect(path_db_file)
    return conn