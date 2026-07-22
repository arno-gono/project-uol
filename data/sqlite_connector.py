import sqlite3
from config import DB_DIR


def connecting_to_sqlite(kaggle_dataset: str):
    # we need a name for the db files. we will use the second part of the kaggle_dataset
    db_name = kaggle_dataset.split("/")[-1]
    db_name = db_name.replace("-", "_")  # using database conventions

    # all db files are stored within a specific folder
    path_db_file = DB_DIR / f"{db_name}.db"

    # sqlite connector
    conn = sqlite3.connect(path_db_file)
    return conn