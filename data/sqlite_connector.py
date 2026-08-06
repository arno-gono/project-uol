import sqlite3
from config import DB_DIR, DB_DIR_AGENT, get_db_name


def connecting_to_sqlite(kaggle_dataset: str, database_type: str = "clean") -> sqlite3.Connection | None:
    # all db files are stored within a specific folder, each one named after the dataset it holds
    if database_type == "clean":
        path_db_file = DB_DIR / f"{get_db_name(kaggle_dataset)}.db"
    elif database_type == "test":
        path_db_file = DB_DIR / f"{get_db_name(kaggle_dataset)}_test.db"
    elif database_type == "agent":
        path_db_file = DB_DIR_AGENT / f"{get_db_name(kaggle_dataset)}.db"
    else:
        print("Database name not recognized")
        return None

    # sqlite connector
    conn = sqlite3.connect(path_db_file)
    return conn