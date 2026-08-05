import sqlite3
from config import DB_DIR, get_db_name


def connecting_to_sqlite(kaggle_dataset: str, is_clean: bool = True) -> sqlite3.Connection:
    # all db files are stored within a specific folder, each one named after the dataset it holds
    if is_clean:
        path_db_file = DB_DIR / f"{get_db_name(kaggle_dataset)}.db"
    else:
        path_db_file = DB_DIR / f"{get_db_name(kaggle_dataset)}_test.db"

    # sqlite connector
    conn = sqlite3.connect(path_db_file)
    return conn