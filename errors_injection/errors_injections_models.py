import random
from data.utils import get_calibration_file_as_dict
import pandas as pd



def inject_wrong_datatype(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    # Inserting another datatype in one column of the test data
    available_datatypes = ["int", "float", "bool", "str", "datetime"]

    # Starting with picking a random column
    col_error = random.choice(df.columns)

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    current_datatype = d_calibration["columns_details"][col_error]["datatype"]
    available_datatypes.remove(current_datatype)

    new_datatype = random.choice(available_datatypes)

    # Generate a list of len(df) random new datatypes
    new_col_data = []
    if new_datatype == "datetime":
        new_col_data = pd.date_range("2000-01-01", "2050-12-31", periods=len(df)).tolist()
    elif new_datatype == "int":
        new_col_data = [random.randint(0, 100) for _ in range(len(df))]
    elif new_datatype == "float":
        new_col_data = [round(random.random(), 5) for _ in range(len(df))]
    elif new_datatype == "bool":
        new_col_data = [random.randint(0, 1) for _ in range(len(df))]
    elif new_datatype == "str":
        new_col_data = [str(random.randint(0, 100)) for _ in range(len(df))]

    print(col_error, current_datatype, new_datatype)
    # Inserting corrupted data in the dataframe
    df[col_error] = new_col_data
    return df


def inject_nulls(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    return df


def inject_duplicate_rows(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    return df

def inject_new_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    return df


if __name__ == "__main__":
    from data.sqlite_connector import connecting_to_sqlite
    from config import KAGGLE_DATASET_NAME

    conn_test = connecting_to_sqlite(KAGGLE_DATASET_NAME, is_clean=False)

    table_name = "application_record"
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn_test)

    ERROR_TYPES_DICT = {
        "wrong_datatype": inject_wrong_datatype,
        "insert_null": 1,
        "duplicate_primary_key": 1,
        "duplicate_rows": 1,
        "referential_drift": 1,
    }

    f = ERROR_TYPES_DICT['wrong_datatype']
    q = f(df, table_name)

