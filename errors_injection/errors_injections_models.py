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

    # Finding the datatype of the column that was picked up to inject errors in
    current_datatype = d_calibration["columns_details"][col_error]["datatype"]

    # Picking up a random new datatype for this column
    available_datatypes.remove(current_datatype)
    new_datatype = random.choice(available_datatypes)

    # Generate a list of len(df) random new datatypes
    new_col_data = []

    if current_datatype in ["int", "float"] and new_datatype == "str":
        # This case is more subtle than the other cases that might be more obvious for an agent to detect
        new_col_data = [str(i) for i in df[col_error]]
    elif new_datatype == "datetime":
        new_col_data = pd.date_range("2000-01-01", "2050-12-31", periods=len(df)).tolist()
    elif new_datatype == "int":
        new_col_data = [random.randint(0, 100) for _ in range(len(df))]
    elif new_datatype == "float":
        new_col_data = [round(random.random(), 5) for _ in range(len(df))]
    elif new_datatype == "bool":
        new_col_data = [random.randint(0, 1) for _ in range(len(df))]
    elif new_datatype == "str":
        new_col_data = [str(random.randint(0, 100)) for _ in range(len(df))]

    # Insert corrupted data in the dataframe
    df[col_error] = new_col_data
    return df


def inject_nulls(df: pd.DataFrame, table_name: str) -> pd.DataFrame:

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    # Listing all columns accepting NULLs
    null_columns = [col for col in d_calibration["columns_details"]
                    if d_calibration["columns_details"][col]["null_values"] == False]

    # Choosing a random column that does not accept NULLs
    col_error = random.choice(null_columns)

    # Choosing a random threshold of NULL values that will be inserted
    threshold_nulls = random.random()

    # Looping through all values of the picked up column and replacing with NULL in threshold_nulls percent of cases
    df[col_error] = df[col_error].apply(lambda x: x if random.random() > threshold_nulls else None)
    return df


def inject_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Choosing a random number of rows that will be duplicated
    threshold_duplicate = random.random()

    # Picking up index from the dataframe to duplicate, according to a threshold
    index_to_duplicate = [i for i in df.index if random.random() < threshold_duplicate]

    # Selecting all rows that will be duplicated
    df_dups = df[df.index.isin(index_to_duplicate)]

    # Adding rows to the initial df and sorting by index
    df = pd.concat([df, df_dups])
    df = df.sort_index()

    return df


def inject_new_column(df: pd.DataFrame) -> pd.DataFrame:
    # Choosing a random column
    col_error = random.choice(df.columns)

    # Inserting a new column, taking the exact same data as the column that was picked up
    df[f"NEW_{col_error}"] = df[col_error]

    return df


if __name__ == "__main__":
    from data.sqlite_connector import connecting_to_sqlite
    from config import KAGGLE_DATASET_NAME

    conn_test = connecting_to_sqlite(KAGGLE_DATASET_NAME, database_type="test")

    table_name = "application_record"
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn_test)
    df_raw = df.copy()

    ERROR_TYPES_DICT = {
        "wrong_datatype": inject_wrong_datatype,
        "insert_null": inject_nulls,
        "duplicate_primary_key": 1,
        "duplicate_rows": inject_duplicate_rows,
        "insert_column": inject_new_column,
        "referential_drift": 1,
    }

