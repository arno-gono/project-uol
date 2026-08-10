import random
import numpy as np
from typing import Any
from pandas import DataFrame
from data.utils import get_calibration_file_as_dict, read_column_from_whole_dataset
from data.sqlite_connector import connecting_to_sqlite
from errors_injection.injection_logs import append_injection_logs
from config import KAGGLE_DATASET_NAME
import pandas as pd


# TODO
# Listing corrupted rows: this needs to align with what we have in SQLite. The index register is not the same
# Delete a column from the table

def inject_wrong_datatype(df: pd.DataFrame, table_name: str) -> tuple[DataFrame, dict[
    str, str | float | int | list[Any] | Any]] | None:
    # Inserting another datatype in one column of the test data
    available_datatypes = ["int", "float", "bool", "str", "datetime"]

    # Choosing a threshold for the proportion of data that will be corrupted
    threshold_datatype = random.random()

    # Starting with picking a random column
    col_error = random.choice(df.columns)

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    if col_error not in d_calibration["columns_details"]:
        # Case where this function is run on a column that was newly created and is not in the calibration
        return None

    # Finding the datatype of the column that was picked up to inject errors in
    current_datatype = d_calibration["columns_details"][col_error]["datatype"]

    # Picking up a random new datatype for this column
    available_datatypes.remove(current_datatype)
    new_datatype = random.choice(available_datatypes)

    # Creating a mask determining which rows get corrupted
    mask = np.random.random(len(df)) < threshold_datatype
    nb_data_corrupted = int(mask.sum())
    corrupted_rows = list(df.loc[mask].index)

    # Generating replacement values only for the rows being corrupted
    n = nb_data_corrupted

    if current_datatype in ["int", "float"] and new_datatype == "str":
        # This case is more subtle than the others: the values stay the same,
        # only the datatype changes, making it harder for an agent to detect
        new_values = df.loc[mask, col_error].astype(str).tolist()
    elif new_datatype == "datetime":
        dates = pd.date_range("2000-01-01", "2050-12-31", periods=max(n, 1))
        new_values = dates.strftime("%Y-%m-%d %H:%M:%S").tolist()[:n]
    elif new_datatype == "int":
        new_values = [random.randint(0, 100) for _ in range(n)]
    elif new_datatype == "float":
        new_values = [round(random.random(), 5) for _ in range(n)]
    elif new_datatype == "bool":
        new_values = [random.randint(0, 1) for _ in range(n)]
    elif new_datatype == "str":
        new_values = [str(random.randint(0, 100)) for _ in range(n)]

    # The column needs to accept mixed types, so it is cast to object first
    # There might be an issue when uploading back to SQLite as str / int might be cast back to original datatype
    df[col_error] = df[col_error].astype(object)
    df.loc[mask, col_error] = new_values

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "former_datatype": current_datatype,
        "new_datatype": new_datatype,
        "threshold_datatype": round(threshold_datatype, 4),
        "nb_data_corrupted": nb_data_corrupted,
        "total_nb_rows": len(df),
        "index_row_corrupted": corrupted_rows
    }
    return df, params


def inject_nulls(df: pd.DataFrame, table_name: str) -> tuple[DataFrame, dict[str, float | Any]]:

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

    # Creating a mask determining which rows get corrupted
    mask = np.random.random(len(df)) < threshold_nulls

    # Inserting NULL values for the selected mask
    df.loc[mask, col_error] = None

    # Count number of NULLs inserted and listing row numbers that were corrupted
    nb_nulls_injected = int(df[col_error].isna().sum())
    corrupted_rows = list(df.loc[mask].index)

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "threshold_nulls": threshold_nulls,
        "nb_nulls_injected": nb_nulls_injected,
        "total_nb_rows": len(df),
        "index_row_corrupted": corrupted_rows
    }

    return df, params


def inject_duplicate_rows(df: pd.DataFrame) -> tuple[DataFrame, dict[str, list[Any] | float]]:
    # Choosing a random number of rows that will be duplicated
    threshold_duplicate = random.random()

    # Picking up index from the dataframe to duplicate, according to a threshold
    index_to_duplicate = [i for i in df.index if random.random() < threshold_duplicate]

    # Selecting all rows that will be duplicated
    df_dups = df[df.index.isin(index_to_duplicate)]

    # Adding rows to the initial df and sorting by index
    df = pd.concat([df, df_dups])
    df = df.sort_index()

    # Keeping params used in injection logs
    params = {
        "threshold_duplicate": threshold_duplicate,
        "nb_duplicated_rows_inserted": len(df_dups),
        "total_nb_rows_before_dups": len(df) - len(df_dups),
        "index_rows_duplicated": index_to_duplicate,
    }

    return df, params


def inject_new_column(df: pd.DataFrame) -> tuple[DataFrame, dict[str, str | int | Any]] | None:
    # Choosing a random column
    col_error = random.choice(df.columns)
    name_new_column = f"NEW_{col_error}"

    # this column might already be in the dataset (repetition)
    if name_new_column in df.columns:
        return None

    # Inserting a new column, taking the exact same data as the column that was picked up
    df[name_new_column] = df[col_error]

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "name_new_column": name_new_column,
        "total_nb_rows": len(df)
    }

    return df, params


def _get_primary_keys(table_name: str, col_name: str) -> set:
    co_test = connecting_to_sqlite(KAGGLE_DATASET_NAME, database_type="test")
    co_clean = connecting_to_sqlite(KAGGLE_DATASET_NAME, database_type="clean")
    col_values = read_column_from_whole_dataset(table_name, col_name, co_clean, co_test)
    co_test.close()
    co_clean.close()
    return set(col_values)


def inject_orphan_foreign_key(df: pd.DataFrame, table_name: str) -> tuple[DataFrame, dict[str, Any]] | None:
    # Breaking a foreign key: the key points to a parent that does not exist.

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    # Getting all foreign keys from the calibration file
    foreign_keys = {col: details for col, details in d_calibration["potential_foreign_key"].items()
                    if col in df.columns}

    if not foreign_keys:
        return None

    # Picking a random foreign key to break
    col_error = random.choice(list(foreign_keys))
    parent_table = foreign_keys[col_error]["parent_table"]

    # Getting primary keys so the injected ones are not existing keys
    primary_keys = _get_primary_keys(table_name=parent_table, col_name=foreign_keys[col_error]["parent_column"])

    # Choosing a threshold for the proportion of data that will be corrupted
    threshold_orphan = random.random()

    # Creating a mask determining which rows get corrupted. Rows holding no foreign key are left out
    mask = (np.random.random(len(df)) < threshold_orphan) & df[col_error].notna().to_numpy()
    nb_data_corrupted = int(mask.sum())
    corrupted_rows = list(df.loc[mask].index)

    # Building random foreign keys
    def _generate_random_fkey(fkey: str) -> str:

        # Two cases: swapping 2 chars, or inserting one.
        if random.random() < 0.5:
            # Swapping 2 characters
            i, j = random.sample(range(len(fkey)), 2)
            chars = list(fkey)
            chars[i], chars[j] = chars[j], chars[i]
            return "".join(chars)

        # Inserting a random character into a random location
        random_loc = random.choice(range(len(fkey)))
        random_char = random.choice("0123456789abcdefghijklmnopqrstuvwxyz")
        random_char = random_char.upper() if random.choice(["upper", "lower"]) == "upper" else random_char
        return fkey[:random_loc] + random_char + fkey[random_loc:]


    def _get_corrupted_fkey(fkey: str) -> str:
        new_fkey = _generate_random_fkey(fkey)
        while new_fkey in primary_keys or new_fkey == fkey:
            new_fkey = _generate_random_fkey(fkey)
        return new_fkey

    # The same key always maps to the same corrupted one, so a key repeated in the table stays repeated
    new_values_dict = {fkey: _get_corrupted_fkey(fkey) for fkey in set(df.loc[mask, col_error])}
    df.loc[mask, col_error] = df.loc[mask, col_error].map(new_values_dict)

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "parent_table": parent_table,
        "parent_column": foreign_keys[col_error]["parent_column"],
        "former_coverage": foreign_keys[col_error]["coverage"],
        "threshold_orphan": round(threshold_orphan, 4),
        "nb_data_corrupted": nb_data_corrupted,
        "total_nb_rows": len(df),
        "dict_new_foreign_keys": new_values_dict,
        "index_row_corrupted": corrupted_rows
    }

    return df, params


def inject_new_category(df: pd.DataFrame, table_name: str) -> tuple[DataFrame, dict[str, Any]] | None:
    # Inserting a category that was not seen at calibration time, impacting cardinality distribution.
    available_labels = ["UNKNOWN", "N/A", "-99999", "TO_BE_DEFINED", "OTHER", "Not available", "Not applicable"]

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    # Selecting categorical columns that are strings and not primary keys.
    categorical_columns = [col for col, details in d_calibration["columns_details"].items()
                           if details["cardinality_distribution"] is not None
                           and details["datatype"] == "str"
                           and details["potential_primary_key"] is False
                           and col in df.columns]

    if not categorical_columns:
        return None

    # Picking a random column and the label that will be inserted in it
    col_error = random.choice(categorical_columns)

    # Ensuring the label that will be picked is not in the column picked up
    available_labels = [label for label in available_labels if label not in df[col_error].unique()]
    new_label = random.choice(available_labels)

    # Choosing a threshold for the proportion of data that will be corrupted
    threshold_category = random.random()

    # Creating a mask determining which rows get corrupted
    mask = np.random.random(len(df)) < threshold_category
    nb_data_corrupted = int(mask.sum())
    corrupted_rows = list(df.loc[mask].index)

    df.loc[mask, col_error] = new_label

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "new_label": new_label,
        "known_categories": list(d_calibration["columns_details"][col_error]["cardinality_distribution"]),
        "threshold_category": round(threshold_category, 4),
        "nb_data_corrupted": nb_data_corrupted,
        "total_nb_rows": len(df),
        "index_row_corrupted": corrupted_rows
    }

    return df, params


def inject_correlation_break(df: pd.DataFrame, table_name: str) -> tuple[DataFrame, dict[str, Any]] | None:
    # Shuffling the values of one column that has high correlation with another column. Distribution and cardinality
    # for the shuffled value stays the same, but the correlation with the column it is correlated with is impacted
    min_correlation = 0.3

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    # Keeping the pairs that are correlated enough for a break to be measurable, and whose two columns
    # are both still in the dataframe
    correlated_pairs = [pair for pair, correlation in d_calibration["correlations"].items()
                        if abs(correlation) >= min_correlation
                        and all(col.strip() in df.columns for col in pair.split("|"))]

    if not correlated_pairs:
        return None

    # Picking a random pair, and randomly choosing which column will be shuffled
    pair = random.choice(correlated_pairs)
    col_paired, col_error = random.sample([col.strip() for col in pair.split("|")], k=2)

    # Choosing a threshold for the proportion of data that will be corrupted
    threshold_correlation = random.random()

    # Creating a mask determining which rows get corrupted
    mask = np.random.random(len(df)) < threshold_correlation
    nb_data_corrupted = int(mask.sum())
    corrupted_rows = list(df.loc[mask].index)

    # Calculating correlations on corrupted rows before and after injecting errors.
    former_correlation = round(float(df[col_paired].corr(df[col_error])), 4)

    # Shuffling the columns' values for the rows that were picked up
    df.loc[mask, col_error] = df.loc[mask, col_error].sample(frac=1).tolist()

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "col_paired": col_paired,
        "calibrated_correlation": d_calibration["correlations"][pair],
        "former_correlation": former_correlation,
        "new_correlation": round(float(df[col_paired].corr(df[col_error])), 4),
        "threshold_correlation": round(threshold_correlation, 4),
        "nb_data_corrupted": nb_data_corrupted,
        "total_nb_rows": len(df),
        "index_row_corrupted": corrupted_rows
    }

    return df, params


def inject_duplicate_primary_key(df: pd.DataFrame, table_name: str) -> tuple[DataFrame, dict[str, Any]] | None:
    # Giving a few rows a primary key that another row already uses.

    # Get the calibration file as a dictionary
    d_calibration = get_calibration_file_as_dict()
    d_calibration = d_calibration[table_name]

    # Selecting potential primary keys
    primary_keys = [col for col, details in d_calibration["columns_details"].items()
                    if details["potential_primary_key"] and col in df.columns]

    if not primary_keys:
        return None

    # Picking a random key to duplicate
    col_error = random.choice(primary_keys)

    # Choosing a threshold for the proportion of data that will be corrupted
    threshold_duplicate_key = random.random()

    # Creating a mask determining which rows get corrupted.
    mask = np.random.random(len(df)) < threshold_duplicate_key
    nb_data_corrupted = int(mask.sum())

    if nb_data_corrupted == 0:
        return None

    corrupted_rows = list(df.loc[mask].index)

    # Selecting all keys available that will be copied from (the mask's inverse)
    keys_available = df.loc[~mask, col_error].dropna().unique()

    if len(keys_available) == 0:
        return None

    nb_unique_keys_before = int(df[col_error].nunique())
    df.loc[mask, col_error] = list(np.random.choice(keys_available, size=nb_data_corrupted))

    # Keeping params used in injection logs
    params = {
        "col_error": col_error,
        "threshold_duplicate_key": round(threshold_duplicate_key, 4),
        "nb_data_corrupted": nb_data_corrupted,
        "nb_unique_keys_before": nb_unique_keys_before,
        "nb_keys_duplicated": int((df[col_error].value_counts() > 1).sum()),
        "total_nb_rows": len(df),
        "index_row_corrupted": corrupted_rows
    }

    return df, params


class ErrorInjectionsModels:
    def __init__(self):

        self.df = None
        self.table_name = None
        self.run_number = None

        self.error_types_dict = {
            "wrong_datatype": inject_wrong_datatype,
            "insert_null": inject_nulls,
            "duplicate_rows": inject_duplicate_rows,
            "insert_column": inject_new_column,
            "orphan_foreign_key": inject_orphan_foreign_key,
            "new_category": inject_new_category,
            "correlation_break": inject_correlation_break,
            "duplicate_primary_key": inject_duplicate_primary_key,
        }

    def run_errors(self, df_test: pd.DataFrame, table_name: str, run_number: int) -> pd.DataFrame:
        self.df = df_test.copy()
        self.table_name = table_name
        self.run_number = run_number

        # Randomly choose how many errors will be injected in the given table (with maximum 5)
        nb_error = random.randint(0, 5)

        # Randomly choose the type of errors that will be injected
        all_errors = random.choices(list(self.error_types_dict.keys()), k=nb_error)
        print(f"Running {nb_error} errors in {table_name}. Errors: {all_errors}")

        # Injecting errors in the table
        for error_name in all_errors:
            func_error = self.error_types_dict[error_name]

            res = None

            # Attributing the correct arguments per function
            if error_name == "wrong_datatype":
                res = func_error(self.df, self.table_name)
            elif error_name == "insert_null":
                res = func_error(self.df, self.table_name)
            elif error_name == "duplicate_rows":
                res = func_error(self.df)
            elif error_name == "insert_column":
                res = func_error(self.df)
            elif error_name == "orphan_foreign_key":
                res = func_error(self.df, self.table_name)
            elif error_name == "new_category":
                res = func_error(self.df, self.table_name)
            elif error_name == "correlation_break":
                res = func_error(self.df, self.table_name)
            elif error_name == "duplicate_primary_key":
                res = func_error(self.df, self.table_name)

            # Checking if the error was actually injected
            if res is not None:
                print(f"\tError {error_name} was injected. Creating Log")
                df, params = res
                self.df = df.copy()

                # Adding a log of the error injected
                append_injection_logs(
                    table_name=self.table_name,
                    error_type=error_name,
                    run_number=self.run_number,
                    params=params
                )

        return self.df


if __name__ == "__main__":
    from data.sqlite_connector import connecting_to_sqlite
    from config import KAGGLE_DATASET_NAME

    conn_test = connecting_to_sqlite(KAGGLE_DATASET_NAME, database_type="test")

    table_name = "application_record"
    table_name = "olist_products_dataset"
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn_test)
    df_raw = df.copy()

