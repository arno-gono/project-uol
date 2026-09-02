import json
import os
from typing import Any
from app.config import (KAGGLE_DATASET_NAME, DB_DIR, DB_NAME, MAX_CARDINALITY_NB, MIN_DATETIME_PARSE_RATIO,
                        KAGGLE_TABLE_MAX_ROWS)
from app.data.sqlite_connector import connecting_to_sqlite
import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import contingency
from app.data.utils import read_column_from_whole_dataset
from app.data_calibration.calibration_cluster import get_ml_profile


# remapping numpy / pandas datatypes to Python built-ins
type_map = {
    "int64": "int",
    "float64": "float",
    "str": "str",
    "bool": "bool",
    "datetime64[ns]": "datetime",
    "datetime64[us]": "datetime",
    "datetime": "datetime",
}


def _get_correlation_dict(df: pd.DataFrame) -> dict[str, float]:
    d_correl: dict[str, float] = {}

    d_correl_raw = df.corr(numeric_only=True).to_dict()

    # half the values repeat themselves. Looping data to only keep pairs once
    for k1, v1 in d_correl_raw.items():
        for k2, v2 in v1.items():

            # Skipping data that will not be useful to the agent
            if f"{k1} | {k2}" in d_correl or f"{k2} | {k1}" in d_correl or k1 == k2 or str(v2) == "nan":
                continue

            # Adding the correlation value to a temp dictionary
            d_correl[f"{k1} | {k2}"] = round(v2, 4)

    return d_correl


def _get_cramers_v(series_a: pd.Series, series_b: pd.Series) -> float:
    # Cramer's V measures the association between two categorical columns.
    # The output ranges from 0 (independent) to 1 (one column fully determines the other).
    # Reference: Cramer, H. (1946). Mathematical Methods of Statistics.
    # Princeton University Press, Section 21.4.
    # Explanation and worked example: https://www.geeksforgeeks.org/data-analysis/calculate-cramer-s-coefficient-matrix-using-pandas/
    df_contingency = pd.crosstab(series_a, series_b)

    # A column left with a single category carries no association to measure.
    if min(len(df_contingency.columns), len(df_contingency)) < 2:
        return 0.0

    # scipy's association() returns V directly.
    # Documentation:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html
    return float(contingency.association(df_contingency, method="cramer"))


def _get_categorical_association_dict(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, float]:
    d_association = {}

    # Looping through each categorical column, calculating Cramer's V value and adding it to a dict
    for i in range(len(categorical_cols)):
        for j in range(len(categorical_cols)):
            if i < j:
                col_i = categorical_cols[i]
                col_j = categorical_cols[j]
                d_association[f"{col_i} | {col_j}"] = round(_get_cramers_v(df[col_i], df[col_j]), 4)

    return d_association


def _get_correlation_ratio(categorical_series: pd.Series, numerical_series: pd.Series) -> float:
    # The correlation ratio measures the association between a categorical column and a numerical one.
    # It ranges from 0 to 1 (low to high association).
    # Reference: Pearson, K. (1905). On the General Theory of Skew Correlation and Non-linear
    # Regression. Drapers' Company Research Memoirs, Biometric Series II.
    # Formula: https://en.wikipedia.org/wiki/Correlation_ratio
    # Programming implementation: https://github.com/shakedzy/dython/blob/master/dython/nominal.py

    df_pairs = pd.DataFrame({"category": categorical_series, "value": numerical_series}).dropna()

    if df_pairs.empty:
        return 0.0

    measurements = df_pairs["value"].to_numpy()

    # factorize() puts a number on categories.
    fcat, _ = pd.factorize(df_pairs["category"])
    cat_num = fcat.max() + 1

    n_array = np.zeros(cat_num)
    y_avg_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(n_array * y_avg_array) / np.sum(n_array)
    numerator = np.sum(n_array * (y_avg_array - y_total_avg) ** 2)
    denominator = np.sum((measurements - y_total_avg) ** 2)

    if numerator == 0:
        return 0.0

    return float(np.sqrt(numerator / denominator))


def _get_mixed_association_dict(df: pd.DataFrame, categorical_cols: list[str],
                                numerical_cols: list[str]) -> dict[str, float]:
    d_association = {}

    # Looping through categorical and numerical columns, calculating the correlation ratio and keeping it in a dict
    for col_categorical in categorical_cols:
        for col_numerical in numerical_cols:
            # Some categorical columns are also numerical columns. Skipping this case
            if col_categorical == col_numerical:
                continue

            d_association[f"{col_categorical} | {col_numerical}"] = round(
                _get_correlation_ratio(df[col_categorical], df[col_numerical]), 4)

    return d_association


def _get_general_data(df: pd.DataFrame) -> dict[str, Any]:
    # Main dict of metadata, containing nested dictionaries.
    dict_metadata = {}

    # Main data
    nb_entries = len(df)
    nb_columns = len(df.columns)

    dict_metadata["nb_entries"] = nb_entries
    dict_metadata["nb_columns"] = nb_columns

    # Check if there are duplicated rows and if yes, how many as a percentage
    nb_duplicates = int(df.duplicated().sum())
    dict_metadata["nb_duplicated_rows"] = nb_duplicates
    dict_metadata["duplicates_distribution"] = round(nb_duplicates / nb_entries, 4) if nb_entries else 0.0

    return dict_metadata


def _is_datetime_column(pd_series: pd.Series) -> bool:
    # Some timestamps are stored as plain strings. Trying to cast the column as datetime
    non_null_values = pd_series.dropna()

    # errors="coerce" turns strings that cannot be read as a date into NaT instead of raising allowing tolerance
    # for errors.
    # format="mixed" reads each value on its own, keeping day-first dates dd/mm/yyyy
    # from being dropped because the first value looked month-first
    parsed_values = pd.to_datetime(non_null_values, errors="coerce", format="mixed")

    # Requiring a date separator to filter out plain numbers that can be read as dates
    has_separator = non_null_values.str.contains(r"[-/:]", regex=True)

    return bool((parsed_values.notna() & has_separator).mean() >= MIN_DATETIME_PARSE_RATIO)


def _get_profile_datatype(dtype: str, col_series: pd.Series) -> str | None:
    temp_type = str(dtype)

    # Checking if this data type is referenced in type_map
    if temp_type not in type_map:
        # Add a log rather than printing on console only
        e = f"{temp_type} is not referenced in type_map"
        print(e)
        return None

    # Y/N columns and 0/1 columns are remapped to bool.
    # Using issubset as some data actually have only one value that seems trivial to be marked as True or False.
    if set(col_series.unique()).issubset({"Y", "N"}) or set(col_series.unique()).issubset({0, 1}):
        temp_type = "bool"

    # Some timestamps are stored as strings. Trying to cast as a datetime object
    if temp_type == "str" and _is_datetime_column(col_series):
        temp_type = "datetime"

    return type_map[temp_type]


def _get_column_unique_values(pd_series: pd.Series) -> list[Any]:
    # Return unique values of a dataframe column / pandas series.
    return list(pd_series.unique())


def _get_profile_cardinality_distribution(pd_series: pd.Series) -> dict[Any, float] | None:
    # Getting unique values of the column
    unique_values = _get_column_unique_values(pd_series)

    # Filtering out columns that do not seem to be categorical with an absolute threshold. Might be refined later.
    if len(unique_values) <= MAX_CARDINALITY_NB:
        dict_categories: dict[Any, float] = {}

        # looping through the value_counts method, listing each category and the absolute number of times they
        # appear in the table
        for i, value in pd_series.value_counts(normalize=True).items():
            dict_categories[i] = value  # Storing the relative value

        return dict_categories

    return None


def _is_primary_key(pd_series: pd.Series, datatype: str | None) -> bool:
    # Considering that a float is never used as a key. This excludes fields like longitude / latitude for example.
    if datatype == "float":
        return False

    # If all values are unique, the column might be a potential primary key
    # Getting unique values of the column
    unique_values = _get_column_unique_values(pd_series)

    return len(unique_values) == len(pd_series)


def _is_null_allowed(pd_series: pd.Series) -> bool:
    if pd_series.isna().any():
        return True
    else:
        return False


def _get_distribution_for_numerical_fields(col_name: str, df: pd.DataFrame) -> dict[Any, Any] | None:
    if col_name in df.columns:
        return df[col_name].dropna().to_dict()

    return None


def _get_metadata_profiling_from_table(table_name: str, co: sqlite3.Connection) -> dict[str, Any]:

    # Getting the actual table to gather data
    df = pd.read_sql(f"SELECT * FROM {table_name}", co)

    # df.describe() returns distribution data already that will be saved.
    # Some timestamps are stored as strings. Casting them before profiling
    # so describe() shows a min, max and percentiles
    for col_name in df.columns:
        if str(df[col_name].dtype) == "str" and _is_datetime_column(df[col_name]):
            df[col_name] = pd.to_datetime(df[col_name], errors="coerce", format="mixed")

    # Adding a couple percentiles to data profiling
    df_dist = df.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], include="all")

    # Data profiling for the table. Number of entries, columns
    dict_metadata = _get_general_data(df=df)

    # Next step is to get data profiling for each column. Stored in the key columns_details
    dict_metadata["columns_details"] = {}

    for col_name, dtype in df.dtypes.items():
        # Getting the column's datatype
        datatype = _get_profile_datatype(dtype, df[col_name])

        # Columns' metadata are stored in a dictionary and then aggregated to the table's metadata dictionary
        dict_column: dict[str, Any] = {"datatype": datatype,
                                       "cardinality_distribution": _get_profile_cardinality_distribution(df[col_name]),
                                       "potential_primary_key": _is_primary_key(df[col_name], datatype),
                                       "null_values": _is_null_allowed(df[col_name]),
                                       "values_distribution": _get_distribution_for_numerical_fields(col_name, df_dist)}

        # Adding column's metadata to the table metadata
        dict_metadata["columns_details"][col_name] = dict_column

    # Ending with correlations between columns: simple correlations, categorical associations using Cramer's V method
    # and another one.
    dict_metadata["correlations"] = _get_correlation_dict(df=df)

    # Most analysis on relationship is run on numerical columns.
    # Cramer's V model calculates association between categorical columns.
    dict_metadata["cramers_v_categorical_associations"] = _get_categorical_association_dict(
        df=df, categorical_cols=_get_categorical_columns(dict_metadata["columns_details"]))

    # The profiling that runs on numerical columns is added by _add_numerical_profiling once every table has
    # been through this function, as picking those columns needs the foreign keys of the whole dataset.
    return dict_metadata


def _get_categorical_columns(columns_details: dict[str, Any]) -> list[str]:
    # Categorical columns are the ones the cardinality was measured on, keys left aside.
    return [col_name for col_name, details in columns_details.items()
            if details["cardinality_distribution"] is not None
            and not details["potential_primary_key"]]


def _get_referenced_primary_keys(dict_metadata: dict[str, Any]) -> dict[str, list[str]]:
    # Listing every primary key that another table points at through a foreign key, with the table holding it
    d_referenced_keys = {}

    for table_details in dict_metadata.values():

        # Foreign keys are mapped once every table has been profiled, so the key is not always there yet
        if "potential_foreign_key" not in table_details:
            continue

        for foreign_key in table_details["potential_foreign_key"].values():
            parent_table = foreign_key["parent_table"]
            parent_column = foreign_key["parent_column"]

            if parent_table in d_referenced_keys:
                if parent_column not in d_referenced_keys[parent_table]:
                    d_referenced_keys[parent_table].append(parent_column)
            else:
                d_referenced_keys[parent_table] = [parent_column]

    return d_referenced_keys


def _get_numerical_columns(table_name: str, dict_metadata: dict[str, Any]) -> list[str]:
    # Numerical columns that the statistical and the machine learning profiling can be run on.
    # Excluding columns flagged as potential primary keys and that have been linked to a foreign key in another table.
    # This is to avoid attributes like longitude or latitude being considered as primary key and then excluded from
    # being injected errors in.
    d_referenced_keys = _get_referenced_primary_keys(dict_metadata)

    # Only the keys of this table matter here
    referenced_columns = d_referenced_keys[table_name] if table_name in d_referenced_keys else []

    return [col_name for col_name, details in dict_metadata[table_name]["columns_details"].items()
            if details["datatype"] in ["int", "float"]
            and col_name not in referenced_columns]


def _add_numerical_profiling(table_name: str, dict_metadata: dict[str, Any], co: sqlite3.Connection) -> None:
    # Profiling that depends on the numerical columns, hence run once the foreign keys of the whole dataset
    # are known. The metadata of the table passed as argument is completed in place.
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', co)

    categorical_columns = _get_categorical_columns(dict_metadata[table_name]["columns_details"])
    numerical_columns = _get_numerical_columns(table_name=table_name, dict_metadata=dict_metadata)

    # Run on the raw column as a placeholder tied to a single category, ie a disguised missing value,
    # shows up here as a ratio close to 1 and tells the agent which category the placeholder stands for.
    dict_metadata[table_name]["categorical_numerical_associations"] = _get_mixed_association_dict(
        df=df, categorical_cols=categorical_columns, numerical_cols=numerical_columns)

    # Machine Learning approach: Dimensionality reduction, Disguised Missing Values and KMeans clusters
    if len(numerical_columns) >= 2:
        dict_metadata[table_name]["ml_calibration"] = get_ml_profile(df=df, numerical_columns=numerical_columns)
    else:
        dict_metadata[table_name]["ml_calibration"] = (
            f"Not enough numerical columns to run ML profiling ({len(numerical_columns)} column)")

    return None


def _get_primary_key_columns(dict_metadata: dict[str, Any]) -> dict[str, list[str]]:
    # Listing every column flagged as a potential primary key, with the tables that are holding it
    d_primary_keys = {}

    for table_name, table_details in dict_metadata.items():
        for col_name, col_details in table_details["columns_details"].items():
            if col_details["potential_primary_key"]:
                if col_name in d_primary_keys:
                    d_primary_keys[col_name].append(table_name)
                else:
                    d_primary_keys[col_name] = [table_name]

    return d_primary_keys


def _get_fk_coverage(child_series: pd.Series, parent_series: pd.Series) -> float:
    # Share of the child's values that are found in the parent key
    child_series = child_series.dropna()

    if len(child_series) == 0:
        return 0.0

    return round(float(child_series.isin(set(parent_series)).mean()), 4)


def _get_foreign_keys_dict(table_name: str, dict_metadata: dict[str, Any], co_clean: sqlite3.Connection,
                           co_test: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    # A column is a candidate foreign key when the values of this column are actually found in the parent key column.
    # Only the values decide: the child column and the parent key do not have to carry the same name, but if all
    # values of the child column are included in the parent column (within threshold) then it is considered as a
    # potential foreign key.
    d_primary_keys = _get_primary_key_columns(dict_metadata)

    # Flattening the primary keys into (parent table, parent column) pairs. Every column of the profiled table is
    # compared against every primary key held by another table.
    parent_keys = [(parent_table, parent_column)
                   for parent_column, parent_tables in d_primary_keys.items()
                   for parent_table in parent_tables
                   if parent_table != table_name]

    if not parent_keys:
        return {}

    d_foreign_keys = {}

    # Minimum share of a column's values that must be found in the parent key for the pair to be recorded as a
    # foreign key. The data is truncated (KAGGLE_TABLE_MAX_ROWS) so a tolerance is accepted, for tests purposes.
    min_foreign_key_coverage = 0.95 if KAGGLE_TABLE_MAX_ROWS else 1

    # The same parent key is compared against every column of the table. Reading it from SQLite once and keeping
    # it in memory, as the comparison is now run on all the pairs instead of the ones sharing a column name.
    parent_series_cache = {}

    for col_name in dict_metadata[table_name]["columns_details"]:
        child_series = read_column_from_whole_dataset(table_name, col_name, co_clean, co_test)

        for parent_table, parent_column in parent_keys:

            if (parent_table, parent_column) not in parent_series_cache:
                parent_series_cache[(parent_table, parent_column)] = read_column_from_whole_dataset(
                    parent_table, parent_column, co_clean, co_test)

            coverage = _get_fk_coverage(child_series=child_series,
                                        parent_series=parent_series_cache[(parent_table, parent_column)])

            if coverage < min_foreign_key_coverage:
                continue

            # A column can match several parents. Only the best covered one is kept as the relationship.
            if col_name in d_foreign_keys and d_foreign_keys[col_name]["coverage"] >= coverage:
                continue

            # Keeping a relationship foreign to primary key mapping in the calibration.
            d_foreign_keys[col_name] = {
                "parent_table": parent_table,
                "parent_column": parent_column,
                "coverage": coverage,
            }

    return d_foreign_keys


def _aggregate_metadata(kaggle_dataset: str = KAGGLE_DATASET_NAME) -> dict[str, Any]:
    # This function loops through all tables and views available in the dataset passed as argument.
    # For each table, we save the metadata as a dictionary, and aggregate them in a list of dictionaries.
    # Once all the data is collected, it is written as a json file.

    conn = connecting_to_sqlite(kaggle_dataset, database_type="clean")

    # Getting the scope of tables / views
    df = pd.read_sql("SELECT * FROM sqlite_master", conn)

    dict_metadata = {}

    for n, t in zip(df["name"], df["type"]):
        print(f"Starting calibration for {t} {n}")
        dict_table_metadata = _get_metadata_profiling_from_table(table_name=n, co=conn)
        dict_metadata[n] = {"type": t} | dict_table_metadata

    # Potential foreign keys are searched once all tables have been profiled
    # and potential primary keys have been identified.
    conn_test = connecting_to_sqlite(kaggle_dataset, database_type="test")

    for table_name in dict_metadata:
        print(f"Looking for foreign keys in {table_name}")
        dict_metadata[table_name]["potential_foreign_key"] = _get_foreign_keys_dict(
            table_name=table_name, dict_metadata=dict_metadata, co_clean=conn, co_test=conn_test)

    # The profiling running on numerical columns comes last: a key another table points at is left out of those
    # columns, which is only known once every foreign key of the dataset has been mapped.
    for table_name in dict_metadata:
        print(f"Profiling numerical columns of {table_name}")
        _add_numerical_profiling(table_name=table_name, dict_metadata=dict_metadata, co=conn)

    conn.close()
    conn_test.close()

    return dict_metadata


def _round_floats(data: Any, nb_decimals: int = 4) -> Any:
    # Rounding all numbers to a maximum of nb_decimals. Storing data with 10+ floating decimals which
    # are passed on to the agent for every single request to the API. Trimming float numbers lightens the size
    # of the calibration file, and therefore the cost for the investigations.
    if isinstance(data, float):
        # numpy floats are a subclass of float, casting keeps the value a plain float in the JSON file
        return float(round(data, nb_decimals))

    if isinstance(data, dict):
        # Error with timestamp that could not be written in JSON. Default casting to strings.
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _round_floats(v, nb_decimals)
                for k, v in data.items()}

    if isinstance(data, list):
        return [_round_floats(v, nb_decimals) for v in data]

    return data


def _save_metadata(dict_data: dict[str, Any], dataset_name: str = DB_NAME) -> None:
    # Saving data under the name of the dataset name
    path_json_file = os.path.join(DB_DIR, f"{dataset_name}.json")

    with open(path_json_file, "w") as f:
        json.dump(_round_floats(dict_data), f, indent=4, default=str)

    return None


def dataset_calibration() -> dict[str, Any]:
    # Full analysis of each table of the dataset
    dict_metadata = _aggregate_metadata()

    # Saving the data as a json file
    _save_metadata(dict_data=dict_metadata, dataset_name=DB_NAME)

    return dict_metadata


if __name__ == "__main__":
    # d_metadata = dataset_calibration()

    table_name = "olist_orders_dataset"
    table_name = "application_record"
    kaggle_dataset = KAGGLE_DATASET_NAME
    # co = connecting_to_sqlite(kaggle_dataset, database_type="clean")
