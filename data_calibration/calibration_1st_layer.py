import json
import os
from typing import Any
from config import KAGGLE_DATASET_NAME, DB_DIR, DB_NAME, MAX_CARDINALITY_NB, MIN_DATETIME_PARSE_RATIO
from data.sqlite_connector import connecting_to_sqlite
import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import contingency
from data_calibration.calibration_cluster import get_ml_profile


# remapping numpy / pandas datatypes to Python built-ins
type_map = {
    "int64": "int",
    "float64": "float",
    "str": "str",
    "bool": "bool",
    "datetime64[ns]": "datetime",
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
    dict_metadata: dict[str, Any] = {}

    # Main data
    nb_entries = len(df)
    nb_columns = len(df.columns)

    dict_metadata["nb_entries"] = nb_entries
    dict_metadata["nb_columns"] = nb_columns

    if df.duplicated().any():
        dict_metadata["duplicates_distribution"] = df.duplicated().sum() / nb_entries

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


def _is_primary_key(pd_series: pd.Series, total_entries: int) -> bool:
    # If all values are unique, the column might be a potential primary key
    # Getting unique values of the column
    unique_values = _get_column_unique_values(pd_series)

    return len(unique_values) == total_entries


def _is_null_allowed(pd_series: pd.Series) -> bool:
    if pd_series.isna().any():
        return True
    else:
        return False


def _get_distribution_for_numerical_fields(col_name, df: pd.DataFrame) -> dict[Any, Any] | None:
    if col_name in df.columns:
        return df[col_name].to_dict()

    return None


def _get_metadata_profiling_from_table(table_name: str, co: sqlite3.Connection) -> dict[str, Any]:

    # Getting the actual table to gather data
    df = pd.read_sql(f"SELECT * FROM {table_name}", co)

    # df.describe() returns distribution data already that will be saved.
    # Adding a couple percentiles to data profiling
    df_dist = df.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    # Data profiling for the table. Number of entries, columns
    dict_metadata = _get_general_data(df=df)

    # Next step is to get data profiling for each column. Stored in the key columns_details
    dict_metadata["columns_details"] = {}

    for col_name, dtype in df.dtypes.items():
        # Columns' metadata are stored in a dictionary and then aggregated to the table's metadata dictionary
        dict_column: dict[str, Any] = {"datatype": _get_profile_datatype(dtype, df[col_name]),
                                       "cardinality_distribution": _get_profile_cardinality_distribution(df[col_name]),
                                       "potential_primary_key": _is_primary_key(df[col_name], total_entries=len(df)),
                                       "null_values": _is_null_allowed(df[col_name]),
                                       "values_distribution": _get_distribution_for_numerical_fields(col_name, df_dist)}

        # Adding column's metadata to the table metadata
        dict_metadata["columns_details"][col_name] = dict_column

    # Ending with correlations between columns: simple correlations, categorical associations using Cramer's V method
    # and another one.
    dict_metadata["correlations"] = _get_correlation_dict(df=df)

    # Most analysis on relationship is run on numerical columns.
    # Cramer's V model calculates association between categorical columns.
    categorical_columns = [col for col, details in dict_metadata["columns_details"].items()
                           if details["cardinality_distribution"] is not None
                           and not details["potential_primary_key"]]

    dict_metadata["cramers_v_categorical_associations"] = _get_categorical_association_dict(
        df=df, categorical_cols=categorical_columns)

    # Passing only numerical values and non-primary keys
    numerical_columns = [c for c in dict_metadata["columns_details"]
                         if dict_metadata["columns_details"][c]["datatype"] in ["int", "float"]
                         and dict_metadata["columns_details"][c]["potential_primary_key"] is False]

    # Last pairing: the correlation ratio associates a categorical column with a numerical one.
    # Run on the raw column as a placeholder tied to a single category, ie a disguised missing value,
    # shows up here as a ratio close to 1 and tells the agent which category the placeholder stands for.
    dict_metadata["categorical_numerical_associations"] = _get_mixed_association_dict(
        df=df, categorical_cols=categorical_columns, numerical_cols=numerical_columns)

    # Machine Learning approach: Dimensionality reduction, Disguised Missing Values and KMeans clusters
    if len(numerical_columns) >= 2:
        dict_metadata["ml_calibration"] = get_ml_profile(df=df, numerical_columns=numerical_columns)
    else:
        dict_metadata["ml_calibration"] = f"Not enough numerical columns to run ML profiling ({len(numerical_columns)} column)"
    return dict_metadata


def _aggregate_metadata(kaggle_dataset: str = KAGGLE_DATASET_NAME) -> dict[str, Any]:
    # This function loops through all tables and views available in the dataset passed as argument.
    # For each table, we save the metadata as a dictionary, and aggregate them in a list of dictionaries.
    # Once all the data is collected, it is written as a json file.

    conn = connecting_to_sqlite(kaggle_dataset)

    # Getting the scope of tables / views
    df = pd.read_sql("SELECT * FROM sqlite_master", conn)

    dict_metadata: dict[str, Any] = {}

    for n, t in zip(df["name"], df["type"]):
        dict_table_metadata = _get_metadata_profiling_from_table(table_name=n, co=conn)
        dict_metadata[n] = {"type": t} | dict_table_metadata

    conn.close()

    return dict_metadata


def _save_metadata(dict_data: dict, dataset_name: str = DB_NAME):
    # Saving data under the name of the dataset name
    path_json_file = os.path.join(DB_DIR, f"{dataset_name}.json")

    with open(path_json_file, "w") as f:
        json.dump(dict_data, f, indent=4, default=str)

    return None


def dataset_calibration():
    # Full analysis of each table of the dataset
    dict_metadata = _aggregate_metadata()

    # Saving the data as a json file
    _save_metadata(dict_data=dict_metadata, dataset_name=DB_NAME)

    return dict_metadata


if __name__ == "__main__":
    d_metadata = dataset_calibration()

    # table_name = "olist_products_dataset"
    # table_name = "application_record"
    # kaggle_dataset: str = KAGGLE_DATASET_NAME
    # co = connecting_to_sqlite(kaggle_dataset)
