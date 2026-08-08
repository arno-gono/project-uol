import pytest
import pandas as pd
from data_calibration.calibration_1st_layer import (_get_general_data, _get_profile_datatype, _is_datetime_column,
                                                    _get_cramers_v)

def test_get_general_data():
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3],
            "COLUMN_B": ["a", "b", "c"],
            "COLUMN_C": ["d", "e", "f"],
            "COLUMN_D": [1.1, 2.2, 3.3],
        }
    )

    assert _get_general_data(df) == {
        "nb_entries": 3,
        "nb_columns": 4,
        "nb_duplicated_rows": 0,
        "duplicates_distribution": 0.0
    }

    # Inserting duplicates
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3, 1],
            "COLUMN_B": [1, 3, 5, 1],
            "COLUMN_C": [1, 4, 7, 1],
            "COLUMN_D": [1, 5, 9, 1],
        }
    )

    assert _get_general_data(df) == {
        "nb_entries": 4,
        "nb_columns": 4,
        "nb_duplicated_rows": 1,
        "duplicates_distribution": 0.25
    }


def test_get_profile_datatype():
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3],
            "COLUMN_B": [1.1, 2.2, 3.3],
            "COLUMN_C": ["a", "b", "c"],
            "COLUMN_D": [True, False, True],
            "COLUMN_E": ["Y", "Y", "No"],
        }
    )

    assert _get_profile_datatype(df["COLUMN_A"].dtype, df["COLUMN_A"]) == "int"
    assert _get_profile_datatype(df["COLUMN_B"].dtype, df["COLUMN_B"]) == "float"
    assert _get_profile_datatype(df["COLUMN_C"].dtype, df["COLUMN_C"]) == "str"
    assert _get_profile_datatype(df["COLUMN_D"].dtype, df["COLUMN_D"]) == "bool"

    # Y/N columns and 0/1 columns are remapped to bool, timestamps stored as strings are read as datetime
    df = pd.DataFrame(
        {
            "COLUMN_A": ["Y", "N", "Y"],
            "COLUMN_B": [0, 1, 1],
            "COLUMN_C": ["2020-01-01", "2021-06-05", "2022-12-31"],
        }
    )

    assert _get_profile_datatype(df["COLUMN_A"].dtype, df["COLUMN_A"]) == "bool"
    assert _get_profile_datatype(df["COLUMN_B"].dtype, df["COLUMN_B"]) == "bool"
    assert _get_profile_datatype(df["COLUMN_C"].dtype, df["COLUMN_C"]) == "datetime"

    # A column holding mixed types is stored as object, which is not referenced in type_map
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, "a", 2.5],
        }
    )

    assert _get_profile_datatype(df["COLUMN_A"].dtype, df["COLUMN_A"]) is None


def test_is_datetime_column():
    # Dates written as strings are the case this function is meant to catch
    assert _is_datetime_column(pd.Series(["2020-01-01", "2021-06-05", "2022-12-31"])) is True
    assert _is_datetime_column(pd.Series(["alpha", "beta", "gamma"])) is False
    assert _is_datetime_column(pd.Series(["20200101", "20210605", "20221231"])) is False
    assert _is_datetime_column(pd.Series(["2020-01-01", "alpha", "2021-06-05", "beta"])) is False


def test_get_cramers_v():
    # COLUMN_A fully determines COLUMN_B, so the association is 1
    assert _get_cramers_v(pd.Series(["a", "a", "b", "b"]), pd.Series(["x", "x", "y", "y"])) == 1.0

    # Example with no association to measure
    assert _get_cramers_v(pd.Series(["a", "a", "a", "a"]), pd.Series(["x", "x", "y", "y"])) == 0.0


