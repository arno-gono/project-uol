import pandas as pd
from app.data_calibration import (_get_general_data, _get_profile_datatype, _is_datetime_column,
                                  _get_cramers_v, _get_correlation_dict,
                                  _get_profile_cardinality_distribution, _is_primary_key,
                                  _is_null_allowed, _get_primary_key_columns, _get_fk_coverage)

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


def test_get_correlation_dict():
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3, 4],
            "COLUMN_B": [2, 4, 6, 8],
            "COLUMN_C": [4, 3, 2, 1],
            "COLUMN_D": ["a", "b", "c", "d"],
        }
    )

    # A pair is only kept once, a column is not paired with itself, and the string column is left out
    assert _get_correlation_dict(df) == {
        "COLUMN_A | COLUMN_B": 1.0,
        "COLUMN_A | COLUMN_C": -1.0,
        "COLUMN_B | COLUMN_C": -1.0,
    }

    # A column holding the same value gives a nan correlation, which is skipped
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3, 4],
            "COLUMN_B": [5, 5, 5, 5],
        }
    )

    assert _get_correlation_dict(df) == {}


def test_get_profile_cardinality_distribution():
    # Categories are stored with the share of the rows they cover
    assert _get_profile_cardinality_distribution(pd.Series(["a", "a", "b", "c"])) == {"a": 0.5, "b": 0.25, "c": 0.25}


def test_is_primary_key():
    assert _is_primary_key(pd.Series(["a", "b", "c"])) is True
    assert _is_primary_key(pd.Series(["a", "b", "b"])) is False


def test_is_null_allowed():
    assert _is_null_allowed(pd.Series([1, 2, 3])) is False
    assert _is_null_allowed(pd.Series([1, None, 3])) is True


def test_get_primary_key_columns():
    dict_metadata = {
        "TABLE_A": {
            "columns_details": {
                "COLUMN_A": {"potential_primary_key": True},
                "COLUMN_B": {"potential_primary_key": False},
            }
        },
        "TABLE_B": {
            "columns_details": {
                "COLUMN_A": {"potential_primary_key": True},
                "COLUMN_C": {"potential_primary_key": True},
            }
        },
    }

    # COLUMN_A is a potential primary key in both tables
    assert _get_primary_key_columns(dict_metadata) == {
        "COLUMN_A": ["TABLE_A", "TABLE_B"],
        "COLUMN_C": ["TABLE_B"],
    }


def test_get_fk_coverage():
    # Two of the three child values are found in the parent key
    assert _get_fk_coverage(child_series=pd.Series(["a", "b", "c", "x"]), parent_series=pd.Series(["a", "b", "c", "d"])) == 0.75

    # Rows holding no key are left out of the calculation
    assert _get_fk_coverage(child_series=pd.Series(["a", "b", None]), parent_series=pd.Series(["a", "b", "c"])) == 1.0

    # A column holding no key has nothing to measure
    assert _get_fk_coverage(child_series=pd.Series([None, None]), parent_series=pd.Series(["a", "b", "c"])) == 0.0


