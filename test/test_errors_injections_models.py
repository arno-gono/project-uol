import pandas as pd
from app.errors_injection.errors_injections_models import (inject_new_column, inject_duplicate_rows, inject_nulls,
                                                              inject_out_of_range)


def test_inject_new_column() -> None:
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3],
        }
    )

    df_injected, params = inject_new_column(df)

    # The new column holds the exact same data as the column that was picked up
    assert list(df_injected.columns) == ["COLUMN_A", "NEW_COLUMN_A"]
    assert list(df_injected["NEW_COLUMN_A"]) == [1, 2, 3]

    assert params == {
        "column": "NEW_COLUMN_A",
        "column_copied": "COLUMN_A",
        "total_nb_rows": 3
    }


def test_inject_duplicate_rows() -> None:
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3, 4, 5],
            "COLUMN_B": ["a", "b", "c", "d", "e"],
        }
    )

    # The rows being duplicated are picked randomly
    df_injected, params = inject_duplicate_rows(df)

    assert len(df_injected) >= len(df)
    assert len(df_injected.drop_duplicates()) == len(df)
    assert len(df) == params["total_nb_rows_before_dups"]
    assert len(df_injected) == params["total_nb_rows_before_dups"] + params["nb_data_corrupted"]


def test_inject_nulls() -> None:
    # COLUMN_A had no NULL during the calibration, COLUMN_B already had some
    d_calibration = {"TABLE_A": {"columns_details": {"COLUMN_A": {"null_values": False},
                                                     "COLUMN_B": {"null_values": True}}}}

    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3, 4, 5],
            "COLUMN_B": ["a", "b", "c", "d", "e"],
        }
    )

    # The share of rows corrupted is random, the column is not: only COLUMN_A can receive the NULLs
    df_injected, params = inject_nulls(df, "TABLE_A", d_calibration=d_calibration)

    assert params["column"] == "COLUMN_A"
    assert params["nb_data_corrupted"] == int(df_injected["COLUMN_A"].isna().sum())
    assert df_injected["COLUMN_B"].isna().sum() == 0


def test_inject_out_of_range() -> None:
    # COLUMN_A stayed between 0 and 100 during the calibration. COLUMN_B is a primary key, never pushed out of range.
    d_calibration = {"TABLE_A": {"columns_details": {
        "COLUMN_A": {"datatype": "float", "potential_primary_key": False,
                     "values_distribution": {"min": 0, "max": 100}},
        "COLUMN_B": {"datatype": "int", "potential_primary_key": True,
                     "values_distribution": {"min": 0, "max": 9999}},
    }}}

    # Enough rows for the random share of corrupted rows (0.1% to 5%) to never round down to zero
    df = pd.DataFrame(
        {
            "COLUMN_A": [float(i % 101) for i in range(10_000)],
            "COLUMN_B": list(range(10_000)),
        }
    )

    df_injected, params = inject_out_of_range(df, "TABLE_A", d_calibration=d_calibration)

    # Every corrupted value sits outside the calibrated bounds, and only those
    out_of_range = (df_injected["COLUMN_A"] < 0) | (df_injected["COLUMN_A"] > 100)

    assert params["column"] == "COLUMN_A"
    assert params["nb_data_corrupted"] == int(out_of_range.sum())
    assert list(df_injected.loc[out_of_range].index) == params["index_row_corrupted"]
    assert list(df_injected["COLUMN_B"]) == list(range(10_000))
