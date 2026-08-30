import pandas as pd
from app.errors_injection import inject_new_column, inject_duplicate_rows


def test_inject_new_column():
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


def test_inject_duplicate_rows():
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
    assert len(df_injected) == params["total_nb_rows_before_dups"] + params["nb_duplicated_rows_inserted"]
