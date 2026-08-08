import pandas as pd
from errors_injection.errors_injections_models import inject_new_column


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
        "col_error": "COLUMN_A",
        "name_new_column": "NEW_COLUMN_A",
        "total_nb_rows": 3
    }


