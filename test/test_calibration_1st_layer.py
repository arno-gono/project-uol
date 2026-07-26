import pytest
import pandas as pd
from data_calibration.calibration_1st_layer import _get_general_data

def test_get_general_data():
    df = pd.DataFrame(
        {
            "COLUMN_A": [1, 2, 3],
            "COLUMN_B": ["a", "b", "c"],
            "COLUMN_C": ["d", "e", "f"],
            "COLUMN_D": [1.1, 2.2, 3.3],
        }
    )

    assert _get_general_data(df) == {"nb_entries": 3, "nb_columns": 4}

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
        "duplicates_distribution": 0.25
    }


