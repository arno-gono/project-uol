import sqlite3
from agent.agent_tools import run_sql, read_calibration
from agent.agent_run import _count_error_types
from app.config import AGENT_MAX_ROWS_RETURNED, AGENT_READS_ML_CALIBRATION


def test_run_sql_only_select() -> None:
    # Refused before any connection to the database is opened
    assert run_sql("DELETE FROM listings") == {"error": "only SELECT queries are allowed"}
    assert run_sql("DROP TABLE listings") == {"error": "only SELECT queries are allowed"}
    assert run_sql("UPDATE listings SET price = 0") == {"error": "only SELECT queries are allowed"}

    # Lower case, spaces and a semicolon around the query
    assert run_sql("  insert into listings values (1);  ") == {"error": "only SELECT queries are allowed"}


def test_read_calibration() -> None:
    d_calibration = {"TABLE_A": {"columns_details": {}, "ml_calibration": {"pca": []}}}

    # An unknown table returns the tables calibrated, so the agent can correct itself
    assert read_calibration("TABLE_B", d_calibration=d_calibration) == {
        "error": "TABLE_B was not calibrated",
        "calibrated_tables": ["TABLE_A"]
    }

    # The first layer is always returned, the second one only when AGENT_READS_ML_CALIBRATION is on
    d_profile = read_calibration("TABLE_A", d_calibration=d_calibration)

    assert d_profile["columns_details"] == {}
    assert ("ml_calibration" in d_profile) == AGENT_READS_ML_CALIBRATION


def test_count_error_types() -> None:
    # The memory of previous runs is read on the error type alone, the column and the table being dropped
    details = [
        "insert_null | COLUMN_A | TABLE_A",
        "insert_null | COLUMN_B | TABLE_B",
        "duplicate_rows |  | TABLE_A",
    ]

    assert _count_error_types(details=details) == {"duplicate_rows": 1, "insert_null": 2}

    # No previous run, nothing to count
    assert _count_error_types(details=[]) == {}

    # Error types returned in alphabetical order, whatever the order they were logged in
    details = [
        "out_of_range | COLUMN_A | TABLE_A",
        "correlation_break | COLUMN_B | TABLE_A",
        "out_of_range | COLUMN_C | TABLE_B",
    ]

    assert list(_count_error_types(details=details)) == ["correlation_break", "out_of_range"]
