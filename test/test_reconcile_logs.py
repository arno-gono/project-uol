from reconciliation.reconcile_logs import _compare_lists


def _record(id_: int, table: str = "TABLE_A", column: str = "COLUMN_A",
            anomaly: str = "missing_values", nb_rows_affected: int = 10) -> dict:
    # Minimal record holding the keys _compare_lists relies on. Both the injection logs and the agent
    # diagnostics carry more keys, but only these are read when matching.
    return {
        "id": id_,
        "table": table,
        "column": column,
        "anomaly": anomaly,
        "nb_rows_affected": nb_rows_affected,
    }


def test_compare_lists_matches_on_required_keys():
    injected = [_record(1)]
    diagnostics = [_record(2)]

    # Same table, column and anomaly: the injected error is considered found by the agent
    assert _compare_lists(list1=injected, list2=diagnostics) == [
        {"id1": 1, "id2": 2, "nb_rows_affected": True}
    ]


def test_compare_lists_no_match_when_a_required_key_differs():
    injected = [_record(1)]

    # Each diagnostic is right on 2 of the 3 required keys, which is not enough to be a match
    assert _compare_lists(list1=injected, list2=[_record(2, table="TABLE_B")]) == []
    assert _compare_lists(list1=injected, list2=[_record(2, column="COLUMN_B")]) == []
    assert _compare_lists(list1=injected, list2=[_record(2, anomaly="duplicate_rows")]) == []


def test_compare_lists_ignores_column_when_one_side_is_empty():
    # Some errors are not tied to a column (duplicated rows for example), so an empty column on either
    # side is skipped and the match is decided on the table and the anomaly only
    injected = [_record(1, column="", anomaly="duplicate_rows")]
    diagnostics = [_record(1, column="COLUMN_A", anomaly="duplicate_rows")]

    assert _compare_lists(list1=injected, list2=diagnostics) == [
        {"id1": 1, "id2": 1, "nb_rows_affected": True}
    ]

    # Same when the agent is the one not reporting a column
    assert _compare_lists(list1=diagnostics, list2=injected) == [
        {"id1": 1, "id2": 1, "nb_rows_affected": True}
    ]

