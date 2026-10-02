from typing import Any
from app.reconciliation.reconcile_logs import _compare_lists
from app.reconciliation.utils import calc_score_agent


def _record(id_: int, table: str = "TABLE_A", column: str = "COLUMN_A",
            anomaly: str = "missing_values", nb_rows_affected: int = 10) -> dict[str, Any]:
    # Minimal record holding the keys _compare_lists relies on. Both the injection logs and the agent
    # diagnostics carry more keys, but only these are read when matching.
    return {
        "id": id_,
        "table": table,
        "column": column,
        "anomaly": anomaly,
        "nb_rows_affected": nb_rows_affected,
    }


def test_compare_lists_matches_on_required_keys() -> None:
    injected = [_record(id_=1)]
    diagnostics = [_record(id_=2)]

    # Same table, column and anomaly: the injected error is considered found by the agent
    assert _compare_lists(list1=injected, list2=diagnostics) == [
        {"id1": 1, "id2": 2, "nb_rows_affected": True}
    ]


def test_compare_lists_no_match_when_a_required_key_differs() -> None:
    injected = [_record(id_=1)]

    # Each diagnostic is right on 2 of the 3 required keys, which is not enough to be a match
    assert _compare_lists(list1=injected, list2=[_record(id_=2, table="TABLE_B")]) == []
    assert _compare_lists(list1=injected, list2=[_record(id_=2, column="COLUMN_B")]) == []
    assert _compare_lists(list1=injected, list2=[_record(id_=2, anomaly="duplicate_rows")]) == []


def test_compare_lists_ignores_column_when_one_side_is_empty() -> None:
    # Some errors are not tied to a column (duplicated rows for example), so an empty column on either
    # side is skipped and the match is decided on the table and the anomaly only
    injected = [_record(id_=1, column="", anomaly="duplicate_rows")]
    diagnostics = [_record(id_=1, column="COLUMN_A", anomaly="duplicate_rows")]

    assert _compare_lists(list1=injected, list2=diagnostics) == [
        {"id1": 1, "id2": 1, "nb_rows_affected": True}
    ]

    # Same when the agent is the one not reporting a column
    assert _compare_lists(list1=diagnostics, list2=injected) == [
        {"id1": 1, "id2": 1, "nb_rows_affected": True}
    ]


def test_compare_lists_flags_the_number_of_rows_affected() -> None:
    injected = [_record(id_=1, nb_rows_affected=10)]

    # The agent logs its findings as strings, the injection logs hold integers: both are cast before comparing
    assert _compare_lists(list1=injected, list2=[_record(id_=2, nb_rows_affected="10")]) == [
        {"id1": 1, "id2": 2, "nb_rows_affected": True}
    ]

    # A match on the anomaly still stands when the agent sized it wrong, only the bonus is lost
    assert _compare_lists(list1=injected, list2=[_record(id_=2, nb_rows_affected=11)]) == [
        {"id1": 1, "id2": 2, "nb_rows_affected": False}
    ]


def test_calc_score_agent() -> None:
    # Nothing injected: nothing to miss, full marks
    assert calc_score_agent(nb_errors_injected=0, nb_errors_found=0, nb_false_positive_high=0,
                            nb_false_positive_critical=0, correct_nb_rows_affected=0) == 1

    # 2 out of 4 found, one High (-0.05) and one Critical (-0.1) false positive, one row count right (+0.05)
    assert round(calc_score_agent(nb_errors_injected=4, nb_errors_found=2, nb_false_positive_high=1,
                                  nb_false_positive_critical=1, correct_nb_rows_affected=1), 4) == 0.4

    # Every error found, nothing else reported
    assert calc_score_agent(nb_errors_injected=3, nb_errors_found=3, nb_false_positive_high=0,
                            nb_false_positive_critical=0, correct_nb_rows_affected=0) == 1

    # Nothing found
    assert calc_score_agent(nb_errors_injected=5, nb_errors_found=0, nb_false_positive_high=0,
                            nb_false_positive_critical=0, correct_nb_rows_affected=0) == 0

    # A clean batch still loses points on false positives: 1 - 2 x 0.05
    assert round(calc_score_agent(nb_errors_injected=0, nb_errors_found=0, nb_false_positive_high=2,
                                  nb_false_positive_critical=0, correct_nb_rows_affected=0), 4) == 0.9

    # Every error found and sized right: the score goes above 1
    assert round(calc_score_agent(nb_errors_injected=2, nb_errors_found=2, nb_false_positive_high=0,
                                  nb_false_positive_critical=0, correct_nb_rows_affected=2), 4) == 1.1

