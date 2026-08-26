from streamlit_pages.tab_investigation import (_parse_error_type_from_list, _calc_score_from_log,
                                              _count_false_positives_by_severity)


def test_parse_error_type():
    # The reconciliation logs store each anomaly as "error type | column | table", but the details
    # table only shows the error type
    anomalies = [
        "wrong_datatype | reviewer_name  | reviews",
        "duplicate_primary_key | longitude  | listings",
    ]

    assert _parse_error_type_from_list(error_type_list=anomalies) == [
        "wrong_datatype",
        "duplicate_primary_key",
    ]

    # Incorrect diagnostics carry a fourth element (the severity given by the agent), which must not
    # change what is extracted
    anomalies = ["orphan_foreign_key | listing_id | calendar | High"]
    assert _parse_error_type_from_list(error_type_list=anomalies) == ["orphan_foreign_key"]

    # An agent finding every injected error leaves anomalies_not_found_by_agent empty
    assert _parse_error_type_from_list(error_type_list=[]) == []

    # Nothing to split on: the value is returned trimmed rather than dropped
    assert _parse_error_type_from_list(error_type_list=[" insert_null "]) == ["insert_null"]



def test_count_false_positives_by_severity():
    # The severity is the last criterion of the chain, and the agent is free on the casing it returns
    incorrect_diagnostics = [
        "orphan_foreign_key | listing_id | reviews | High",
        "orphan_foreign_key | listing_id | calendar | Critical",
        "duplicate_rows | (all columns) | calendar | high",
        "wrong_datatype | price | calendar | Medium",
    ]

    assert _count_false_positives_by_severity(
        incorrect_diagnostics=incorrect_diagnostics, severity="High") == 2
    assert _count_false_positives_by_severity(
        incorrect_diagnostics=incorrect_diagnostics, severity="Critical") == 1


def test_calc_score_from_log_false_positives():
    # 1 anomaly out of 1 found, but 1 High (-0.05) and 1 Critical (-0.1) false positive on top
    d_rec = {
        "total_anomalies": 1,
        "total_anomalies_detected_by_agent": 1,
        "incorrect_diagnostics_made_by_agent": [
            "orphan_foreign_key | listing_id | reviews | High",
            "orphan_foreign_key | listing_id | calendar | Critical",
            "duplicate_rows | (all columns) | calendar | Medium",
        ],
    }

    assert _calc_score_from_log(d_rec=d_rec) == 0.85

    d_rec = {
        "total_anomalies": 8,
        "total_anomalies_detected_by_agent": 1,
        "incorrect_diagnostics_made_by_agent": []
    }

    assert _calc_score_from_log(d_rec=d_rec) == 0.125

