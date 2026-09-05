from typing import Any
from app.config import INJECTION_LOG_DIR, AGENT_LOG_DIR, RECONCILIATION_LOG_DIR, KAGGLE_DATASET_NAME
from agent.agent_run import _read_agent_logs, _save_agent_logs
from app.errors_injection.injection_logs import find_latest_id
from datetime import datetime, timezone
import json


def _compare_lists(list1: list[dict], list2: list[dict]) -> list[dict[str, Any]]:
    # Comparing if the elements (dictionaries) in list 1 are found in list 2. list1 and list2 have to have the same
    # structure, i.e. dictionaries with same keys to work.

    # To be considered a match, the dicts need to have the same values for the following keys:
    required_keys_to_match = ["table", "column", "anomaly"]

    # Bonus for finding the correct result for the following key(s):
    accurate_keys_to_match = ["nb_rows_affected"]

    all_matches = []
    for l1 in list1:
        for l2 in list2:

            # Checking if all the required keys have been found
            match_found = True

            for required_key in required_keys_to_match:
                # Exception: no column entered for some errors (duplicate rows for example).
                # This key can sometimes not be matched.
                if required_key == "column" and (l1["column"] == "" or l2["column"] == ""):
                    continue

                if required_key == "column":
                    # Handling cases where 2 columns are involved: they should be separated with a " & ". It is also
                    # the instruction passed on to the agent.
                    columns_1 = [col.strip() for col in l1["column"].split("&")]
                    columns_2 = [col.strip() for col in l2["column"].split("&")]

                    # When 2 columns have been logged:
                    if len(columns_1) == 2 and len(columns_2) == 2:
                        col_a_1, col_b_1 = columns_1
                        col_a_2, col_b_2 = columns_2

                        # The agent might log "col a & col b" whereas "col b & col a" was logged in the injection error
                        # file. Checking both scenarios.
                        same_order = col_a_1 == col_a_2 and col_b_1 == col_b_2
                        reversed_order = col_a_1 == col_b_2 and col_b_1 == col_a_2

                        if not same_order and not reversed_order:
                            match_found = False
                            break

                    # Single column case.
                    elif columns_1 != columns_2:
                        match_found = False
                        break

                    continue

                if l1[required_key] != l2[required_key]:
                    match_found = False
                    break

            # The injected error matches a diagnostic on 3 required keys. It is considered to have been found
            if match_found:
                # We loop through l1. We want to make sure that an element of l2 has not been already
                # attributed to an element in l1, as the matching need to be unique (no 2 diagnostics for 1 error
                # injected and vice versa)

                if len([i for i in all_matches if i["id2"] == l2["id"]]) == 0:
                    dict_rec = {
                        "id1": l1["id"],
                        "id2": l2["id"],
                    }
                else:
                    continue
            else:
                continue

            for bonus_keys in accurate_keys_to_match:
                # Casting all variables as strings (agent might save in different datatype)
                dict_rec[bonus_keys] = True if str(l1[bonus_keys]) == str(l2[bonus_keys]) else False

            all_matches.append(dict_rec)

    return all_matches


def _append_reconciliation_log(d_rec: dict[str, Any]) -> None:
    # Not reset between runs: this file is the score history of the agent, the same way usage.json keeps
    # the cost history. Scores are only comparable within a dataset, so they are kept in one list per dataset.
    if RECONCILIATION_LOG_DIR.exists():
        d_logs = _read_agent_logs(RECONCILIATION_LOG_DIR)
    else:
        d_logs = {
            "datetime_created_utc": datetime.now(timezone.utc).isoformat(),
            "reconciliations": {},
        }

    # First reconciliation ever run on that dataset
    if KAGGLE_DATASET_NAME not in d_logs["reconciliations"]:
        d_logs["reconciliations"][KAGGLE_DATASET_NAME] = []

    # Same identifier logic as the injection and agent logs, numbered within the dataset
    d_rec["id"] = find_latest_id(d_logs["reconciliations"][KAGGLE_DATASET_NAME])

    d_logs["reconciliations"][KAGGLE_DATASET_NAME].append(d_rec)

    _save_agent_logs(d_logs=d_logs, log_dir=RECONCILIATION_LOG_DIR)
    return None


def _filter_records_for_run(records: list[dict[str, Any]], run_number: int,
                            params_key: str) -> list[dict[str, Any]]:
    # The injection logs and the agent logs have the same dictionary shape under different keys.
    # Only the run being reconciled is kept.
    return [record[params_key] for record in records if record["run_number"] == run_number]


def _get_found_and_missed_ids(matches: list[dict[str, Any]],
                              injections: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    # A match holds the id of the injected error as "id1". Every injected error missing from that list is an
    # anomaly the agent did not find.
    ids_found = [err_found["id1"] for err_found in matches]
    ids_not_found = [err_inj["id"] for err_inj in injections if err_inj["id"] not in ids_found]

    return ids_found, ids_not_found


def _get_false_positive_ids(matches: list[dict[str, Any]], investigations: list[dict[str, Any]]) -> list[int]:
    # A match holds the id of the agent's diagnostic as "id2". A diagnostic matching no injected error is
    # a false positive.
    diagnostic_ids_found = [err_found["id2"] for err_found in matches]

    return [diagnostic["id"] for diagnostic in investigations if diagnostic["id"] not in diagnostic_ids_found]


def _describe_injected_anomalies(injections: list[dict[str, Any]], ids: list[int]) -> list[str]:
    # Building a memory that will be passed on to the agent. Storing error type | column name | table name
    return [f"{err_inj['error_type']} | "
            f"{err_inj['dict_rec']['column']} | "
            f"{err_inj['table_name']}"
            for err_inj in injections if err_inj["id"] in ids]


def _describe_incorrect_diagnostics(investigations: list[dict[str, Any]], ids: list[int]) -> list[str]:
    # Same format as the injected anomalies, with the severity the agent gave the diagnostic added to it
    return [f"{diagnostic['params']['anomaly']} | "
            f"{diagnostic['params']['column']} | "
            f"{diagnostic['params']['table']} | "
            f"{diagnostic['params']['severity']}"
            for diagnostic in investigations if diagnostic["id"] in ids]


def reconcile_agent_vs_injection_logs(run_number: int = 1, usage_id: int | None = None) -> dict[str, Any]:
    dict_agent = json.load(open(AGENT_LOG_DIR))
    dict_inj = json.load(open(INJECTION_LOG_DIR))

    all_inj_errors = _filter_records_for_run(records=dict_inj["injections"], run_number=run_number,
                                             params_key="dict_rec")

    all_agents_diagnostics = _filter_records_for_run(records=dict_agent["investigation"], run_number=run_number,
                                                     params_key="params")

    # To be checked: those should be the same, no need to run that twice. Think about edge cases
    injected_errors_found_by_agents = _compare_lists(list1=all_inj_errors, list2=all_agents_diagnostics)
    diagnostics_matching = _compare_lists(list1=all_agents_diagnostics, list2=all_inj_errors)

    nb_false_positive = len(all_agents_diagnostics) - len(diagnostics_matching)
    nb_errors_not_found = len(all_inj_errors) - len(injected_errors_found_by_agents)

    # Counting the number of time the agent was able to find out the number of rows affected by the impact.
    correct_nb_rows_affected = len(
        [err_found for err_found in injected_errors_found_by_agents if err_found["nb_rows_affected"]]
    )

    print(f"Matched errors: {len(injected_errors_found_by_agents)}, "
          f"False positives: {nb_false_positive}, "
          f"Errors not found: {nb_errors_not_found}, "
          f"Accurate number of rows affected: {correct_nb_rows_affected}")

    # Splitting the injected errors between the ones the agent found and the ones it missed
    ids_found, ids_not_found = _get_found_and_missed_ids(matches=injected_errors_found_by_agents,
                                                         injections=dict_inj["injections"])

    # The diagnostics the agent made that were tied to no injected error
    diagnostic_false_positive_ids = _get_false_positive_ids(matches=injected_errors_found_by_agents,
                                                            investigations=dict_agent["investigation"])

    # Listing the anomalies that were found by the agent, the ones it missed, and the false positive it diagnosed
    anomalies_found_by_agent = _describe_injected_anomalies(injections=dict_inj["injections"], ids=ids_found)
    anomalies_not_found_by_agent = _describe_injected_anomalies(injections=dict_inj["injections"], ids=ids_not_found)
    incorrect_diagnostics_made_by_agent = _describe_incorrect_diagnostics(
        investigations=dict_agent["investigation"], ids=diagnostic_false_positive_ids)

    # Aggregating all data into one dict
    d_rec = {
        "run_number": run_number,
        "usage_id": usage_id,
        "datetime_created_utc": datetime.now(timezone.utc).isoformat(),
        "total_anomalies": len(all_inj_errors),
        "total_anomalies_detected_by_agent": len(injected_errors_found_by_agents),
        "total_diagnostics_made_by_agent": len(all_agents_diagnostics),
        "total_correct_nb_rows_affected": correct_nb_rows_affected,
        "anomalies_detected_by_agent": anomalies_found_by_agent,
        "anomalies_not_found_by_agent": anomalies_not_found_by_agent,
        "incorrect_diagnostics_made_by_agent": incorrect_diagnostics_made_by_agent,
    }

    _append_reconciliation_log(d_rec=d_rec)

    return d_rec


if __name__ == "__main__":
    run_number = 1
