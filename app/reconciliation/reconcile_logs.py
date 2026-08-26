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


def reconcile_agent_vs_injection_logs(run_number: int = 1) -> dict[str, Any]:
    dict_agent = json.load(open(AGENT_LOG_DIR))
    dict_inj = json.load(open(INJECTION_LOG_DIR))

    all_inj_errors = [inj_errors["dict_rec"]
                      for inj_errors in dict_inj["injections"] if inj_errors["run_number"] == run_number]

    all_agents_diagnostics = [investigation["params"]
                              for investigation in dict_agent["investigation"] if investigation["run_number"] == run_number]

    # To be checked: those should be the same, no need to run that twice. Think about edge cases
    injected_errors_found_by_agents = _compare_lists(list1=all_inj_errors, list2=all_agents_diagnostics)
    diagnostics_matching = _compare_lists(list1=all_agents_diagnostics, list2=all_inj_errors)

    nb_false_positive = len(all_agents_diagnostics) - len(diagnostics_matching)
    nb_errors_not_found = len(all_inj_errors) - len(injected_errors_found_by_agents)

    print(f"Matched errors: {len(injected_errors_found_by_agents)}, "
          f"False positives: {nb_false_positive}, "
          f"Errors not found: {nb_errors_not_found}")

    # Building a memory that will be passed on to the agent. Storing error type | column name | table name
    ids_found = [err_found["id1"] for err_found in injected_errors_found_by_agents]
    ids_not_found = [err_inj["id"] for err_inj in dict_inj["injections"] if err_inj["id"] not in ids_found]

    diagnostic_id_found = [err_found["id2"] for err_found in injected_errors_found_by_agents]
    diagnostic_false_positive_id = [diagnostic["id"]
                                    for diagnostic in dict_agent["investigation"]
                                    if diagnostic["id"] not in diagnostic_id_found]

    # Listing the anomalies that were found by the agent
    anomalies_found_by_agent = []

    for err_found in injected_errors_found_by_agents:
        for err_inj in dict_inj["injections"]:
            if err_found["id1"] == err_inj["id"]:
                anomalies_found_by_agent.append(
                    f"{err_inj['error_type']} | "
                    f"{err_inj['dict_rec']['column']}  | "
                    f"{err_inj['table_name']}")

    # Listing the anomalies that were not found by the agent
    anomalies_not_found_by_agent = []
    for id_not_found in ids_not_found:
        for err_inj in dict_inj["injections"]:
            if err_inj["id"] == id_not_found:
                anomalies_not_found_by_agent.append(
                    f"{err_inj['error_type']} | "
                    f"{err_inj['dict_rec']['column']}  | "
                    f"{err_inj['table_name']}")

    # Listing the false positive the agent diagnosed but were wrong
    incorrect_diagnostics_made_by_agent = []
    for false_positive in diagnostic_false_positive_id:
        for diagnostic in dict_agent["investigation"]:
            if false_positive == diagnostic["id"]:
                incorrect_diagnostics_made_by_agent.append(
                    f"{diagnostic['params']['anomaly']} | "
                    f"{diagnostic['params']['column']} | "
                    f"{diagnostic['params']['table']} | "
                    f"{diagnostic['params']['severity']}")

    # Aggregating all data into one dict
    d_rec = {
        "run_number": run_number,
        "datetime_created_utc": datetime.now(timezone.utc).isoformat(),
        "total_anomalies": len(all_inj_errors),
        "total_anomalies_detected_by_agent": len(injected_errors_found_by_agents),
        "total_diagnostics_made_by_agent": len(all_agents_diagnostics),
        "anomalies_detected_by_agent": anomalies_found_by_agent,
        "anomalies_not_found_by_agent": anomalies_not_found_by_agent,
        "incorrect_diagnostics_made_by_agent": incorrect_diagnostics_made_by_agent,
    }

    _append_reconciliation_log(d_rec=d_rec)

    return d_rec


if __name__ == "__main__":
    run_number = 1
