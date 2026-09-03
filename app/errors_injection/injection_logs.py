import json
from datetime import datetime, timezone
from typing import Any
from app.config import INJECTION_LOG_DIR

"""
Keeping logs of all errors injected, including various parameters (nb of duplicated rows added etc) to test 
how accurate can the agent be if it finds an error. Reading / adding into a JSON file (easier to parse than a text file)
"""

def _save_injection_logs(d_logs: dict[str, Any]) -> None:
    with open(INJECTION_LOG_DIR, "w") as f:
        json.dump(d_logs, f, indent=4, default=str)
    return None


def clean_injection_logs(**params: Any) -> None:
    # The file needs to be cleaned for each new run, and potentially created for all first usage
    # Saving an empty dict as an easy way to reset the file
    d_logs = {
        "datetime_created_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "injections": [],
        "failed_injections": [],
    }

    _save_injection_logs(d_logs=d_logs)
    return None


def _read_injection_logs() -> dict[str, Any]:
    with open(INJECTION_LOG_DIR, "r") as f:
        d_logs = json.load(f)
    return d_logs


def find_latest_id(list_inj_diag: list[dict[str, Any]]) -> int:
    # Function shared with the id created by the agent
    if not list_inj_diag:
        return 1

    id_num = 1

    for i in list_inj_diag:
        if i["id"] > id_num:
            id_num = i["id"]
    return id_num + 1


def append_injection_logs(table_name: str, error_type: str, run_number: int, dict_rec: dict[str, Any],
                          **params: Any) -> None:
    d_logs = _read_injection_logs()

    # Adding an identifier to the error, to facilitate the reconciliation with the agent's findings
    id_number = find_latest_id(d_logs["injections"])
    dict_rec["id"] = id_number

    # Appending
    d_logs["injections"].append({
        "id": id_number,
        "run_number": run_number,
        "table_name": table_name,
        "error_type": error_type,
        "datetime_entered_utc": datetime.now(timezone.utc).isoformat(),
        "dict_rec": dict_rec,
        "params": params,
    })

    _save_injection_logs(d_logs=d_logs)
    return None


def append_failed_injection_logs(table_name: str, error_type: str, error_message: str) -> None:
    # Kept apart from the injections: nothing was written in the table, so the agent cannot be asked to find it.
    d_logs = _read_injection_logs()

    # A log file written before this list existed does not hold the key yet
    if "failed_injections" not in d_logs:
        d_logs["failed_injections"] = []

    d_logs["failed_injections"].append({
        "table_name": table_name,
        "error_type": error_type,
        "datetime_entered_utc": datetime.now(timezone.utc).isoformat(),
        "error_message": error_message,
    })

    _save_injection_logs(d_logs=d_logs)
    return None


if __name__ == "__main__":
    # append_injection_logs(table_name='test_table', error_type='error', run_number=2, hi="test", threshold=0.22)
    d_logs = _read_injection_logs()
