import json
from datetime import datetime, timezone
from config import INJECTION_LOG_DIR

"""
Keeping logs of all errors injected, including various parameters (nb of duplicated rows added etc) to test 
how accurate can the agent be if it finds an error. Reading / adding into a JSON file (easier to parse than a text file)
"""

def _save_injection_logs(d_logs: dict) -> None:
    with open(INJECTION_LOG_DIR, "w") as f:
        json.dump(d_logs, f, indent=4, default=str)
    return None


def clean_injection_logs(**params) -> None:
    # The file needs to be cleaned for each new run, and potentially created for all first usage
    # Saving an empty dict as an easy way to reset the file
    d_logs = {
        "datetime_created_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "injections": [],
    }

    _save_injection_logs(d_logs=d_logs)
    return None


def _read_injection_logs() -> dict:
    with open(INJECTION_LOG_DIR, "r") as f:
        d_logs = json.load(f)
    return d_logs


def append_injection_logs(table_name: str, error_type: str, run_number: int, **params) -> None:
    d_logs = _read_injection_logs()

    # Appending
    d_logs["injections"].append({
        "run_number": run_number,
        "table_name": table_name,
        "error_type": error_type,
        "datetime_entered_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
    })

    _save_injection_logs(d_logs=d_logs)
    return None

if __name__ == "__main__":
    if '1' == '2':
        append_injection_logs(table_name='test_table', error_type='error', run_number=2, hi="prout", threshold=0.22)
        clean_injection_logs(test="test", youpu="foff")

        q = _read_injection_logs()
