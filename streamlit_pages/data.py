from typing import Any
import os
import json
from app.config import RECONCILIATION_LOG_DIR, AGENT_USAGE_DIR
from datetime import datetime

"""
    Data parsing to help visualising on the dashboard. The main source for the data is coming from the logs.
"""

def _read_json(path_log: str) -> dict[str, Any] | None:
    if os.path.exists(path_log):
        with open(path_log, "r") as f:
            return json.load(f)

    print(f"Cannot read this JSON {path_log}")
    return None


def get_all_reconciliation_logs(dataset_name: str) -> list[dict[str, Any]] | None:
    d_logs = _read_json(RECONCILIATION_LOG_DIR)

    if dataset_name not in d_logs["reconciliations"]:
        return None

    return [l for l in d_logs["reconciliations"][dataset_name]]


def get_last_reconciliation_log(dataset_name: str, time_created: datetime = None) -> dict[str, Any] | None:
    d_logs = _read_json(RECONCILIATION_LOG_DIR)

    if dataset_name not in d_logs["reconciliations"]:
        return None

    recs = d_logs["reconciliations"][dataset_name]
    latest_rec = max([d["datetime_created_utc"] for d in recs])

    # Filter on the datetime to make sure this is the latest created
    if time_created and latest_rec < time_created:
        return None

    return max(recs, key=lambda x: x["datetime_created_utc"])


def get_all_usage_for_dataset(dataset_name: str) -> list[dict[str, Any]] | None:

    d_logs = _read_json(AGENT_USAGE_DIR)

    if d_logs is None or "historical_usage" not in d_logs:
        return None

    return [l for l in d_logs["historical_usage"] if l["kaggle_dataset"] == dataset_name]


def get_latest_usage(dataset_name: str, time_created: datetime = None) -> dict[str, Any] | None:

    d_logs = get_all_usage_for_dataset(dataset_name=dataset_name)

    if d_logs is None or len(d_logs) == 0:
        return None

    latest_rec = max([d["datetime_created_utc"] for d in d_logs if d["kaggle_dataset"] == dataset_name])

    # Filter on the datetime to make sure this is the latest created
    if time_created and latest_rec < time_created:
        return None

    return max(d_logs, key=lambda x: x["datetime_created_utc"])


if __name__ == "__main__":
    dataset_name = "airbnb/seattle"

    get_latest_usage(dataset_name=dataset_name)
    get_last_reconciliation_log(dataset_name=dataset_name)
