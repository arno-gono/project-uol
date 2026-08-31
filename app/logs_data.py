from typing import Any
import os
import json
import pandas as pd
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


def get_all_reconciliation_logs(dataset_name: str, agent_model: str | None = None) -> list[dict[str, Any]] | None:
    d_logs = _read_json(RECONCILIATION_LOG_DIR)

    if dataset_name not in d_logs["reconciliations"]:
        return None

    all_recs = [l for l in d_logs["reconciliations"][dataset_name]]

    # Returning all logs if agent is omitted from the call.
    if agent_model is None:
        return all_recs

    # The model used is in usage.json. Getting usage data and join on the usage_id key.
    d_usage = get_all_usage_for_dataset(dataset_name=dataset_name)

    if d_usage is None:
        return None

    # Getting all ids for that agent model
    usage_ids_for_model = [usage["id"] for usage in d_usage if usage["agent_model"] == agent_model]

    # Returning all the corresponding keys in the reconciliations logs
    return [rec for rec in all_recs if rec["usage_id"] in usage_ids_for_model]


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


def get_investigations_for_dataset(dataset_name: str) -> pd.DataFrame:

    df_recs = pd.DataFrame(get_all_reconciliation_logs(dataset_name=dataset_name))
    df_usage = pd.DataFrame(get_all_usage_for_dataset(dataset_name=dataset_name))

    # No investigation reconciled for that dataset, there is nothing to join and nothing to trim
    if df_recs.empty:
        return df_recs

    # usage.json and reconciliation_logs.json are joined by the key usage_id (reconciliation_logs.json) / id (usage.json).
    # This is to know how much an investigation cost and on what model it was run on.
    if not df_usage.empty:
        df_recs = df_recs.merge(
            df_usage,
            how="left",
            left_on="usage_id",
            right_on="id",
            suffixes=("", "_usage")
        )

    # Trimming to the columns for what is displayed on the dashboard.
    return df_recs[[
        "id",
        "usage_id",
        "datetime_created_utc",
        "total_anomalies",
        "total_anomalies_detected_by_agent",
        "total_diagnostics_made_by_agent",
        "total_correct_nb_rows_affected",
        "anomalies_detected_by_agent",
        "anomalies_not_found_by_agent",
        "incorrect_diagnostics_made_by_agent",
        "agent_model",
        "total_cost_usd"
    ]]


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
    agent_model = "claude-haiku-4-5"

    # get_latest_usage(dataset_name=dataset_name)
    # get_last_reconciliation_log(dataset_name=dataset_name)
