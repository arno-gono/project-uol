from typing import Any
from pathlib import Path
import pandas as pd
from app.logs_data import _read_json
from app.reconciliation.utils import calc_score_agent

"""
    Tables of the report appendix, rebuilt from the logs. Table H.1 and H.2 come from the first experiment (eight batches,
    archived), table H.3 from the second one (one batch, five models, archived).
"""

FIRST_EXPERIMENT_LOGS_DIR = Path(__file__).resolve().parent.parent / "archive_logs" / "20260925"
FIRST_EXPERIMENT_DATE = "2026-09-19"
SECOND_EXPERIMENT_LOGS_DIR = Path(__file__).resolve().parent.parent / "archive_logs" / "20260929"

DATASET_NAME = "airbnb/seattle"

# Names used in the report, in the order the models are displayed
MODEL_NAMES = {
    "claude-haiku-4-5": "Haiku 4.5",
    "claude-sonnet-5": "Sonnet 5",
    "claude-opus-5": "Opus 5",
    "claude-opus-5-5": "Opus 5.5",
    "claude-fable-5-1": "Fable 5.1"
}

STATISTICAL_ERROR_TYPES = ["correlation_break", "distribution_shift", "out_of_range"]


def _get_investigations(logs_dir: Path, date: str | None = None) -> list[dict[str, Any]]:
    # Reconciliation logs joined with usage.json on usage_id, so that each investigation carries its model and cost.
    recs = _read_json(logs_dir / "reconciliation_logs.json")["reconciliations"][DATASET_NAME]
    d_usage = {u["id"]: u for u in _read_json(logs_dir / "usage.json")["historical_usage"]}

    investigations = []
    for rec in recs:
        investigation = d_usage[rec["usage_id"]].copy()
        investigation.update(rec)
        investigations.append(investigation)

    # The archive also holds runs made after the experiment, filtered out on the day it was run
    if date is not None:
        investigations = [i for i in investigations if i["datetime_created_utc"][:10] == date]

    # Sorting by model, in the order of the report
    return sorted(investigations, key=lambda x: list(MODEL_NAMES).index(x["agent_model"]))


def _get_error_types(anomalies: list[str]) -> list[str]:
    # An anomaly is stored as "error type | column | table", the error type being the first criterion of the chain
    return [a.split(" | ")[0].strip() for a in anomalies]


def _count_false_positives_by_severity(incorrect_diagnostics: list[str], severity: str) -> int:
    # An incorrect diagnostic is stored as "error type | column | table | severity"
    return [d.split(" | ")[-1].strip().lower() for d in incorrect_diagnostics].count(severity.lower())


def _calc_score(investigation: dict[str, Any]) -> float:
    # Same score as the dashboard, recalculated from the reconciliation log
    incorrect_diagnostics = investigation["incorrect_diagnostics_made_by_agent"]

    return calc_score_agent(
        nb_errors_injected=investigation["total_anomalies"],
        nb_errors_found=investigation["total_anomalies_detected_by_agent"],
        nb_false_positive_high=_count_false_positives_by_severity(incorrect_diagnostics, severity="High"),
        nb_false_positive_critical=_count_false_positives_by_severity(incorrect_diagnostics, severity="Critical"),
        correct_nb_rows_affected=investigation["total_correct_nb_rows_affected"]
    )


def _format_ratio(nb: int, total: int, with_rate: bool = True) -> str:
    if not with_rate or total == 0:
        return f"{nb}/{total}"

    return f"{nb}/{total} ({nb / total:.0%})"


def get_table_first_experiment_by_model() -> pd.DataFrame:
    """
        Table H.1: eight batches investigated by each model. Detection is found over injected, precision is found over
        every diagnostic made by the agent. Last row for all the models together, the cost being the total spent.
    """

    investigations = _get_investigations(logs_dir=FIRST_EXPERIMENT_LOGS_DIR, date=FIRST_EXPERIMENT_DATE)

    rows = []
    for agent_model in [m for m in MODEL_NAMES if m in {i["agent_model"] for i in investigations}]:
        inv_model = [i for i in investigations if i["agent_model"] == agent_model]
        nb = len(inv_model)
        found = sum(i["total_anomalies_detected_by_agent"] for i in inv_model)

        rows.append({
            "Model": MODEL_NAMES[agent_model],
            "Runs": nb,
            "Detection": _format_ratio(found, sum(i["total_anomalies"] for i in inv_model)),
            "Precision": _format_ratio(found, sum(i["total_diagnostics_made_by_agent"] for i in inv_model)),
            "Correct row counts": sum(i["total_correct_nb_rows_affected"] for i in inv_model),
            "Mean score": f"{sum(_calc_score(i) for i in inv_model) / nb:.2f}",
            "Mean cost (USD)": f"{sum(i['total_cost_usd'] for i in inv_model) / nb:.2f}",
            "Mean duration (s)": f"{sum(i['investigation_time_seconds'] for i in inv_model) / nb:.0f}"
        })

    found = sum(i["total_anomalies_detected_by_agent"] for i in investigations)

    rows.append({
        "Model": "All models",
        "Runs": len(investigations),
        "Detection": _format_ratio(found, sum(i["total_anomalies"] for i in investigations)),
        "Precision": _format_ratio(found, sum(i["total_diagnostics_made_by_agent"] for i in investigations)),
        "Correct row counts": sum(i["total_correct_nb_rows_affected"] for i in investigations),
        "Mean score": "",
        "Mean cost (USD)": f"{sum(i['total_cost_usd'] for i in investigations):.2f} in total",
        "Mean duration (s)": ""
    })

    return pd.DataFrame(rows)


def get_table_first_experiment_by_error_type() -> pd.DataFrame:
    """
        Table H.2: errors found over errors injected, per error type and per model. The false positives are counted
        under the error type the agent reported. Error types sorted from the most missed to the most found.
    """

    investigations = _get_investigations(logs_dir=FIRST_EXPERIMENT_LOGS_DIR, date=FIRST_EXPERIMENT_DATE)
    all_models = [m for m in MODEL_NAMES if m in {i["agent_model"] for i in investigations}]

    # Counting per error type and per model: {error_type: {agent_model: [found, injected]}}
    d_counts = {}
    false_positives = []

    for i in investigations:
        found = _get_error_types(i["anomalies_detected_by_agent"])
        not_found = _get_error_types(i["anomalies_not_found_by_agent"])

        for error_type in set(found + not_found):
            count = d_counts.setdefault(error_type, {m: [0, 0] for m in all_models})[i["agent_model"]]
            count[0] += found.count(error_type)
            count[1] += found.count(error_type) + not_found.count(error_type)

        false_positives += _get_error_types(i["incorrect_diagnostics_made_by_agent"])

    rows = []
    for error_type, d_model in d_counts.items():
        found = sum(c[0] for c in d_model.values())
        injected = sum(c[1] for c in d_model.values())

        row = {"Error type": error_type}

        for agent_model in all_models:
            row[MODEL_NAMES[agent_model]] = _format_ratio(d_model[agent_model][0], d_model[agent_model][1],
                                                          with_rate=False)

        row["All models"] = _format_ratio(found, injected)
        row["False positives"] = false_positives.count(error_type)
        row["_rate"] = found / injected
        row["_injected"] = injected

        rows.append(row)

    # Lowest detection first. Equal rates: the most injected first, then alphabetical
    rows = sorted(rows, key=lambda x: (x["_rate"], -x["_injected"], x["Error type"]))

    found = sum(i["total_anomalies_detected_by_agent"] for i in investigations)

    row = {"Error type": "All types"}

    for agent_model in all_models:
        inv_model = [i for i in investigations if i["agent_model"] == agent_model]
        row[MODEL_NAMES[agent_model]] = _format_ratio(sum(i["total_anomalies_detected_by_agent"] for i in inv_model),
                                                      sum(i["total_anomalies"] for i in inv_model), with_rate=False)

    row["All models"] = _format_ratio(found, sum(i["total_anomalies"] for i in investigations))
    row["False positives"] = len(false_positives)

    rows.append(row)

    return pd.DataFrame(rows).drop(columns=["_rate", "_injected"])


def get_table_second_experiment() -> pd.DataFrame:
    """
        Table H.3: one batch investigated by each model with the machine learning layer passed, then withheld. The
        statistical errors are counted apart, being the ones the machine learning layer is meant to help with.
    """

    investigations = _get_investigations(logs_dir=SECOND_EXPERIMENT_LOGS_DIR)

    rows = []
    for i in investigations:
        found = _get_error_types(i["anomalies_detected_by_agent"])
        not_found = _get_error_types(i["anomalies_not_found_by_agent"])

        rows.append({
            "Model": MODEL_NAMES[i["agent_model"]],
            "ML layer": "passed" if i["ml_calibration_available"] else "withheld",
            "Errors found": _format_ratio(i["total_anomalies_detected_by_agent"], i["total_anomalies"],
                                          with_rate=False),
            "Statistical found": _format_ratio(len([e for e in found if e in STATISTICAL_ERROR_TYPES]),
                                               len([e for e in found + not_found if e in STATISTICAL_ERROR_TYPES]),
                                               with_rate=False),
            "False positives reported": len(i["incorrect_diagnostics_made_by_agent"]),
            "Correct row counts": i["total_correct_nb_rows_affected"],
            "Score": f"{_calc_score(i):.2f}",
            "Cost (USD)": f"{i['total_cost_usd']:.2f}",
            "Duration (s)": f"{i['investigation_time_seconds']:.0f}"
        })

    # The investigations keep the order they were run in for each model: passed, then withheld
    return pd.DataFrame(rows)


if __name__ == "__main__":

    df_h1 = get_table_first_experiment_by_model()
    df_h2 = get_table_first_experiment_by_error_type()
    df_h3 = get_table_second_experiment()
