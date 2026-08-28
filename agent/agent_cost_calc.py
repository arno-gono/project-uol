from app.config import AGENT_MODEL, AGENT_USAGE_DIR, KAGGLE_DATASET_NAME, KAGGLE_TABLE_MAX_ROWS, AGENT_MAX_ROWS_RETURNED
import json
from datetime import datetime, timezone

# API costs per MTok, in USD
# Doc: https://platform.claude.com/docs/en/build-with-claude/prompt-caching

dict_costs_api = {
    "claude-haiku-4-5": {
        "total_input_tokens": 1,
        "total_output_tokens": 5,
        "total_cache_read": 0.1,
        "total_cache_written": 1.25
    },
    "claude-sonnet-5": {
        "total_input_tokens": 2,
        "total_output_tokens": 10,
        "total_cache_read": 0.2,
        "total_cache_written": 2.5
    },
    "claude-opus-5": {
        "total_input_tokens": 5,
        "total_output_tokens": 25,
        "total_cache_read": 0.5,
        "total_cache_written": 6.25
    },
}


def _read_usage_log() -> dict:
    with open(AGENT_USAGE_DIR, "r") as f:
        d_logs = json.load(f)
    return d_logs


def _add_usage_to_log(**params) -> int:
    d_logs = _read_usage_log()

    # Attributing an identifier
    params["id"] = 1 if not d_logs["historical_usage"] \
        else max([usage["id"] for usage in d_logs["historical_usage"]]) + 1

    # Recording time of creation
    params["datetime_created_utc"] = datetime.now(timezone.utc).isoformat()

    # Adding latest usage
    d_logs["historical_usage"].append(params)

    # Appending to logs
    with open(AGENT_USAGE_DIR, "w") as f:
        json.dump(d_logs, f, indent=4, default=str)

    # Returned so the investigation can be linked to its cost, in the reconciliation log
    return params["id"]


def calculate_costs(dict_costs: dict) -> tuple[float, int]:
    """
        Expect a dict with the following keys:
            total_input_tokens
            total_output_tokens
            total_cache_read
            total_cache_written
        Calculates the cost in USD given a number of input tokens and output tokens.
        Returns the cost and the id given to the usage entry it just logged.
    """

    model_costs = dict_costs_api[AGENT_MODEL]

    # Nb tokens multiplied by the cost for 1 million tokens
    cost_input = dict_costs["total_input_tokens"] * model_costs["total_input_tokens"]
    cost_output = dict_costs["total_output_tokens"] * model_costs["total_output_tokens"]
    cost_cache_read = dict_costs["total_cache_read"] * model_costs["total_cache_read"]
    cost_cache_write = dict_costs["total_cache_written"] * model_costs["total_cache_written"]

    # Converting to USD from million
    total_cost = (cost_input + cost_output + cost_cache_read + cost_cache_write) / 1_000_000

    # Creating a dict with all relevant information and storing it
    dict_costs["total_cost_usd"] = round(total_cost, 4)
    dict_costs["kaggle_dataset"] = KAGGLE_DATASET_NAME
    dict_costs["kaggle_table_max_rows"] = KAGGLE_TABLE_MAX_ROWS
    dict_costs["agent_max_sql_rows_read"] = AGENT_MAX_ROWS_RETURNED
    dict_costs["agent_model"] = AGENT_MODEL
    dict_costs["notes"] = ""

    usage_id = _add_usage_to_log(**dict_costs)

    return total_cost, usage_id
