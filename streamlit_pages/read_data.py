from typing import Any
import os
import json
from config import AGENT_LOG_DIR, RECONCILIATION_LOG_DIR, INJECTION_LOG_DIR, AGENT_USAGE_DIR

"""
    Data parsing to help visualising on the dashboard. The main source for the data is coming from the logs.
"""

def _read_json(path_log: str) -> dict[str, Any] | None:
    if os.path.exists(path_log):
        with open(path_log, "r") as f:
            return json.load(f)

    print(f"Cannot read this JSON {path_log}")
    return None



