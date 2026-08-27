import json
from datetime import datetime, timezone
from app.config import (LOGS_DIR, INJECTION_LOG_DIR, AGENT_LOG_DIR, AGENT_FEEDBACK_DIR, AGENT_USAGE_DIR,
                        RECONCILIATION_LOG_DIR)

"""
    The logs folder is not versioned, so it is missing on a fresh clone of the repository. 
    Every log file is read before being written to, ensuring the script does not break later when trying to 
    write in a non-existent file.
"""


def init_logs() -> None:

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    datetime_created_utc = datetime.now(timezone.utc).isoformat()

    # Structure expected by the functions reading each file.
    empty_logs = {
        INJECTION_LOG_DIR: {
            "datetime_created_utc": datetime_created_utc,
            "params": {},
            "injections": [],
            "failed_injections": [],
        },
        AGENT_LOG_DIR: {
            "datetime_created_utc": datetime_created_utc,
            "params": {},
            "investigation": [],
        },
        AGENT_FEEDBACK_DIR: {
            "feedback": [],
        },
        AGENT_USAGE_DIR: {
            "historical_usage": [],
        },
        RECONCILIATION_LOG_DIR: {
            "datetime_created_utc": datetime_created_utc,
            "reconciliations": {},
        },
    }

    for log_dir, d_empty_log in empty_logs.items():

        # Only the missing files are created: an existing log is a history that must not be reset
        if log_dir.exists():
            continue

        with open(log_dir, "w") as f:
            json.dump(d_empty_log, f, indent=4, default=str)

        print(f"Log file created: {log_dir}")

    return None


if __name__ == "__main__":
    init_logs()
