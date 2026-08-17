from agent.agent_api import ask_agent
from config import AGENT_LOG_DIR
from datetime import datetime, timezone
import json


prompt_agent = """You are a data quality analyst working on a database that has just received a new batch of
rows.

Before adding the rows, the data was profiled into a file called calibration, which maps all tables and
columns: number of rows, columns, datatype, distribution, correlations, primary and foreign keys, and also
clustering using machine learning techniques. 

The new rows were appended after the clean data. 

Investigate and report anything that looks anomalous compared with the calibration file. Small changes in data that 
could be qualified as acceptable or expected do not need to be reported. Only Medium to Critical severity need to be
investigated further and reported. Report numbers that you see in the new data rather than the whole dataset that 
also includes the clean data.

The output needs to be organised in a way that can be stored and run against a file to check the errors. 
Return a section as a TextBlock starting with ### **OUTPUT** and the following format:

### **OUTPUT**
Table name | Column Name | Metric | Calibrated | Current | Severity

Example:
### **OUTPUT**
Table A | Col A | NULLs | 0 | 10 | Critical
Table B | Col A & Col B | Correlation | 54% | 13% | Critical
Table C | Col C | Mean | 54 | 50 | Medium
"""


def _read_agent_logs() -> dict:
    with open(AGENT_LOG_DIR, "r") as f:
        d_logs = json.load(f)
    return d_logs


def _save_agent_logs(d_logs: dict) -> None:
    with open(AGENT_LOG_DIR, "w") as f:
        json.dump(d_logs, f, indent=4, default=str)
    return None


def clean_agent_logs(**params) -> None:
    # The file needs to be cleaned for each new run, and potentially created for all first usage
    # Saving an empty dict as an easy way to reset the file
    d_logs = {
        "datetime_created_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "investigation": [],
    }

    _save_agent_logs(d_logs=d_logs)
    return None


def _append_agent_log(run_number: int, **params) -> None:
    d_logs = _read_agent_logs()

    # Appending
    d_logs["investigation"].append({
        "run_number": run_number,
        "datetime_entered_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
    })

    _save_agent_logs(d_logs=d_logs)

    return None


def run_agent_investigation(run_number: int = 1):

    # Cleaning the logs for now - should be in the Automation loop calling this function
    clean_agent_logs()

    response = ask_agent(user_input=prompt_agent)

    # Printing the result of the investigation
    print("\n".join(block.text for block in response.content if block.type == "text"))

    # Saving each outcome of the investigation into an agent log file
    for block in response.content:
        if block.type == "text" and "**OUTPUT**" in block.text:
            agent_logs = block.text.split("**OUTPUT**")[1].strip().split("\n")
            for a_log in agent_logs:
                # format_output as per defined in the prompt to the agent.
                # This will need to be changed if the prompt format changes.
                format_output = [l.strip() for l in a_log.split("|")]
                dict_output = {
                    "Table": format_output[0],
                    "Column": format_output[1],
                    "Metric": format_output[2],
                    "Calibration": format_output[3],
                    "Current": format_output[4],
                    "Severity": format_output[5]
                }
                _append_agent_log(run_number=run_number, **dict_output)
    return None


if __name__ == "__main__":
    run_number = 1