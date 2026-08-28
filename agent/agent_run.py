from agent.agent_api import ask_agent
from typing import Any
from app.config import AGENT_LOG_DIR, AGENT_FEEDBACK_DIR
from datetime import datetime, timezone
from app.errors_injection.errors_injections_models import ERROR_TYPES_DICT
from app.errors_injection.injection_logs import find_latest_id
from agent.agent_cost_calc import calculate_costs
import json


# Importing the types of errors and their description so that the agent can point to a label
# when it identifies an anomaly.
error_types = "\n".join(f"- {error_name}: {details['description']}"
                        for error_name, details in ERROR_TYPES_DICT.items())

# Writing a System prompt in order to not add it to the conversation every round of the investigation.
# Doc: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#give-claude-a-role

system_prompt_agent = f"""You are a data quality analyst investigating a SQLite database.

### **THE SETUP**
 
The database was profiled while it was still clean. That profile is called the calibration and
it describes, for every table:
- total number of rows, number of duplicated rows
- for every column: datatype, if it accepts NULLs, the distribution of its values and number of unique values
- correlations between numeric columns, and associations between categorical ones
- columns that look like they could be primary or foreign keys
- a clustering of the rows obtained with machine learning techniques

A new batch of data arrived, and will be appended to the relevant table: 
they are currently kept separated from the main table. 
You will need to check if there is nothing anomalous in this data by comparing it with the data that was calibrated.

The new rows are in a table named after the calibrated one with a suffix "_new_data".

### **HOW TO INVESTIGATE**

Start with checking the tables available with the tool run_sql and the query "SELECT * FROM sqlite_master;".

Compare what a _new_data table holds against what the calibration says about its table.
A difference between the two is a candidate error.

The new data will most likely slightly drift from the original data: there is no need to report 
changes that are within limits to a reasonable tolerance. The batch might hold fewer rows than the calibrated 
table, so weigh a difference against the number of rows it is measured on before reporting it. 
Report only the anomalies you would rate as Medium, High or Critical.

The calibrated tables might contain a lot of data. Refer to the calibration file rather than selecting all rows
from these tables. You can occasionally query the calibrated tables during an investigation, but in this case 
use a COUNT, AVG, GROUP BY or LIMIT in your statement. 

### **OUTPUT**

End your investigation with a section starting with ### **OUTPUT**, one finding per line, in this
format:

Table name | Column Name | Anomaly | Calibrated | Current | Number of affected rows | Severity

Example of expected output:
### **OUTPUT**
Table A | Col A | insert_null | 0 | 10 | 14 | Critical
Table B | Col A & Col B | correlation_break | 54% | 13% | 30 | Critical
Table C | Col C | distribution_shift | 54 | 50 | 72 | High

- "Calibrated" and "Current" column: No need to have details. For example if the calibrated datatype is text and 
you flag numeric entries, enter "Text values" for Calibrated and "Numeric values" for Current. If this is for a 
correlation, enter "53.5" for Calibrated directly and "34.5" for Current if this is what you calculate. No need for
excessive details
- "Anomaly" column: it is the type of anomaly that has been detected. Apply exactly one of the names listed below, 
so they can be parsed. Only create a new name if none of them describes what you found.

{error_types}

### **FEEDBACK**

Not directly related with the results of the investigation. 
Optional section where you can give feedback about how to improve the model: for example new tools to help investigate,
system prompt needing more accurate details or better guidance, or any other suggestions to optimise this model.  

"""

# The system prompt holds the instructions, so the user message only has to start the run.
prompt_agent = "Investigate the database and report what you find."

def _read_agent_logs(log_dir: str) -> dict[str, Any]:
    with open(log_dir, "r") as f:
        d_logs = json.load(f)
    return d_logs


def _save_agent_logs(d_logs: dict[str, Any], log_dir: str) -> None:
    with open(log_dir, "w") as f:
        json.dump(d_logs, f, indent=4, default=str)
    return None


def clean_agent_logs(**params: Any) -> None:
    # The file needs to be cleaned for each new run, and potentially created for all first usage
    # Saving an empty dict as an easy way to reset the file
    d_logs = {
        "datetime_created_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "investigation": [],
    }

    _save_agent_logs(d_logs=d_logs, log_dir=AGENT_LOG_DIR)
    return None


def _append_feedback_log(feedback_log: list[str]) -> None:
    d_logs = _read_agent_logs(AGENT_FEEDBACK_DIR)

    # Adding an identifier to help reconciling with the injected errors
    id_number = find_latest_id(d_logs["feedback"])

    d_logs["feedback"].append(
        {
            "id": id_number,
            "datetime_created_utc": datetime.now(timezone.utc).isoformat(),
            "feedback": feedback_log
        }
    )

    _save_agent_logs(d_logs=d_logs, log_dir=AGENT_FEEDBACK_DIR)

    return None


def _append_agent_log(run_number: int, **params: Any) -> None:
    d_logs = _read_agent_logs(AGENT_LOG_DIR)

    # Adding an identifier to help reconciling with the injected errors
    id_number = find_latest_id(d_logs["investigation"])
    params["id"] = id_number

    # Removing the suffix _new_data so the table matches the calibrated one it was named after
    if "_new_data" in params["table"]:
        params["table"] = params["table"].replace("_new_data", "")

    # Appending
    d_logs["investigation"].append({
        "id": id_number,
        "run_number": run_number,
        "datetime_entered_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
    })

    _save_agent_logs(d_logs=d_logs, log_dir=AGENT_LOG_DIR)

    return None


def _parse_response_to_log(resp: list[Any], run_number: int = 1) -> None:

    # Saving each outcome of the investigation into an agent log file
    for block in resp:
        if block.type == "text" and "**OUTPUT**" in block.text:

            if "### **FEEDBACK**" in block.text:
                agent_logs, feedback_log = block.text.split("### **FEEDBACK**")
                agent_logs = agent_logs.split("**OUTPUT**")[1].strip().split("\n")
                feedback_log = [f.strip() for f in feedback_log.split("\n") if f != ""]
                _append_feedback_log(feedback_log=feedback_log)
            else:
                agent_logs = block.text.split("**OUTPUT**")[1].strip().split("\n")

            for a_log in agent_logs:
                # format_output as per defined in the prompt to the agent.
                # This will need to be changed if the prompt format changes.
                format_output = [l.strip() for l in a_log.split("|")]
                dict_output = {
                    "table": format_output[0],
                    "column": format_output[1],
                    "anomaly": format_output[2],
                    "calibration": format_output[3],
                    "current": format_output[4],
                    "nb_rows_affected": format_output[5],
                    "severity": format_output[6]
                }
                _append_agent_log(run_number=run_number, **dict_output)

    return None


def run_agent_investigation(run_number: int = 1) -> int:

    # Cleaning the logs for now - should be in the Automation loop calling this function
    clean_agent_logs()

    dict_result = ask_agent(user_input=prompt_agent, system_prompt=system_prompt_agent)

    response = dict_result["response"]

    # Printing the result of the investigation
    print("\n".join(block.text for block in response.content if block.type == "text"))

    # Calculating cost of the investigation and storing it
    cost_investigation, usage_id = calculate_costs(dict_result["usage"])
    print("Cost investigation: $", round(cost_investigation, 3))

    # Parsing the response from the API and saving to a log
    _parse_response_to_log(response.content, run_number)

    # The id of the usage entry is returned so the reconciliation can link a score to its cost
    return usage_id


if __name__ == "__main__":
    run_number = 1