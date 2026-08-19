from agent.agent_api import ask_agent
from config import AGENT_LOG_DIR
from datetime import datetime, timezone
import json


# Writing a System prompt in order to not add it to the conversation every round of the investigation.
# Doc: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#give-claude-a-role

system_prompt_agent = """You are a data quality analyst investigating a SQLite database.

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
changes that are within limits to a reasonable tolerance. 
Report only the anomalies you would rate as Medium, High or Critical.

The calibrated tables might contain a lot of data. Refer to the calibration file rather than selecting all rows
from these tables. You can occasionally query the calibrated tables during an investigation, but in this case 
use a COUNT, AVG, GROUP BY or LIMIT in your statement. 

### **OUTPUT**

End your investigation with a section starting with ### **OUTPUT**, one finding per line, in this
format:

Table name | Column Name | Metric | Calibrated | Current | Number of affected rows | Severity

Example:
### **OUTPUT**
Table A | Col A | NULLs | 0 | 10 | 14 | Critical
Table B | Col A & Col B | Correlation | 54% | 13% | 30 | Critical
Table C | Col C | Mean | 54 | 50 | 72 | High

### **FEEDBACK**

Not directly related with the results of the investigation. 
Optional section where you can give feedback about how to improve the model: for example new tools to help investigate,
system prompt needing more accurate details or better guidance, or any other suggestions to optimise this model.  

"""

# The system prompt holds the instructions, so the user message only has to start the run.
prompt_agent = "Investigate the database and report what you find."


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

    response = ask_agent(user_input=prompt_agent, system_prompt=system_prompt_agent)

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