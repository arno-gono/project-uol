from agent.agent_api import ask_agent
from typing import Any
from app.config import AGENT_LOG_DIR, AGENT_FEEDBACK_DIR, AGENT_MODEL
from datetime import datetime, timezone
from app.errors_injection.errors_injections_models import ERROR_TYPES_DICT
from app.errors_injection.injection_logs import find_latest_id
from agent.agent_cost_calc import calculate_costs
from app.logs_data import get_all_reconciliation_logs
import json


# Importing the types of errors and their description so that the agent can point to a label
# when it identifies an anomaly.
error_types = "\n\t".join(f"- {error_name}: {details['description']}"
                        for error_name, details in ERROR_TYPES_DICT.items())

# Writing a System prompt in order to not add it to the conversation every round of the investigation.
# Doc: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#give-claude-a-role

def _count_error_types(details: list) -> dict[str, int]:
    # The details are logged as "error type | column | table", the column and the table being dropped here. Simplifying
    # task for the agent and avoid overfit: returning the error type only rather than the error type + column and table.
    error_types = [detail.split(" | ")[0].strip() for detail in details]

    return {error_type: error_types.count(error_type) for error_type in sorted(set(error_types))}


def _previous_runs_section(kaggle_dataset: str, agent_model: str = AGENT_MODEL) -> str:

    # Reformatting previous reconciliation logs into a table showing percentage of success on the previous runs.

    # Getting previous investigations results (reconciliations) for the selected database and model used.
    prev_recs = get_all_reconciliation_logs(dataset_name=kaggle_dataset, agent_model=agent_model)

    if prev_recs is None:
        return ""

    # Calculating the percentage of errors found by the agent.
    nb_found = {error_type: 0 for error_type in ERROR_TYPES_DICT}
    nb_introduced = {error_type: 0 for error_type in ERROR_TYPES_DICT}

    for rec in prev_recs:
        # Incrementing the count of errors found by the agent
        for error_type, count in _count_error_types(details=rec["anomalies_detected_by_agent"]).items():
            nb_found[error_type] += count
            nb_introduced[error_type] += count

        # Incrementing the count of errors not found by the agent
        for error_type, count in _count_error_types(details=rec["anomalies_not_found_by_agent"]).items():
            nb_introduced[error_type] += count

    # Scoring every type of error: found / (found + introduced)
    scores = []

    for error_type in ERROR_TYPES_DICT:
        found = nb_found[error_type]
        introduced = nb_introduced[error_type]

        rate = found / introduced if introduced != 0 else 1

        scores.append({"error_type": error_type, "found": found, "introduced": introduced, "rate": rate})

    # Converting scores into a prompt. Starting with the least successful first so it is coming on top.
    prompt_recs = ""

    for score in sorted(scores, key=lambda x: x["rate"]):

        if score["introduced"] == 0:
            temp_score_prompt = f"\n\t- {score['error_type']}: never introduced in a previous run"
        else:
            temp_score_prompt = (f"\n\t- {score['error_type']}: found {score['found']} out of the "
                                 f"{score['introduced']} times it was introduced ({score['rate']:.0%})")

        # Adding the score to the main prompt.
        prompt_recs += temp_score_prompt

    prompt = f"""
    ### **PREVIOUS RUNS**

    Below is how often you found each type of anomaly when it had actually been introduced, measured over previous
    runs on this database and crosschecked against the anomalies that had been introduced. A check that keeps missing
    is worth repeating.

    It should be read as a weakness in your method, not as a guiding rule on how to investigate: the anomalies come
    randomly for every run and not all errors will be present. It says how well you checked, not what errors is to be
    found.

    Investigate every type of anomaly on every table you can apply to, whatever the rate below says. Where the rate is
    low, the checks performed in the past runs were not sufficient. Base your analysis on evidence and measurement.

    {prompt_recs}
    """

    return prompt


def _get_system_prompt(kaggle_dataset: str, agent_model: str = AGENT_MODEL) -> str:

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

    Keys are the exception: a primary or a foreign key belongs to the table as a whole, so check it against the
    calibrated table and its _new_data batch together. A row might point at a parent from an earlier batch and is not 
    an orphan in that case. A unique key in the new batch is still a duplicate if the calibrated table already uses it.

    The new data will most likely slightly drift from the original data: there is no need to report 
    changes that are within limits to a reasonable tolerance. The batch might hold fewer rows than the calibrated 
    table, so weigh a difference against the number of rows it is measured on before reporting it. 
    Report only the anomalies you would rate as Medium, High or Critical.
    
    The calibrated tables might contain a lot of data. Refer to the calibration file rather than selecting all rows
    from these tables. You can occasionally query the calibrated tables during an investigation, but in this case 
    use a COUNT, AVG, GROUP BY or LIMIT in your statement.

    A measurement that has already been made does not need to be made again: rerunning a count that is settled
    spends a round that another column still needs.

    ### **HOW TO MEASURE**

    The calibration already holds the correlations and the distribution of every column, so read the calibrated
    figure from it rather than recomputing it. Only the _new_data batch has to be measured. SQLite has no CORR,
    STDDEV or VARIANCE, but sqrt() and pow() do exist, so a standard deviation and a correlation can both be
    worked out from SUM, AVG and COUNT in a single query.

    A distribution_shift is found by measuring the mean and the standard deviation of the batch and comparing both
    against the calibrated ones, column by column. Same for correlation_break.

    typeof() and the declared schema cannot be used to find a wrong_datatype: they always agree with the
    calibration. Look at the shape of the values instead: a column calibrated as text holding entries that read
    as numbers, or a column calibrated as a number holding entries that do not, is a wrong_datatype.

    A repeated key does not say on its own which anomaly it is. Check whether the whole row is duplicated or not
    before calling it a duplicate_primary_key.

    A correlation_break has its mean and spread relatively close to the calibrated ones. Only the correlation with
    an other column starts drifting. If the mean or the spread drift: it should be a distribution_shift, reported on 
    the column where the drift is witnessed.

    ### **OUTPUT**
    
    End your investigation with a section starting with ### **OUTPUT**, one finding per line, no line breaks, strictly 
    in this format:
    
    Table name | Column Name | Anomaly | Calibrated | Current | Number of affected rows | Severity
    
    Every line holds exactly 7 fields separated by 6 pipes, in the order above. A field you cannot fill is left
    empty between its two pipes: never drop it and never merge two of them into one. A line is parsed on its
    number of fields and is discarded if it does not hold 7, so a finding written on 6 fields is a finding lost.
    
    Example of expected output:
    ### **OUTPUT**
    Table A | Col A | insert_null | 0 | 10 | 14 | Critical
    Table B | Col A & Col B | correlation_break | 54% | 13% | 30 | Critical
    Table C | Col C | distribution_shift | 54 | 50 | 72 | High
    Table D |  | duplicate_rows | 0 duplicated rows | 145 duplicated rows | 145 | Critical
    
    - "Column Name" column: when an anomaly is measured on two columns, name both of them separated by "&",
    as in the example above.
    - "Calibrated" and "Current" column: No need to have details. For example if the calibrated datatype is text and 
    you flag numeric entries, enter "Text values" for Calibrated and "Numeric values" for Current. If this is for a 
    correlation, enter "53.5" for Calibrated directly and "34.5" for Current if this is what you calculate. No need for
    excessive details
    - "Anomaly" column: it is the type of anomaly that has been detected. Apply exactly one of the names listed below, 
    so they can be parsed. Only create a new name if none of them describes what you found.
    
    {error_types}

    {_previous_runs_section(kaggle_dataset=kaggle_dataset, agent_model=agent_model)}    
    ### **FEEDBACK**
    
    Not directly related with the results of the investigation. 
    Optional section where you can give feedback about how to improve the model: for example new tools to help investigate,
    system prompt needing more accurate details or better guidance, or any other suggestions to optimise this model.  
    """
    return system_prompt_agent


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

            # The agent sometimes quotes **OUTPUT** several times, as it keeps thinking whilst writing.
            # Reading the last block from the response.
            if "### **FEEDBACK**" in block.text:
                agent_logs, feedback_log = block.text.split("### **FEEDBACK**")
                agent_logs = agent_logs.split("**OUTPUT**")[-1].strip().split("\n")
                feedback_log = [f.strip() for f in feedback_log.split("\n") if f != ""]
                _append_feedback_log(feedback_log=feedback_log)
            else:
                agent_logs = block.text.split("**OUTPUT**")[-1].strip().split("\n")

            for a_log in agent_logs:
                if a_log.strip() == "":
                    continue

                # format_output as per defined in the prompt to the agent.
                # This will need to be changed if the prompt format changes.
                format_output = [l.strip() for l in a_log.split("|")]

                # The agent occasionally returns a wrong OUTPUT line for its findings. Skipping the row rather
                # than crashing the whole loop. 
                if len(format_output) != 7:
                    print(f"\t\033[91mline ignored, not in the expected format: {a_log}\033[0m")
                    continue

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


def run_agent_investigation(kaggle_dataset: str, agent_model: str = AGENT_MODEL, run_number: int = 1) -> int:

    # Cleaning the logs for now - should be in the Automation loop calling this function
    clean_agent_logs()

    # Recording the time taken for the agent to run its investigation.
    start_time = datetime.now(timezone.utc)

    dict_result = ask_agent(
        user_input=prompt_agent,
        system_prompt=_get_system_prompt(kaggle_dataset=kaggle_dataset, agent_model=agent_model),
        agent_model=agent_model
    )

    # End of the investigation.
    end_time = datetime.now(timezone.utc)
    dict_result["usage"]["investigation_time_seconds"] = round((end_time - start_time).total_seconds(), 2)

    response = dict_result["response"]

    # Printing the result of the investigation
    print("\n".join(block.text for block in response.content if block.type == "text"))

    # Calculating cost of the investigation and storing it
    cost_investigation, usage_id = calculate_costs(dict_result["usage"], agent_model=agent_model)
    print("Cost investigation: $", round(cost_investigation, 3))

    # Parsing the response from the API and saving to a log
    _parse_response_to_log(response.content, run_number)

    # The id of the usage entry is returned so the reconciliation can link a score to its cost
    return usage_id


if __name__ == "__main__":
    run_number = 1
    kaggle_dataset = "airbnb/seattle"
    q = _get_system_prompt(kaggle_dataset=kaggle_dataset, agent_model=AGENT_MODEL)
