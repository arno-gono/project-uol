import pandas as pd
from app.config import KAGGLE_DATASET_NAME, AGENT_MAX_ROWS_RETURNED
from app.data.utils import get_calibration_file_as_dict
from app.data.sqlite_connector import connecting_to_sqlite


# Available tools for the agent. Format as per Claude's Documentation.
# This lists the functions available, what they do, the arguments they take so that they can be called by the agent.
# Doc: https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent
TOOLS = [
    # Some functions are not available in SQLite (like CORR or STDDEV). Flagged by the agent. Adding it to the prompt
    # to avoid wasting rounds on trying those SQL functions.
    # Doc: https://sqlite.org/lang_aggfunc.html
    {
        "name": "run_sql",
        "description": "Run a read-only SQL query against the current state of the database and get the rows back. "
                       "Only SELECT clause is allowed. "
                       "The database is SQLite: CORR, STDDEV, VARIANCE do not exist. "
                       "The tables names can be read from the statement SELECT * FROM sqlite_master. "
                       "The tables corresponding to the calibrated file are separated from the new rows: "
                       "tables with the suffix _new_data are the ones corresponding to new data. A _new_data table "
                       "might be empty if there is no new data. "
                       "In order to optimise costs, try and aggregate data with COUNT, AVG or GROUP BY rather than "
                       "reading a whole table, especially for the one that has been calibrated. "
                       f"At most {AGENT_MAX_ROWS_RETURNED} rows are returned, so aggregate rather than expecting to "
                       "read more than that in one go.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A single SELECT statement"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_calibration",
        "description": "Return the calibration profile of one table: how many rows and duplicated rows it held, the "
                       "details of each of its columns (datatype, missing values, unique values, distribution), the "
                       "correlations and associations between those columns, the columns that are potential primary "
                       "or foreign keys, and a clustering of the rows obtained with machine learning techniques. "
                       "The calibration was computed on the clean data, that is before the new rows were appended, "
                       "(i.e. tables ending with _new_data) and it describes what the table looked like when "
                       "it was known to be correct. "
                       "Calling it with a name that was not calibrated returns the list of the tables that exist in the "
                       "file, which is one way of finding out which tables exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "Name of the table the data profile is wanted for"},
            },
            "required": ["table_name"],
        },
    }
]


def run_sql(query: str) -> dict:
    # tool allowing to read the database.
    # Only allowing SELECT clauses to be run

    # The query is wrapped in a subquery below, and the agent tends to add a semicolon.
    # This results in an error: saving a round for the agent
    query = query.strip().rstrip(";").strip()

    if not query.lower()[:6] == "select":
        return {"error": "only SELECT queries are allowed"}

    conn = connecting_to_sqlite(KAGGLE_DATASET_NAME, database_type="agent")

    try:
        df = pd.read_sql(f"SELECT * FROM ({query}) LIMIT {AGENT_MAX_ROWS_RETURNED}", conn)
    except Exception as e:
        # The error is handed back to the agent so it can correct its query and try again.
        print(f"\t\033[91mquery failed: {e}\033[0m")
        return {"error": f"query failed: {e}"}
    finally:
        conn.close()

    return {
        "nb_rows_returned": len(df),
        "rows": df.to_dict(orient="records"),
    }


def read_calibration(table_name: str) -> dict:
    # Allowing the agent to access the calibration file
    d_calibration = get_calibration_file_as_dict()

    if table_name not in d_calibration:
        # Listing what exists rather than just refusing, so the agent can correct itself
        print(f"\t\033[91merror: {table_name} was not calibrated. Calibrated tables: {list(d_calibration.keys())}\033[0m")
        return {"error": f"{table_name} was not calibrated", "calibrated_tables": list(d_calibration.keys())}

    # Testing cost without passing the ml_calibration part of the calibration file.
    return {key: value for key, value in d_calibration[table_name].items() if key != "ml_calibration"}


# The model refers to the tools by their name. Mapping tools names with their function in the following dictionary
TOOLS_FUNCTIONS = {
    "run_sql": run_sql,
    "read_calibration": read_calibration
}

