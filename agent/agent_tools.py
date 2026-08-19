import pandas as pd
from config import KAGGLE_DATASET_NAME
from data.utils import get_calibration_file_as_dict
from data.sqlite_connector import connecting_to_sqlite

#  TODO
"""
    ### **FEEDBACK**
    The investigation model works well overall. A few suggestions for improvement:
    
    1. **Tool Enhancements**: A dedicated tool to check for unexpected categorical values would be useful - 
    the current approach requires manual SQL queries to identify invalid categorical values 
    that weren't in the calibrated set.
    
    2. **Cardinality Checking**: It would be helpful to have an automatic check for columns marked as 
    "potential_primary_key" to verify no duplicates or missing values in new data.
    
    3. **Missing Value Tracking**: A summary comparison tool that specifically tracks NULL/missing 
    value changes would help identify violations faster, especially for columns that previously had no NULLs.
    
    4. **Data Type Validation**: Automatic detection when column data types appear to have changed 
    (e.g., numeric values stored as text strings in OCCUPATION_TYPE).

"""


# Available tools for the agent. Format as per Claude's Documentation.
# This lists the functions available, what they do, the arguments they take so that they can be called by the agent.
# Doc: https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent
TOOLS = [
    {
        "name": "run_sql",
        "description": "Run a read-only SQL query against the current state of the database and get the rows back. "
                       "Only SELECT clause is allowed. "
                       "The tables names can be read from the statement SELECT * FROM sqlite_master. "
                       "The tables corresponding to the calibrated file are separated from the new rows: "
                       "tables with the suffix _new_data are the ones corresponding to new data. A _new_data table "
                       "might be empty if there is no new data. "
                       "In order to optimise costs, try and aggregate data with COUNT, AVG or GROUP BY rather than "
                       "reading a whole table, especially for the one that has been calibrated. ",
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

    if not query.lower()[:6] == "select":
        return {"error": "only SELECT queries are allowed"}

    conn = connecting_to_sqlite(KAGGLE_DATASET_NAME, database_type="agent")

    try:
        df = pd.read_sql(f"SELECT * FROM ({query})", conn)
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

    return d_calibration[table_name]


# The model refers to the tools by their name. Mapping tools names with their function in the following dictionary
TOOLS_FUNCTIONS = {
    "run_sql": run_sql,
    "read_calibration": read_calibration
}

