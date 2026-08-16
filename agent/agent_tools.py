import pandas as pd
from config import KAGGLE_DATASET_NAME
from data.utils import get_calibration_file_as_dict
from data.sqlite_connector import connecting_to_sqlite

# Available tools for the agent. Format as per Claude's Documentation.
# This lists the functions available, what they do, the arguments they take so that they can be called by the agent.
# Doc: https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent
TOOLS = [
    {
        "name": "run_sql",
        "description": f"Run a read-only SQL query on a SQLite database and get the rows back. "
                       f"Only SELECT is allowed, one statement per call.",
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
        "description": f"Read the JSON file with calibration data and return it as a dictionary.  "
                       f"It is run on the clean data, ie before the new rows were added to the tables. ",
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

