import pandas as pd

from config import KAGGLE_DATASET_NAME
from data.sqlite_connector import connecting_to_sqlite

# Available tools for the agent. Format as per Claude's Documentation.
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
    }
]


def run_sql(query: str) -> dict:
    # tool allowing to read the database.
    # Only allowing SELECT clauses to be run

    if not query.lower()[:6]:
        return {"error": "only SELECT queries are allowed"}

    conn = connecting_to_sqlite(KAGGLE_DATASET_NAME, database_type="agent")

    try:
        df = pd.read_sql(f"SELECT * FROM ({query})", conn)
    except Exception as e:
        # The error is handed back to the agent so it can correct its query and try again.
        return {"error": f"query failed: {e}"}
    finally:
        conn.close()

    return {
        "nb_rows_returned": len(df),
        "rows": df.to_dict(orient="records"),
    }

# The model refers to the tools by their name. Mapping tools names with their function in the following dictionary
TOOLS_FUNCTIONS = {
    "run_sql": run_sql,
}

