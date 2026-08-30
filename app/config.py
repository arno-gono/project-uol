from pathlib import Path


KAGGLE_DATASET_NAME = "airbnb/seattle"

AVAILABLE_KAGGLE_DATASETS = [
    "olistbr/brazilian-ecommerce",
    "shrinivasv/retail-store-star-schema-dataset",
    "rikdifos/credit-card-approval-prediction",
    "airbnb/seattle"
]

# Managing API cost for iterating purposes. cutting all tables to a max of KAGGLE_TABLE_MAX_ROWS
# Use None to uncap
KAGGLE_TABLE_MAX_ROWS = None
DATA_CLEAN_TEST_SPLIT = 0.97

# All db files are stored within a specific folder, anchored to this file so that
# the path is the same no matter which directory the script is run from
DB_DIR = Path(__file__).resolve().parent / "db_files"
DB_DIR_AGENT = Path(__file__).resolve().parent / "db_agent"
LOGS_DIR = Path(__file__).resolve().parent / "logs"
INJECTION_LOG_DIR = LOGS_DIR / "injection_logs.json"


def get_db_name(kaggle_dataset: str) -> str:
    # A dataset is referenced on Kaggle as "owner/dataset-name". Only the second part is kept as the
    # reference name, with dashes turned into underscores so that it is usable as a file / table name.
    return kaggle_dataset.split("/")[-1].replace("-", "_")


# DB_NAME will be used in various places as the reference name for the dataset currently selected.
DB_NAME = get_db_name(KAGGLE_DATASET_NAME)

### 1st Calibration - Metadata ###

# Maximum number of unique values for a field to be considered as a categorical field.
# Expressed as an absolute number, ie if the number of unique values is less than MAX_CARDINALITY_NB,
# it is considered as a categorical field. Left as a parameter for now to readjust for new datasets.
MAX_CARDINALITY_NB = 100

# Minimum share of the non-null values of a string column that must parse as dates for the column to be
# flagged as a datetime one. Left below 1 so that a few dirty values do not disqualify a genuine date column.
MIN_DATETIME_PARSE_RATIO = 0.95

### Agent ###

# API costs per million Tokens, in USD, for every model the agent can be run on.
# Doc: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
AGENT_MODELS_COSTS = {
    "claude-haiku-4-5": {
        "total_input_tokens": 1,
        "total_output_tokens": 5,
        "total_cache_read": 0.1,
        "total_cache_written": 1.25
    },
    "claude-sonnet-5": {
        "total_input_tokens": 2,
        "total_output_tokens": 10,
        "total_cache_read": 0.2,
        "total_cache_written": 2.5
    },
    "claude-opus-5": {
        "total_input_tokens": 5,
        "total_output_tokens": 25,
        "total_cache_read": 0.5,
        "total_cache_written": 6.25
    },
}

# A model can only be selected if its cost is known, so the choices offered are the models priced above.
AVAILABLE_MODELS = list(AGENT_MODELS_COSTS.keys())

# Model the agent runs on by default, and a ceiling in tokens being used (cost control).
AGENT_MODEL = "claude-haiku-4-5"
AGENT_MAX_TOKENS = 3_000

# Maximum number of rows a single run_sql call can return (cost control). A tool result stays in the
# conversation, so a whole table read is sent back to the API in every following round of the investigation.
AGENT_MAX_ROWS_RETURNED = 100

AGENT_LOG_DIR = LOGS_DIR / "agent_logs.json"
AGENT_USAGE_DIR = LOGS_DIR / "usage.json"
AGENT_FEEDBACK_DIR = LOGS_DIR / "agent_feedback.json"
RECONCILIATION_LOG_DIR = LOGS_DIR / "reconciliation_logs.json"
