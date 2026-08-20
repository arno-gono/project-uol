from pathlib import Path


KAGGLE_DATASET_NAME = "rikdifos/credit-card-approval-prediction"
# other dataset to try:
# olistbr/brazilian-ecommerce
# shrinivasv/retail-store-star-schema-dataset
# rikdifos/credit-card-approval-prediction

# Managing API cost for iterating purposes. cutting all tables to a max of KAGGLE_TABLE_MAX_ROWS
# Use None to uncap
KAGGLE_TABLE_MAX_ROWS = 1_000

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

# Model the agent runs on, and a ceiling in tokens being used (cost control).
AGENT_MODEL = "claude-haiku-4-5"
AGENT_MAX_TOKENS = 3_000

# Maximum number of rows a single run_sql call can return (cost control). A tool result stays in the
# conversation, so a whole table read is sent back to the API in every following round of the investigation.
AGENT_MAX_ROWS_RETURNED = 100

AGENT_LOG_DIR = LOGS_DIR / "agent_logs.json"
AGENT_USAGE_DIR = LOGS_DIR / "usage.json"
AGENT_FEEDBACK_DIR = LOGS_DIR / "agent_feedback.json"
