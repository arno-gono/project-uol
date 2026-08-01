from pathlib import Path


KAGGLE_DATASET_NAME = "rikdifos/credit-card-approval-prediction"
# other dataset to try:
# olistbr/brazilian-ecommerce
# shrinivasv/retail-store-star-schema-dataset
# rikdifos/credit-card-approval-prediction

# All db files are stored within a specific folder, anchored to this file so that
# the path is the same no matter which directory the script is run from
DB_DIR = Path(__file__).resolve().parent / "db_files"


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
MAX_CARDINALITY_NB = 50

# Minimum share of the non-null values of a string column that must parse as dates for the column to be
# flagged as a datetime one. Left below 1 so that a few dirty values do not disqualify a genuine date column.
MIN_DATETIME_PARSE_RATIO = 0.95
