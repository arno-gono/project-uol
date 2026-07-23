from pathlib import Path


KAGGLE_DATASET_NAME = "rikdifos/credit-card-approval-prediction"
# other dataset to try:
# olistbr/brazilian-ecommerce
# shrinivasv/retail-store-star-schema-dataset
# rikdifos/credit-card-approval-prediction

# All db files are stored within a specific folder, anchored to this file so that
# the path is the same no matter which directory the script is run from
DB_DIR = Path(__file__).resolve().parent / "db_files"

# DB_NAME will be used in various places as the reference name for the dataset.
DB_NAME = KAGGLE_DATASET_NAME.split("/")[-1].replace("-", "_")

### 1st Calibration - Metadata ###

# Maximum number of unique values for a field to be considered as a categorical field.
# Expressed as an absolute number, ie if the number of unique values is less than MAX_CARDINALITY_NB,
# it is considered as a categorical field. Left as a parameter for now to readjust for new datasets.
MAX_CARDINALITY_NB = 50