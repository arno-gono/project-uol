from pathlib import Path


KAGGLE_DATASET_NAME = "rikdifos/credit-card-approval-prediction"
# other dataset to try:
# olistbr/brazilian-ecommerce
# shrinivasv/retail-store-star-schema-dataset
# rikdifos/credit-card-approval-prediction

# All db files are stored within a specific folder, anchored to this file so that
# the path is the same no matter which directory the script is run from
DB_DIR = Path(__file__).resolve().parent / "db_files"
