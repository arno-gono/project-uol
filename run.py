from data.kaggle_data import download_kaggle_upload_to_sqlite
from data_calibration.calibration_1st_layer import dataset_calibration


def main():
    download_kaggle_upload_to_sqlite()
    dataset_calibration()

