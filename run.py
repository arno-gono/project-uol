from data.kaggle_data import download_kaggle_upload_to_sqlite
from data_calibration.calibration_1st_layer import dataset_calibration
from errors_injection.random_injection_runner import run_multiple_rounds

def main():

    # Downloading data from a Kaggle dataset and migrating to SQLite
    download_kaggle_upload_to_sqlite()

    # Data profiling: writing a calibration file on the data considered clean
    calibration = dataset_calibration()

    # Automation loop: running loops of errors randomly injected into a dataset and calling an agent investigation
    nb_runs = 10

    for run_number in range(1, nb_runs + 1):
        # Generating errors and injecting them in the test dataset
        print(f"\n\n######## Round {run_number} ########")

        run_multiple_rounds(run_number)

        # Calling the agent for the investigation

