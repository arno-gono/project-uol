from data.kaggle_data import download_kaggle_upload_to_sqlite
from data_calibration.calibration_1st_layer import dataset_calibration
from errors_injection.random_injection_runner import run_multiple_rounds
from agent.agent_run import run_agent_investigation
from reconciliation.reconcile_logs import reconcile_agent_vs_injection_logs


def main(import_dataset_and_upload: bool = True,
         run_calibration: bool = True,
         inject_errors: bool = True,
         call_agent_investigation: bool = True,
         reconcile_agent_vs_injection: bool = True):


    # Downloading data from a Kaggle dataset and migrating to SQLite
    if import_dataset_and_upload:
        download_kaggle_upload_to_sqlite()

    # Data profiling: writing a calibration file on the data considered clean
    if run_calibration:
        calibration = dataset_calibration()

    # Automation loop: running loops of errors randomly injected into a dataset and calling an agent investigation
    if inject_errors:
        nb_runs = 1

        for run_number in range(1, nb_runs + 1):
            # Generating errors and injecting them in the test dataset
            print(f"\n\n######## Round {run_number} ########")
            run_multiple_rounds(run_number)

    # Calling the agents to run an investigation
    if call_agent_investigation:
        run_agent_investigation()

    # Checking agent's findings against what was actually injected
    if reconcile_agent_vs_injection:
        reconcile_agent_vs_injection_logs()