from data.kaggle_data import download_kaggle_upload_to_sqlite
from data_calibration.calibration_1st_layer import dataset_calibration
from errors_injection.random_injection_runner import run_multiple_rounds
from agent.agent_run import run_agent_investigation
from reconciliation.reconcile_logs import reconcile_agent_vs_injection_logs

"""
    Parameters for the run:
        1- Importing and uploading a new database to SQLite
        2- Run a calibration on the dataset
        3- Inject errors
        4- Call the agent to run an investigation
        5- Reconcile the agent's investigation with the errors injected
"""

IMPORT_DATASET_AND_UPLOAD = True
RUN_CALIBRATION = True
INJECT_ERRORS = True
CALL_AGENT_INVESTIGATION = True
RECONCILE_AGENTS_FINDINGS = True

def main():

    # Downloading data from a Kaggle dataset and migrating to SQLite
    if IMPORT_DATASET_AND_UPLOAD:
        download_kaggle_upload_to_sqlite()

    # Data profiling: writing a calibration file on the data considered clean
    if RUN_CALIBRATION:
        calibration = dataset_calibration()

    # Automation loop: running loops of errors randomly injected into a dataset and calling an agent investigation
    if INJECT_ERRORS:
        nb_runs = 1

        for run_number in range(1, nb_runs + 1):
            # Generating errors and injecting them in the test dataset
            print(f"\n\n######## Round {run_number} ########")
            run_multiple_rounds(run_number)

    # Calling the agents to run an investigation
    if CALL_AGENT_INVESTIGATION:
        run_agent_investigation()

    # Checking agent's findings against what was actually injected
    if RECONCILE_AGENTS_FINDINGS:
        reconcile_agent_vs_injection_logs()