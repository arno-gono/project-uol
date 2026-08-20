from config import INJECTION_LOG_DIR, AGENT_LOG_DIR
import json


def reconcile_agent_vs_injection_logs(run_number: int) -> None:
    dict_agent = json.load(open(AGENT_LOG_DIR))
    dict_inj = json.load(open(INJECTION_LOG_DIR))

    all_inj_errors = [inj_errors["dict_rec"]
                      for inj_errors in dict_inj["injections"] if inj_errors["run_number"] == run_number]

    all_agents_diagnostics = [investigation["params"]
                              for investigation in dict_agent["investigation"] if investigation["run_number"] == run_number]

    required_keys_to_match = ["table", "column", "anomaly"]
    accurate_keys_to_match = ["nb_rows_affected"]

    # Checking if all the errors injected have been found
    errors_found_by_agent = []
    for error_injected in all_inj_errors:
        for diagnostic in all_agents_diagnostics:
            # Checking if all the required keys have been found
            error_found = True
            for required_key in required_keys_to_match:

                # Exception: no column entered for some errors (duplicate rows for example)
                # The agent enters something random. Needs to be fixed at agent level though
                if required_key == "column" and error_injected["column"] == "":
                    continue

                if error_injected[required_key] != diagnostic[required_key]:
                    error_found = False
                    break

            # The injected error matches a diagnostic on 3 required keys. It is considered to have been found
            if error_found:
                dict_rec = {
                    "error_id": error_injected["id"],
                    "diagnostic_id": diagnostic["id"],
                }
            else:
                continue

            for bonus_keys in accurate_keys_to_match:
                # Casting all variables as strings
                if str(error_injected[bonus_keys]) != str(diagnostic[bonus_keys]):
                    dict_rec[bonus_keys] = False
                else:
                    dict_rec[bonus_keys] = True

            errors_found_by_agent.append(dict_rec)

    for i in errors_found_by_agent:
        print(i)

    return None


if __name__ == "__main__":
    run_number = 1
