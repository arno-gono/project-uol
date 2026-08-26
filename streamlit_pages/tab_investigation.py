import streamlit as st
from app.run import main
import io
from contextlib import redirect_stdout
from streamlit_pages.data import (get_last_reconciliation_log, get_latest_usage, get_all_usage_for_dataset,
                                  get_all_reconciliation_logs)
import pandas as pd
from app.reconciliation.utils import calc_score_agent


def _section_user_input():
    st.title("Run an investigation")
    st.divider()

    # User Inputs: choosing which function is going to be run
    st.subheader("Functions to run")

    rb_col_1, rb_col_2, rb_col_3, rb_col_4, rb_col_5 = st.columns(5)

    with rb_col_1:
        option_import_data = st.radio("Import Data", [True, False], index=0)

    with rb_col_2:
        option_run_calibration = st.radio("Run Calibration", [True, False], index=0)

    with rb_col_3:
        option_inject_errors = st.radio("Inject Errors", [True, False], index=0)

    with rb_col_4:
        option_call_agent = st.radio("Call Agent", [True, False], index=0)

    with rb_col_5:
        option_reconcile = st.radio("Reconcile Logs", [True, False], index=0)

    output = []
    if st.button("Run Investigation"):

        output = io.StringIO()
        with redirect_stdout(output):
            main(
                import_dataset_and_upload=option_import_data,
                run_calibration=option_run_calibration,
                inject_errors=option_inject_errors,
                call_agent_investigation=option_call_agent,
                reconcile_agent_vs_injection=option_reconcile
            )

    with st.expander("Console Output", expanded=True if not output else False):
        if output:
            st.text(output.getvalue())
            st.write("Operation(s) Completed")
        else:
            st.text("Click the button")


def _parse_error_type_from_list(error_type_list: list) -> list:
    all_errors = [t.split(" | ")[0].strip() for t in error_type_list]
    return all_errors


def _count_false_positives_by_severity(incorrect_diagnostics: list, severity: str) -> int:
    # An incorrect diagnostic is stored as "error type | column | table | severity", the severity being
    # the last criterion of the chain
    all_severities = [d.split(" | ")[-1].strip().lower() for d in incorrect_diagnostics]
    return all_severities.count(severity.lower())


def _calc_score_from_log(d_rec) -> float:
    # The score is recalculated from the reconciliation log rather than read from it, so a change in the
    # scoring applies to the runs that are already logged
    incorrect_diagnostics = d_rec["incorrect_diagnostics_made_by_agent"]

    score_agent = calc_score_agent(
        nb_errors_injected=d_rec["total_anomalies"],
        nb_errors_found=d_rec["total_anomalies_detected_by_agent"],
        nb_false_positive_high=_count_false_positives_by_severity(
            incorrect_diagnostics=incorrect_diagnostics, severity="High"
        ),
        nb_false_positive_critical=_count_false_positives_by_severity(
            incorrect_diagnostics=incorrect_diagnostics, severity="Critical"
        )
    )

    return round(score_agent, 4)



def _section_investigation_result_details():

    st.divider()
    st.subheader("Previous Runs")

    # Details. Getting the data and converting it to a dataframe.
    all_usage = get_all_usage_for_dataset(dataset_name=st.session_state.dataset)
    all_recs = get_all_reconciliation_logs(dataset_name=st.session_state.dataset)
    df_usage = pd.DataFrame(all_usage)
    df_recs = pd.DataFrame(all_recs)

    if not df_usage.empty:
        # Trimming columns
        df_usage = df_usage[[
            "id",
            "datetime_created_utc",
            "kaggle_table_max_rows",
            "agent_max_sql_rows_read",
            "agent_model",
            "total_cost_usd"
        ]]

    if not df_recs.empty:
        df_recs = df_recs[[
            "id",
            "datetime_created_utc",
            "total_anomalies",
            "total_anomalies_detected_by_agent",
            "anomalies_detected_by_agent",
            "anomalies_not_found_by_agent",
            "incorrect_diagnostics_made_by_agent"
        ]]

        # Scoring each run before the chains are trimmed, the severity of the false positives is needed
        df_recs.insert(4, "score_agent", df_recs.apply(_calc_score_from_log, axis=1))

        # The column and table are removed from the details, we are more interested in the type of errors that are not
        # correctly flagged. Each cell holds a list of chains such as "error_type | column | table", so we only keep
        # the first element of every chain.
        for c in [
            "anomalies_detected_by_agent", "anomalies_not_found_by_agent", "incorrect_diagnostics_made_by_agent"
        ]:
            df_recs[c] = df_recs[c].apply(lambda x: ", ".join(set(_parse_error_type_from_list(x))))

        # Sorting and columns' header
        df_recs = df_recs.sort_values(by=["datetime_created_utc"], ascending=True)
        df_recs.columns = [c.replace("_", " ").title() for c in df_recs.columns]

        st.markdown(
            df_recs.to_html(index=False),
            unsafe_allow_html=True
        )


def _section_investigation_result_metrics() -> None:

    # Top Metrics
    st.divider()
    st.subheader("Top Metrics")

    d_last_result = get_last_reconciliation_log(
        dataset_name=st.session_state.dataset
    )

    d_last_usage = get_latest_usage(
        dataset_name=st.session_state.dataset
    )

    if d_last_result is None:
        st.write(f"No investigation results for {st.session_state.dataset}")
        return None

    # Main metrics
    st.write(f"Last investigation: {d_last_result['datetime_created_utc']}")
    col_metrics = st.columns(5)

    col_metrics[0].metric(label="Errors Injected", value=d_last_result["total_anomalies"])
    col_metrics[1].metric(label="Nb Diagnostics", value=d_last_result["total_diagnostics_made_by_agent"])
    col_metrics[2].metric(label="Anomalies Found by Agent", value=d_last_result["total_anomalies_detected_by_agent"])
    col_metrics[3].metric(label="Score Agent", value=_calc_score_from_log(d_last_result))

    if d_last_usage:
        col_metrics[4].metric(label="Cost Investigation $", value=d_last_usage["total_cost_usd"])

    # Details about type of anomalies found / not found
    st.divider()
    st.subheader("Details")

    col_details = st.columns(3)

    anomalies_detected_html = "<ul><li>" + "</li><li>".join(d_last_result["anomalies_detected_by_agent"]) + "</li></ul>"
    anomalies_undetected = "<ul><li>" + "</li><li>".join(d_last_result["anomalies_not_found_by_agent"]) + "</li></ul>"
    false_positive = "<ul><li>" + "</li><li>".join(d_last_result["incorrect_diagnostics_made_by_agent"]) + "</li></ul>"

    col_details[0].markdown(f"<b>Anomalies detected by agent:</b><br>(Error Type | Column | Table)"
                            f" {anomalies_detected_html}",
                            unsafe_allow_html=True)
    col_details[1].markdown(f"<b>Anomalies not detected by agent:</b><br>(Error Type | Column | Table)"
                            f" {anomalies_undetected}",
                            unsafe_allow_html=True)
    col_details[2].markdown(f"<b>Incorrect diagnostics from agent:</b><br>(Error Type | Column | Table | Severity) "
                            f"{false_positive}",
                            unsafe_allow_html=True)

    return None


def tab_investigation_config():

    # First section: radio buttons to select which function to run
    _section_user_input()
    st.divider()

    # Second section: Result of the investigation, last one by default
    st.divider()
    st.title("Investigation Results")

    # Main metrics on latest investigation run
    _section_investigation_result_metrics()

    # Details for all investigations for that table (might move to tab_results)
    _section_investigation_result_details()

