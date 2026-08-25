import streamlit as st
from app.run import main
import io
from contextlib import redirect_stdout
from streamlit_pages.data import (get_last_reconciliation_log, get_latest_usage, get_all_usage_for_dataset,
                                  get_all_reconciliation_logs)
import pandas as pd


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

def _section_investigation_result_details():

    # Details. Getting the data and converting it to a dataframe.
    all_usage = get_all_usage_for_dataset(dataset_name=st.session_state.dataset)
    all_recs = get_all_reconciliation_logs(dataset_name=st.session_state.dataset)
    df_usage = pd.DataFrame(all_usage)
    df_recs = pd.DataFrame(all_recs)

    # Trimming columns
    df_usage = df_usage[[
        "id",
        "datetime_created_utc",
        "kaggle_table_max_rows",
        "agent_max_sql_rows_read",
        "agent_model",
        "total_cost_usd"
    ]]

    df_recs = df_recs[[
        "id",
        "datetime_created_utc",
        "total_anomalies",
        "total_anomalies_detected_by_agent",
        "anomalies_detected_by_agent",
        "anomalies_not_found_by_agent",
        "incorrect_diagnostics_made_by_agent",
        "score_agent",
    ]]

    # Sorting and columns' header
    df_recs = df_recs.sort_values(by=["datetime_created_utc"], ascending=True)
    df_recs.columns = [c.replace("_", " ").title() for c in df_recs.columns]


    st.markdown(
        df_recs.to_html(index=False),
        unsafe_allow_html=True
    )

def _section_investigation_result_metrics() -> None:
    st.title("Investigation Results")

    # Top Metrics
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
    col_metrics[3].metric(label="Score Agent", value=d_last_result["score_agent"])

    if d_last_usage:
        col_metrics[4].metric(label="Cost Investigation $", value=d_last_usage["total_cost_usd"])

    # Details about type of anomalies found / not found
    anomalies_detected_html = "<ul><li>" + "</li><li>".join(d_last_result["anomalies_detected_by_agent"]) + "</li></ul>"
    anomalies_undetected = "<ul><li>" + "</li><li>".join(d_last_result["anomalies_not_found_by_agent"]) + "</li></ul>"
    false_positive = "<ul><li>" + "</li><li>".join(d_last_result["incorrect_diagnostics_made_by_agent"]) + "</li></ul>"

    st.markdown(f"""
                    - <b>Anomalies detected by agent:</b> (Error Type | Column | Table) {anomalies_detected_html} <br>
                    - <b>Anomalies not detected by agent:</b> (Error Type | Column | Table) {anomalies_undetected} <br>
                    - <b>Incorrect diagnostics from agent:</b> (Error Type | Column | Table | Severity) {false_positive}
                """,
                unsafe_allow_html=True
                )


    return None


def tab_investigation_config():

    # First section: radio buttons to select which function to run
    _section_user_input()

    # Second section: Result of the investigation, last one by default
    st.divider()
    st.subheader("Investigation's Results")

    # Main metrics on latest investigation run
    _section_investigation_result_metrics()

    # Details for all investigations for that table (might move to tab_results)
    _section_investigation_result_details()

