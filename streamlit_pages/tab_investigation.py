import streamlit as st
from run import main
import io
from contextlib import redirect_stdout


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


def _section_investigation_result():
    st.title("Investigation Results")

def tab_investigation_config():

    # First section: radio buttons to select which function to run
    _section_user_input()

    # Second section: Result of the investigation
    st.divider()
    st.subheader("Investigation's Results")

    # Only show if there is a reconciliation that has been done
    """
        Not sure how to get the latest investigation results. Maybe compare with datetime? Take the last one as default?
        
        Metrics: 
            - nb errors injected
            - nb diagnostics made (by severity)
            - nb matches
            - Agent's score
            - Investigation cost
    """



