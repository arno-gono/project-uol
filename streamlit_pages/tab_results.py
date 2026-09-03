import streamlit as st
import pandas as pd
from typing import Any
from app.logs_data import get_investigations_for_dataset
from streamlit_pages.tab_investigation import _calc_score_from_log, _parse_error_type_from_list


def _get_investigations_dataframe() -> pd.DataFrame:
    # What a run scored and what it cost are read on the usage_id join, so that both charts describe the same
    # investigations.
    df_recs = get_investigations_for_dataset(dataset_name=st.session_state.dataset)

    if df_recs.empty:
        return df_recs

    # Calculating the agent score
    df_recs["score_agent"] = df_recs.apply(lambda x: _calc_score_from_log(x), axis=1)

    # Sorting entries by datetime_created_utc (ISO Format)
    df_recs = df_recs.sort_values(by=["datetime_created_utc"], ascending=True).reset_index(drop=True)

    # Expecting the agent findings and cost to change according to the model used. Splitting the charts per model.
    df_recs["investigation"] = df_recs.groupby("agent_model").cumcount() + 1

    return df_recs


def _section_user_input(df_recs: pd.DataFrame) -> list[str]:
    """
        User Inputs: the dataset can be chosen in the sidebar. Other filters:
          - Model used
          - Removing the Middle severity? Just keep High and Critical?
          -
    """

    st.subheader("Filters")

    all_models = sorted(df_recs["agent_model"].dropna().unique())

    col_input_1, col_input_2 = st.columns(2)

    with col_input_1:
        st.write(f"Dataset: {st.session_state.dataset}")

    with col_input_2:
        option_models = st.multiselect(
            label="Model:",
            options=all_models,
            default=all_models
        )

    return option_models


def _section_score_evolution(df_recs: pd.DataFrame) -> None:

    st.subheader("Agent Score Evolution")

    if df_recs.empty:
        st.write("No investigation for the model(s) selected")
        return None

    # One line per model, so that the models can be compared on the runs they were each given
    st.line_chart(
        df_recs,
        x="investigation",
        y="score_agent",
        color="agent_model",
        x_label="Investigation",
        y_label="Score Agent"
    )

    return None


def _section_cost_evolution(df_recs: pd.DataFrame) -> None:

    st.subheader("Cost Evolution")

    df_cost = df_recs[df_recs["total_cost_usd"].notna()] if not df_recs.empty else df_recs

    if df_cost.empty:
        st.write("No cost logged for the model selected")
        return None

    # One line per model, the cost mostly depends on which model ran the investigation
    st.line_chart(
        df_cost,
        x="investigation",
        y="total_cost_usd",
        color="agent_model",
        x_label="Investigation",
        y_label="Cost $"
    )

    return None


# Colour scheme for the table showing evolution of errors spotted.
ERROR_TYPE_STATUS_COLOURS = {"found": "green", "missed": "red", "mixed": "orange"}


def _get_error_type_status(found: list[str], not_found: list[str], error_type: str) -> str:
    # The same error can be injected several times in the same run. A "found" is only met when this error has been
    # found every single time by the agent. It is missed when the agent could never find it. Otherwise, it
    #  is "mixed"
    if error_type in found and error_type in not_found:
        return "mixed"

    if error_type in found:
        return "found"

    return "missed"


def _build_error_type_matrix(df_recs: pd.DataFrame) -> pd.DataFrame:
    # Columns are investigations, rows are error types. Cells are populated with either "found", "missed" or "mixed".
    d_matrix = {}

    for rec in df_recs.to_dict("records"):
        found = _parse_error_type_from_list(error_type_list=rec["anomalies_detected_by_agent"])
        not_found = _parse_error_type_from_list(error_type_list=rec["anomalies_not_found_by_agent"])

        d_matrix[rec["id"]] = {error_type: _get_error_type_status(found=found, not_found=not_found,
                                                                  error_type=error_type)
                               for error_type in set(found + not_found)}

    # Converting the dict into a dataframe. Error types sorted alphabetically
    df_matrix = pd.DataFrame(d_matrix).sort_index()

    return df_matrix[sorted(df_matrix.columns)]


def _colour_error_type_cell(status: Any) -> str:
    # If the cell has no text in ERROR_TYPE_STATUS_COLOURS (header for example), then no style is returned.
    if status not in ERROR_TYPE_STATUS_COLOURS:
        return ""

    # HTML inline for the background colour of the cell
    return f"background-color: {ERROR_TYPE_STATUS_COLOURS[status]}"


def _section_error_types(df_recs: pd.DataFrame) -> None:

    st.subheader("Error Types Found by Investigation")

    if df_recs.empty:
        st.write("No investigation for the model(s) selected")
        return None

    # Getting a table with investigations as columns and error types as rows with found/mixed/missed or nothing from
    # the reconciliation logs.
    df_matrix = _build_error_type_matrix(df_recs=df_recs)

    # Replacing nan with nothing:
    df_matrix = df_matrix.fillna("")

    # Legend on top of the table
    st.markdown(f'Legend: :color[found]{{background="{ERROR_TYPE_STATUS_COLOURS["found"]}"}} '
                f':color[missed]{{background="{ERROR_TYPE_STATUS_COLOURS["missed"]}"}} '
                f':color[partly found]{{background="{ERROR_TYPE_STATUS_COLOURS["mixed"]}"}}. '
                f'An empty cell means the error type was not injected in that investigation.')

    # The status is blanked out of the cells so that only the colour is read.
    styler = df_matrix.style.map(_colour_error_type_cell).format(lambda v: "")

    st.dataframe(styler)

    return None


def tab_results_config() -> None:

    st.title("Results")

    # Getting Reconciliations data together with Usage.
    df_recs = _get_investigations_dataframe()

    if df_recs.empty:
        st.write(f"No investigation results for {st.session_state.dataset}")
        return None

    # First section: filters on top of the dataset selected in the sidebar
    option_models = _section_user_input(df_recs=df_recs)

    df_filtered = df_recs[df_recs["agent_model"].isin(option_models)]

    # Second section: how the agent scores over the investigations, and what it costs
    st.divider()

    col_chart_1, col_chart_2 = st.columns(2)

    with col_chart_1:
        _section_score_evolution(df_recs=df_filtered)

    with col_chart_2:
        _section_cost_evolution(df_recs=df_filtered)

    # Third section: which error types the agent finds and which ones it keeps missing, run after run
    st.divider()

    _section_error_types(df_recs=df_filtered)

    return None
