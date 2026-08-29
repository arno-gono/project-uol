import streamlit as st
import pandas as pd
from app.logs_data import get_investigations_for_dataset
from streamlit_pages.tab_investigation import _calc_score_from_log


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


def _section_user_input(df_recs: pd.DataFrame) -> list:
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


def tab_results_config():

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
