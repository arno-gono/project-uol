import streamlit as st
from app.config import AVAILABLE_KAGGLE_DATASETS, AVAILABLE_MODELS


def sidebar_config() -> None:
    # Sidebar
    st.sidebar.title("Dataset")

    option_dataset = st.sidebar.radio(
        label="Dataset:",
        options=AVAILABLE_KAGGLE_DATASETS,
        label_visibility="hidden"
    )

    # Need to have the KAGGLE_DATASET_NAME not as a global variable but an input from the user / argument in functions
    # Dataset name saved in session_state, which is streamlit cache, so that it can be reused in pages.
    st.session_state.dataset = option_dataset

    st.sidebar.title("Model")

    option_model = st.sidebar.radio(
        label="Model:",
        options=AVAILABLE_MODELS,
        label_visibility="hidden"
    )

    # Same principle for the model the agent runs on, saved in session_state so that an investigation is
    # priced and logged against the model that was actually selected here.
    st.session_state.model = option_model

    return None
