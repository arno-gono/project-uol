import streamlit as st
from config import AVAILABLE_KAGGLE_DATASETS, KAGGLE_DATASET_NAME

def sidebar_config():
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
    st.sidebar.write("session_state dataset:", st.session_state.dataset)



