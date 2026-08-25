import streamlit as st
from streamlit_pages.sidebar import sidebar_config
from streamlit_pages.tab_results import tab_results_config
from streamlit_pages.tab_investigation import tab_investigation_config


# Page configuration
st.set_page_config(
    page_title="Agentic Diagnostics",
    page_icon="🩺",
    layout="wide"
)

# Sidebar
sidebar_config()

# Two tabs for navigation: Investigation (+ result of the investigation) and the overall,
# aggregated results (charts, tables)
tab1, tab2 = st.tabs(["Investigation", "Results"])

with tab1:
    tab_investigation_config()

with tab2:
    tab_results_config()

