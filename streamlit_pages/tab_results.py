import streamlit as st


def tab_results_config():
    st.write("tab 2")
    st.write("session_state", st.session_state)

    """
        What to show - Dataset granularity
            - Chart: Agent's score evolution on all the available recs
            - Table with all investigations: nb errors injected, nb diagnostics, nb matches, model used, 
            investigation cost
    """