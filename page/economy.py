import streamlit as st
from visualizer import Visualizer


def economy_page():
    st.title("Economy Analysis")

    if st.session_state.json_data is None:
        st.warning("No data available. Please upload a file first.")
        st.session_state.page = "home"
        st.rerun()

    transformed_data = st.session_state.transformed_data
    # Create a nicer layout for "under development" message
    fig = Visualizer.plot_combined_economy_with_reasons(transformed_data["rounds_sum"])
    st.plotly_chart(fig, use_container_width=True)
