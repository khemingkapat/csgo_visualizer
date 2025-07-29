import streamlit as st
from visualizer import Visualizer


def overview_page():
    st.title("Match Overview")

    if st.session_state.demo_data is None:
        st.warning("No data available. Please upload a file first.")
        st.session_state.page = "home"
        st.rerun()

    # Add match information
    clean_dfs = st.session_state.clean_dfs
    transformed_data = st.session_state.transformed_data
    match_info = clean_dfs["other_data"].iloc[0]

    st.subheader("Match Information")
    st.info(f"Map: {match_info['map_name']} | Date: {match_info['tournament_name']}")

    fig = Visualizer.plot_round_timeline_plotly(transformed_data["round_results"])
    st.plotly_chart(fig, use_container_width=True)

    # Use columns to better organize content in full width
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Player Statistics")
        st.dataframe(transformed_data["player_stats"], use_container_width=True)
    with col2:
        st.subheader("Side Win Features' Different")
        percentage_df = Visualizer.get_value_difference(transformed_data["rounds_sum"])
        fig = Visualizer.plot_scaled_feature_difference(percentage_df)
        st.plotly_chart(fig, use_container_width=True)
