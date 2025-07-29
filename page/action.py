import streamlit as st
from visualizer import Visualizer


def action_page():
    st.title("Action Analysis")

    if st.session_state.demo_data is None:
        st.warning("No data available. Please upload a file first.")
        st.session_state.page = "home"
        st.rerun()

    clean_dfs = st.session_state.clean_dfs
    available_rounds = clean_dfs["rounds"].reset_index()["round_num"].unique()
    available_rounds = sorted(available_rounds)[1:]

    # Create two columns for the match info and round selection
    match_col, round_col = st.columns([3, 1])

    with match_col:
        # Match information
        match_info = clean_dfs["other_data"].iloc[0]
        st.subheader("Match Information")
        st.info(
            f"Map: {match_info['map_name']} | Date: {match_info['tournament_name']}"
        )

    with round_col:
        # Round selection
        st.subheader("Round Selection")
        selected_round = st.selectbox(
            "Select Round",
            options=available_rounds,
            index=1,
            format_func=lambda x: f"Round {x}",
            key="round_selector",
        )

    fig = Visualizer.plot_location_change_analysis(clean_dfs, selected_round)

    st.plotly_chart(fig, use_container_width=True)
