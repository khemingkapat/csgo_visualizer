import streamlit as st
import pandas as pd
from visualizer import Visualizer


def movement_page():
    st.title("Movement Analysis")

    if st.session_state.demo_data is None:
        st.warning("No data available. Please upload a file first.")
        st.session_state.page = "home"
        st.rerun()

    # Get processed data from session state
    clean_dfs = st.session_state.clean_dfs
    transformed_data = st.session_state.transformed_data

    # Get available rounds for selection
    available_rounds = clean_dfs["rounds"].reset_index()["round_num"].unique()
    available_rounds = sorted(available_rounds)

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
            format_func=lambda x: f"Round {x}",
            key="round_selector",
        )

    # Display round information
    round_info = clean_dfs["rounds"].loc[selected_round]
    round_result = (
        f"Winner: {round_info['winner'].upper()} | Reason: {round_info['reason']}"
    )
    st.info(round_result)

    # Get tick range for selected round

    round_dfs: dict[str, pd.DataFrame] = {
        k: v[v.index == selected_round] if isinstance(v, pd.DataFrame) else v
        for k, v in transformed_data.items()
    }

    # Create sidebar for controls
    st.sidebar.header("Visualization Controls")
    sampling_rate = st.sidebar.slider("Sampling Rate", 10, 100, 50, step=10)

    # Use round's end tick as the default max_tick

    st.sidebar.subheader("Opacity")
    location_alpha = st.sidebar.slider("Location Alpha", 0.0, 1.0, 0.7)
    community_alpha = st.sidebar.slider("Community Alpha", 0.0, 1.0, 0.7)
    vector_alpha = st.sidebar.slider("Vector Alpha", 0.0, 1.0, 0.7)

    st.subheader(f"Game Actions - Round {selected_round}")
    max_tick = round_dfs["player_locations"].tick.max()

    filtered_data = Visualizer.filter_data_by_tick(round_dfs, 0, int(max_tick * 2))
    fig = Visualizer.create_plotly_actions_plot(
        filtered_data=filtered_data,
        map_name=st.session_state.transformed_data["map"],
        show_loc=True,
        show_smoke=False,
        show_inferno=False,
        show_kills=False,
        show_grenades=False,
        loc_alpha=location_alpha,
        smoke_alpha=0,
        inferno_alpha=0,
        kill_alpha=0,
        grenade_alpha=0,
        smoke_size=0,
        inferno_size=0,
        kill_size=0,
        grenade_size=0,
        show_lines=False,
        fig_height=1200,
    )
    ct_df = Visualizer.get_side_df(
        round_dfs["player_locations"], sampling_rate=sampling_rate, side="ct"
    )
    tpca_result_ct = Visualizer.temporal_pca(ct_df)
    chained_vecs_ct = Visualizer.chain_pc_vectors(tpca_result_ct)
    Visualizer.plot_chained_vector(fig, chained_vecs_ct, alpha=vector_alpha)

    t_df = Visualizer.get_side_df(
        round_dfs["player_locations"], sampling_rate=10, side="t"
    )
    tpca_result_t = Visualizer.temporal_pca(t_df)
    chained_vecs_t = Visualizer.chain_pc_vectors(tpca_result_t)
    Visualizer.plot_chained_vector(fig, chained_vecs_t, alpha=vector_alpha)

    cluster_df = Visualizer.apply_cluster_community(
        round_dfs["player_locations"], sampling_rate=sampling_rate
    )
    Visualizer.plot_community(fig, cluster_df, alpha=community_alpha)

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Explaination")
    st.markdown(
        """
    This page might be a little confusing, but there is 2 main parts there.

    1. **Big chaining arrows** (start with purple one and so on), is the main pushing direction of each team.

    2. **Smaller arrows pointing to each shapes**, each shape correspond to size of group of player, showing the movement, splitting and joining of group of players throughout the match. You can try hovering on each shape to see the data.
    """
    )
