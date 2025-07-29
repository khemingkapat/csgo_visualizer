import streamlit as st
import pandas as pd
from visualizer import Visualizer


def location_page():
    st.title("Location Analysis")

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
    start_tick = round_dfs["player_locations"].tick.min()
    end_tick = round_dfs["player_locations"].tick.max()

    # Create sidebar for controls
    st.sidebar.header("Visualization Controls")

    # Use round's end tick as the default max_tick
    st.sidebar.subheader("Tick Control")
    max_tick = st.sidebar.slider(
        "Max Tick",
        min_value=start_tick,
        max_value=int(end_tick),
        value=(start_tick + end_tick) // 2,
        help="Show events from tick 0 up to this tick value",
    )

    # Event visibility controls
    st.sidebar.subheader("Event Types")
    show_loc = st.sidebar.checkbox("Show Location", value=True)
    show_smoke = st.sidebar.checkbox("Show Smokes", value=True)
    show_inferno = st.sidebar.checkbox("Show Infernos", value=True)
    show_kills = st.sidebar.checkbox("Show Kills", value=True)
    show_grenades = st.sidebar.checkbox("Show Grenades", value=True)
    show_lines = st.sidebar.checkbox("Show Lines", value=True)

    # Visual styling controls
    with st.sidebar.expander("Visual Settings", expanded=False):
        col1, col2 = st.sidebar.columns(2)

        with col1:
            st.subheader("Opacity")
            smoke_alpha = st.slider("Smoke Alpha", 0.1, 1.0, 0.7)
            inferno_alpha = st.slider("Inferno Alpha", 0.1, 1.0, 0.7)
            kill_alpha = st.slider("Kill Alpha", 0.1, 1.0, 0.7)
            grenade_alpha = st.slider("Grenade Alpha", 0.1, 1.0, 0.7)

        with col2:
            st.subheader("Size")
            smoke_size = 30 + int(st.checkbox("Big Smoke", value=False)) * 20
            inferno_size = 30 + int(st.checkbox("Big Inferno", value=False)) * 20
            kill_size = 30 + int(st.checkbox("Big Kill", value=False)) * 20
            grenade_size = 30 + int(st.checkbox("Big Grenade", value=False)) * 20

    st.subheader(f"Game Actions - Round {selected_round}")

    # Update visualization if button clicked or parameters changed
    filtered_data = Visualizer.filter_data_by_tick(round_dfs, 0, max_tick)
    fig = Visualizer.create_plotly_actions_plot(
        filtered_data=filtered_data,
        map_name=st.session_state.transformed_data["map"],
        show_loc=show_loc,
        show_smoke=show_smoke,
        show_inferno=show_inferno,
        show_kills=show_kills,
        show_grenades=show_grenades,
        loc_alpha=0.5,
        smoke_alpha=smoke_alpha,
        inferno_alpha=inferno_alpha,
        kill_alpha=kill_alpha,
        grenade_alpha=grenade_alpha,
        smoke_size=smoke_size,
        inferno_size=inferno_size,
        kill_size=kill_size,
        grenade_size=grenade_size,
        show_lines=show_lines,
        fig_height=1200,
    )

    st.plotly_chart(fig, use_container_width=True)
    # st.session_state.visualization_updated = True

    # Show statistics
    with st.expander("Round Statistics", expanded=False):
        st.subheader(f"Statistics for Round {selected_round}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Smoke",
                (
                    len(filtered_data["smokes"])
                    if isinstance(filtered_data["smokes"], pd.DataFrame)
                    else 0
                ),
            )
            if not filtered_data["smokes"].empty:
                st.write("Smoke Distribution:")
                side_counts = filtered_data["smokes"]["thrower_side"].value_counts()
                st.bar_chart(side_counts, use_container_width=True)

        with col2:
            st.metric(
                "Kills",
                (
                    len(filtered_data["kills"])
                    if isinstance(filtered_data["kills"], pd.DataFrame)
                    else 0
                ),
            )
            if not filtered_data["kills"].empty:
                st.write("Kill Distribution:")
                side_counts = filtered_data["kills"]["attacker_side"].value_counts()
                st.bar_chart(side_counts, use_container_width=True)

        with col3:
            st.metric(
                "Grenades",
                (
                    len(filtered_data["grenades"])
                    if isinstance(filtered_data["grenades"], pd.DataFrame)
                    else 0
                ),
            )
            if not filtered_data["grenades"].empty:
                st.write("Grenade Distribution:")
                side_counts = filtered_data["grenades"]["thrower_side"].value_counts()
                st.bar_chart(side_counts, use_container_width=True)

        with col4:
            st.metric(
                "Infernos",
                (
                    len(filtered_data["infernos"])
                    if isinstance(filtered_data["infernos"], pd.DataFrame)
                    else 0
                ),
            )
            if not filtered_data["infernos"].empty:
                st.write("Inferno Distribution:")
                side_counts = filtered_data["infernos"]["thrower_side"].value_counts()
                st.bar_chart(side_counts, use_container_width=True)
