import streamlit as st
import json
from utils.components import (
    upload_and_parse_json,
)
from utils.new_components import plot_map
from utils.parsers import parse_json_to_dfs
from utils.preprocessing import preprocess
from utils.transformers import transform_all_data
from utils.symbols import *
import plotly.graph_objects as go

st.set_page_config(layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"
if "json_data" not in st.session_state:
    st.session_state.json_data = None
if "clean_dfs" not in st.session_state:
    st.session_state.clean_dfs = None
if "transformed_data" not in st.session_state:
    st.session_state.transformed_data = None


@st.cache_resource
def load_map_data():
    with open(".awpy/maps/map-data.json", "r") as f:
        return json.load(f)


all_map_data = load_map_data()


# Cache the preprocessing steps to avoid redundant computations
@st.cache_data
def load_and_preprocess_data(json_data):
    """Load and preprocess all data at once"""
    dfs = dict(parse_json_to_dfs(json_data))
    clean_dfs = preprocess(dfs)
    return clean_dfs


# Get tick range for a specific round
def get_round_tick_range(clean_dfs, round_num):
    round_data = clean_dfs["rounds"].loc[round_num]
    start_tick = round_data["start_tick"]
    end_tick = round_data["end_tick"]
    return start_tick, end_tick


# Home page for file upload
def home_page():
    st.title("CS:GO Match Analysis Dashboard")
    st.subheader("Upload Match Data")

    st.write(
        """
    Welcome to the CS:GO Match Analysis Dashboard! 
    
    This tool helps you analyze Counter-Strike: Global Offensive match data with detailed visualizations of player movements, kills, flashes, and grenades.
    
    To begin, please upload a CS:GO match JSON file below.
    """
    )

    # Center the upload button in a container with max width
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        uploaded_data = upload_and_parse_json(preview_limit=0)

    if uploaded_data is not None:
        # Store data in session state
        st.session_state.json_data = uploaded_data

        # Process data immediately and store in session state
        with st.spinner("Processing data... This may take a moment."):
            # Parse and preprocess
            dfs = dict(parse_json_to_dfs(uploaded_data))
            st.session_state.clean_dfs = preprocess(dfs)

            # Transform data
            st.session_state.transformed_data = transform_all_data(
                st.session_state.clean_dfs, all_map_data
            )

        # Set page to overview
        st.session_state.page = "location"
        st.success("File successfully uploaded! Redirecting to Overview page...")
        st.rerun()


def location_page():
    st.title("Location Analysis")
    transformed_data = st.session_state.transformed_data
    player_loc = transformed_data["player_locations"]

    map_name = transformed_data["map"]
    # map_name = "de_mirage"
    fig = plot_map(map_name, fig_height=800)

    scatter = go.Scatter(x=player_loc["x"], y=player_loc["y"], mode="markers")

    fig.add_trace(scatter)
    _, mid, _ = st.columns([1, 8, 1])
    with mid:
        st.plotly_chart(fig, use_container_width=True)


# Main app function with navigation
def main():
    pages = {"home": home_page, "location": location_page}

    # Display the current page based on session state
    current_page = st.session_state.page

    # Show navigation bar only for non-home pages
    if current_page != "home":

        # Add a button to return to upload page if needed
        if st.sidebar.button("Upload New File"):
            st.session_state.page = "home"
            st.session_state.json_data = None
            st.session_state.clean_dfs = None
            st.session_state.transformed_data = None
            st.session_state.visualization_updated = False
            st.rerun()

    # Call the selected page function
    pages[current_page]()


if __name__ == "__main__":
    main()
