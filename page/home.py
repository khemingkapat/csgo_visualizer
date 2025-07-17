import streamlit as st
from parser import Parser
from utils.components import upload_and_parse_json
from preprocessor import Preprocessor
import json
from transformer import Transformer


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
        uploaded_data: dict = upload_and_parse_json(preview_limit=0)

    if uploaded_data is not None and uploaded_data:
        # Store data in session state
        st.session_state.json_data = uploaded_data

        # Process data immediately and store in session state
        with st.spinner("Processing data... This may take a moment."):
            # Parse and preprocess
            dfs = Parser.parse_json_to_dfs(uploaded_data)
            st.session_state.clean_dfs = Preprocessor.preprocess_single_match(dfs)

            with open(".awpy/maps/map-data.json", "r") as f:
                all_map_data = json.load(f)

            # Transform data
            st.session_state.transformed_data = Transformer.transform_all_data(
                st.session_state.clean_dfs, all_map_data
            )

        # Set page to overview
        st.session_state.page = "overview"
        st.success("File successfully uploaded! Redirecting to Overview page...")
        st.rerun()
