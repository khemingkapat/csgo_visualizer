import streamlit as st
from parser import Parser
from utils.components import (
    # upload_and_parse_json,
    upload_and_parse_demo,
    load_sample_demo_from_gdrive,
)
from preprocessor import Preprocessor
from transformer import Transformer


def home_page():
    st.title("CS2 Match Analysis Dashboard")

    # Add feedback toggle button in the top right
    col1, col2 = st.columns([6, 1])
    with col2:
        # Replace with your actual Google Form link
        google_form_url = "https://docs.google.com/forms/d/e/1FAIpQLSc7ob7LP-9hDdtXwD9nEqlMct98-dzUZ_M9scnPromamt0hjg/viewform?usp=dialog"  # Replace this with your actual form URL

        # Initialize feedback toggle state
        if "show_feedback" not in st.session_state:
            st.session_state.show_feedback = False

        # Toggle button
        if st.button("💬 Feedback"):
            st.session_state.show_feedback = not st.session_state.show_feedback

    # Show/hide feedback section based on toggle state
    if st.session_state.show_feedback:
        with st.container():
            st.markdown("---")
            col1, col2, _ = st.columns([1, 2, 1])
            with col2:
                st.markdown(
                    f"""
                ### 📝 We'd love your feedback!
                
                Please click the link below to share your thoughts, report bugs, or suggest features:
                
                **[📝 Submit Feedback via Google Forms]({google_form_url})**
                
                Your feedback helps us improve the CS:GO Match Analysis Dashboard!
                """
                )

                # Close button
                if st.button("❌ Close", key="close_feedback"):
                    st.session_state.show_feedback = False
                    st.rerun()
            st.markdown("---")

    st.subheader("Upload Match Data")
    st.markdown(
        """
        Welcome to the CS2 Match Analysis Dashboard!
        
        This tool helps you analyze Counter-Strike 2 match data with detailed visualizations of player movements, kills, smokes, infernos, and grenades.
        
        To begin, you can either upload your own CS2 demo file, or start with our example demo
        
        
        🔧 This project is under development at: [github.com/khemingkapat/csgo_visualizer](https://github.com/khemingkapat/csgo_visualizer)
        """
    )

    # Center the upload button
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        # uploaded_data: dict = upload_and_parse_json(preview_limit=0)
        uploaded_demo = upload_and_parse_demo(preview_limit=0)

    # Provide option to use sample data
    st.markdown("---")
    st.subheader("🎮 Try with Our Sample Data")
    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        if st.button("Austin Major : Vitality vs. The Mongolz-Dust2"):
            uploaded_demo = load_sample_demo_from_gdrive(
                "vitality-vs-the-mongolz-m2-dust2"
            )

    with col2:
        if st.button("Austin Major : Vitality vs. The Mongolz-Inferno"):
            uploaded_demo = load_sample_demo_from_gdrive(
                "vitality-vs-the-mongolz-m3-inferno"
            )

    if uploaded_demo is not None and uploaded_demo:
        st.session_state.demo_data = uploaded_demo
        with st.spinner("Processing data... This may take a moment."):
            st.session_state.parsed_demo = Parser.parse_demo_to_dfs(uploaded_demo)
            st.session_state.clean_dfs = Preprocessor.preprocess_single_match(
                st.session_state.parsed_demo
            )
            st.session_state.transformed_data = Transformer.transform_all_data(
                st.session_state.clean_dfs
            )
        st.session_state.page = "overview"
        st.success("File successfully loaded! Redirecting to Overview page...")
        st.rerun()

    # Footer with additional links and info
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <p>
                📧 <a href='{google_form_url}' target='_blank'>Send Feedback</a> | 
                🐛 <a href='{google_form_url}' target='_blank'>Report Bug</a> | 
                ⭐ <a href='https://github.com/khemingkapat/csgo_visualizer' target='_blank'>Star us on GitHub</a> | 
                📖 <a href='https://github.com/pnxenopoulos/awpy' target='_blank'>CS2 Demo Parser Documentation</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
