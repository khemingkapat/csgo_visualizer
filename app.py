import streamlit as st
from page import (
    home_page,
    overview_page,
    location_page,
    movement_page,
    action_page,
    economy_page,
)

# Configure the page
st.set_page_config(
    page_title="CS:GO Match Analysis Dashboard",
    page_icon="🎮",
    layout="wide",  # This makes the app use the full width of the screen
    initial_sidebar_state="expanded",
)


def main():
    # Check if we're on the home page (no data uploaded yet)
    if st.session_state.get("json_data") is None:
        # Show only the home page
        home_page()
    else:
        # Define pages for navigation (excluding home since we handle it separately)
        pages = {
            "Analysis": [
                st.Page(overview_page, title="Overview", icon="📊"),
                st.Page(location_page, title="Location", icon="🗺️"),
                st.Page(movement_page, title="Movement", icon="🔀"),
                st.Page(action_page, title="Action", icon="⚡"),
                st.Page(economy_page, title="Economy", icon="💰"),
            ]
        }
        # Create navigation
        page = st.navigation(pages)
        # Add sidebar info and controls
        st.sidebar.markdown("---")
        st.sidebar.info("CS:GO Match Analysis Tool")
        # Add button to return to upload page
        if st.sidebar.button("Upload New File", type="secondary"):
            # Clear session state and return to home
            st.session_state.page = "home"
            st.session_state.json_data = None
            st.session_state.clean_dfs = None
            st.session_state.transformed_data = None
            st.session_state.visualization_updated = False
            st.rerun()
        # Run the selected page
        page.run()


if __name__ == "__main__":
    main()
