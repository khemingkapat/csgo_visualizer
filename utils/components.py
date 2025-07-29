import streamlit as st
import json
import struct
from typing import Dict, Any, Optional


def upload_and_parse_json(preview_limit=10) -> dict:
    """
    Streamlit widget to upload a JSON file and return the parsed content.

    Args:
        preview_limit (int): Number of items to preview in the UI.
                             If 0, no preview is shown.

    Returns:
        dict or list or None: Parsed JSON data if uploaded successfully, else None.
    """
    uploaded_file = st.file_uploader("Upload your JSON file", type="json")

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            st.success("✅ JSON file loaded successfully!")

            # Show preview only if preview_limit > 0
            if preview_limit > 0:
                if isinstance(data, dict):
                    preview = {
                        k: data[k]
                        for i, k in enumerate(data.keys())
                        if i < preview_limit
                    }
                elif isinstance(data, list):
                    preview = data[:preview_limit]
                else:
                    preview = str(data)

                st.write(f"📄 Preview (first {preview_limit} items or keys):")
                st.json(preview)

            return data

        except Exception as e:
            st.error(f"❌ Error loading JSON: {e}")
            return {}

    else:
        st.info("📂 Please upload a JSON file.")
        return {}


import streamlit as st
import tempfile
import os
from awpy import Demo
from typing import Optional


def upload_and_parse_demo(preview_limit: int = 10) -> Optional[Demo]:
    """
    Streamlit widget to upload a CS demo file (.dem) and return awpy Demo object.
    Args:
        preview_limit (int): Number of items to preview in the UI.
                             If 0, no preview is shown.
    Returns:
        awpy.Demo object if uploaded successfully, else None.
    """
    uploaded_file = st.file_uploader("Upload your CS Demo file", type="dem")

    if uploaded_file is not None:
        try:
            # Read the file data
            file_data = uploaded_file.read()

            # Basic validation
            if len(file_data) < 1072:
                st.error("❌ File too small to be a valid demo file")
                return None

            # Create temporary file for awpy
            with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name

            # Create Demo object
            demo = Demo(temp_path)

            # Clean up temporary file
            os.unlink(temp_path)

            st.success("✅ Demo file loaded successfully!")

            # Show preview only if preview_limit > 0
            if preview_limit > 0:
                st.write(f"📄 Demo file: {uploaded_file.name}")
                st.write(f"📊 Size: {len(file_data):,} bytes")

            return demo

        except Exception as e:
            st.error(f"❌ Error loading demo file: {e}")
            # Clean up temp file if it exists
            if "temp_path" in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return None
    else:
        st.info("📂 Please upload a demo file.")
        return None


def load_sample_demo(file_path: str, preview_limit: int = 10) -> Optional[Demo]:
    """
    Load a sample CS demo file (.dem) from a file path and return awpy Demo object.
    Follows the same logic as upload_and_parse_demo but reads from local file.

    Args:
        file_path (str): Path to the demo file
        demo_name (str): Display name for the demo
        preview_limit (int): Number of items to preview in the UI.
                             If 0, no preview is shown.
    Returns:
        awpy.Demo object if loaded successfully, else None.
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"❌ Sample file not found: {file_path}")
            return None

        # Read the file data
        with open(file_path, "rb") as f:
            file_data = f.read()

        # Basic validation (same as upload function)
        if len(file_data) < 1072:
            st.error("❌ File too small to be a valid demo file")
            return None

        # Create temporary file for awpy (same approach as upload function)
        with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as temp_file:
            temp_file.write(file_data)
            temp_path = temp_file.name

        # Create Demo object
        demo = Demo(temp_path)

        # Clean up temporary file
        os.unlink(temp_path)

        st.success("✅ Demo file loaded successfully!")

        # Show preview only if preview_limit > 0 (same as upload function)
        if preview_limit > 0:
            st.write(f"📊 Size: {len(file_data):,} bytes")

        return demo

    except Exception as e:
        st.error(f"❌ Error loading demo file: {e}")
        # Clean up temp file if it exists (same error handling as upload function)
        if "temp_path" in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return None
