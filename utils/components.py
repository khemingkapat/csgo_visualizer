import streamlit as st
import json
from typing import Optional
import os
import tempfile
import requests
from awpy import Demo


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


# Configuration for Google Drive files
GOOGLE_DRIVE_FILES = {
    "vitality-vs-the-mongolz-m2-dust2": {
        "file_id": st.secrets["google_drive"]["m2"],  # Replace with actual file ID
        "filename": "vitality-vs-the-mongolz-m2-dust2.dem",
    },
    "vitality-vs-the-mongolz-m3-inferno": {
        "file_id": st.secrets["google_drive"]["m3"],  # Replace with actual file ID
        "filename": "vitality-vs-the-mongolz-m3-inferno.dem",
    },
}


def download_demo_file(file_id: str, local_path: str) -> bool:
    """
    Download a demo file from Google Drive.
    Fixed for Streamlit Cloud compatibility.
    """
    try:
        # Use temp directory instead of relative path for Streamlit Cloud
        if not os.path.isabs(local_path):
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            local_path = os.path.join(temp_dir, os.path.basename(local_path))

        # Ensure parent directory exists
        parent_dir = os.path.dirname(local_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Google Drive direct download URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # Show download progress
        with st.spinner(f"Downloading demo file from Google Drive..."):
            session = requests.Session()

            # Add headers to avoid bot detection
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = session.get(download_url, stream=True, headers=headers)

            # Handle Google Drive's virus scan warning for large files
            if (
                response.status_code == 200
                and "virus scan warning" in response.text.lower()
            ):
                # Look for the confirmation token
                import re

                confirm_pattern = r"confirm=([a-zA-Z0-9\-_]+)"
                match = re.search(confirm_pattern, response.text)
                if match:
                    confirm_token = match.group(1)
                    confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                    response = session.get(confirm_url, stream=True, headers=headers)

            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get("content-length", 0))

            with open(local_path, "wb") as f:
                if total_size > 0:
                    # Show progress bar
                    progress_bar = st.progress(0)
                    downloaded = 0

                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = min(downloaded / total_size, 1.0)
                            progress_bar.progress(progress)

                    progress_bar.empty()
                else:
                    # No content length, just download
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

        # Verify file was written and has content
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise Exception("Downloaded file is empty or doesn't exist")

        return True

    except Exception as e:
        st.error(f"❌ Failed to download demo file: {str(e)}")
        # Clean up partial file
        if os.path.exists(local_path):
            try:
                os.unlink(local_path)
            except:
                pass
        return False


def load_sample_demo_from_gdrive(
    demo_key: str, preview_limit: int = 10
) -> Optional[Demo]:
    """
    Load a sample CS demo file from Google Drive.
    Downloads the file if not cached locally, then loads it.

    Args:
        demo_key (str): Key to identify the demo in GOOGLE_DRIVE_FILES
        demo_name (str): Display name for the demo
        preview_limit (int): Number of items to preview in the UI

    Returns:
        awpy.Demo object if loaded successfully, else None
    """
    try:
        if demo_key not in GOOGLE_DRIVE_FILES:
            st.error(f"❌ Unknown demo key: {demo_key}")
            return None
        print("demo key found")

        demo_config = GOOGLE_DRIVE_FILES[demo_key]
        local_cache_path = f"sample_data/{demo_config['filename']}"

        # Check if file exists locally (cached)
        if not os.path.exists(local_cache_path):
            if not download_demo_file(demo_config["file_id"], local_cache_path):
                return None
            st.success(f"✅ demo downloaded and cached!")
        print("file downloaded")

        # Now load the file using existing logic
        with open(local_cache_path, "rb") as f:
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
        print("demo created")

        # Clean up temporary file
        os.unlink(temp_path)

        # Store in session state

        st.success("✅ Demo file loaded successfully!")

        # Show preview only if preview_limit > 0
        if preview_limit > 0:
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
