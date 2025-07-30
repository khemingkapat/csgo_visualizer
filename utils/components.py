import streamlit as st
import json
from typing import Optional
import os
import tempfile
from awpy import Demo
import gdown


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


GOOGLE_DRIVE_FILES = {
    "vitality-vs-the-mongolz-m2-dust2": {
        "file_id": st.secrets["google_drive"]["m2"],
        "filename": "vitality-vs-the-mongolz-m2-dust2.dem",
    },
    "vitality-vs-the-mongolz-m3-inferno": {
        "file_id": st.secrets["google_drive"]["m3"],
        "filename": "vitality-vs-the-mongolz-m3-inferno.dem",
    },
}


def download_demo_file(file_id: str, filename: str) -> Optional[str]:
    """
    Download a demo file from Google Drive using gdown.

    Args:
        file_id (str): Google Drive file ID
        filename (str): Original filename for reference

    Returns:
        str: Path to downloaded temporary file, or None if failed
    """
    try:
        if not file_id or len(file_id) < 10:
            st.error(f"❌ Invalid Google Drive file ID")
            return None

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".dem", prefix="demo_")
        os.close(temp_fd)
        os.chmod(temp_path, 0o644)

        # Download using gdown
        url = f"https://drive.google.com/uc?id={file_id}"

        with st.spinner(f"Downloading {filename}..."):
            success = gdown.download(
                url=url,
                output=temp_path,
                quiet=True,
                fuzzy=True,
            )

            if not success:
                # Try alternative method
                try:
                    gdown.download(id=file_id, output=temp_path, quiet=True)
                except:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    return None

        # Validate download
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1072:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return None

        return temp_path

    except ImportError:
        st.error("❌ gdown library not installed")
        return None
    except Exception:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None


def load_sample_demo_from_gdrive(
    demo_key: str, preview_limit: int = 10
) -> Optional[object]:
    """
    Load a sample CS demo file from Google Drive using gdown.

    Args:
        demo_key (str): Key to identify the demo in GOOGLE_DRIVE_FILES
        preview_limit (int): Number of items to preview in the UI

    Returns:
        Demo object if loaded successfully, else None
    """
    try:
        # Validate demo key
        if demo_key not in GOOGLE_DRIVE_FILES:
            st.error(f"❌ Unknown demo key: {demo_key}")
            return None

        demo_config = GOOGLE_DRIVE_FILES[demo_key]
        filename = demo_config["filename"]

        # Check session cache first
        cache_key = f"demo_cache_{demo_key}"

        if cache_key in st.session_state:
            cached_path = st.session_state[cache_key]
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 1072:
                downloaded_path = cached_path
            else:
                del st.session_state[cache_key]
                downloaded_path = None
        else:
            downloaded_path = None

        # Download if not cached
        if not downloaded_path:
            downloaded_path = download_demo_file(demo_config["file_id"], filename)

            if not downloaded_path:
                st.error("❌ Failed to download demo file")
                return None

            # Cache the successful download
            st.session_state[cache_key] = downloaded_path

        # Create Demo object
        try:
            from awpy import Demo

            demo = Demo(downloaded_path)

            # Show preview information if requested
            if preview_limit > 0:
                file_size = os.path.getsize(downloaded_path)
                st.success("✅ Demo file loaded successfully!")
                st.write(f"📄 Demo file: {filename}")
                st.write(f"📊 Size: {file_size:,} bytes")

            return demo

        except ImportError:
            st.error("❌ Cannot import Demo class - check that 'awpy' is installed")
            return None
        except Exception as e:
            st.error(f"❌ Error creating Demo object: {e}")
            return None

    except Exception as e:
        st.error(f"❌ Error loading demo file: {e}")
        return None
