import streamlit as st
import json
from typing import Optional
import os
import tempfile
import requests
from awpy import Demo
import re


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


def download_demo_file(file_id: str, local_path: str) -> Optional[str]:
    """
    Downloads a demo file from Google Drive and returns the absolute local path.
    Works reliably on Streamlit Cloud by using a temp directory and handling Google Drive confirmation tokens.
    """
    try:
        # Use temp directory if relative path
        if not os.path.isabs(local_path):
            temp_dir = tempfile.mkdtemp()
            local_path = os.path.join(temp_dir, os.path.basename(local_path))

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Step 1: Try direct download
        session = requests.Session()
        base_url = "https://drive.google.com/uc?export=download"
        download_url = f"{base_url}&id={file_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        with st.spinner("Downloading demo file from Google Drive..."):
            response = session.get(download_url, stream=True, headers=headers)

            # Step 2: If we hit a warning page, extract the confirmation token
            def get_confirm_token(text):
                match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
                return match.group(1) if match else None

            token = get_confirm_token(response.text)
            if token:
                confirm_url = f"{base_url}&id={file_id}&confirm={token}"
                response = session.get(confirm_url, stream=True, headers=headers)

            response.raise_for_status()

            # Step 3: Write the file to disk
            total_size = int(response.headers.get("content-length", 0))
            with open(local_path, "wb") as f:
                if total_size > 0:
                    progress_bar = st.progress(0)
                    downloaded = 0
                    for chunk in response.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress_bar.progress(min(downloaded / total_size, 1.0))
                    progress_bar.empty()
                else:
                    for chunk in response.iter_content(8192):
                        if chunk:
                            f.write(chunk)

        # Step 4: Validate downloaded file
        if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024:
            raise Exception("Downloaded file is too small or missing")

        # Optional: save copy for debugging (remove in prod)
        # shutil.copy(local_path, "/tmp/debug_downloaded.dem")

        return local_path

    except Exception as e:
        st.error(f"❌ Failed to download demo file: {e}")
        try:
            if os.path.exists(local_path):
                os.unlink(local_path)
        except:
            pass
        return None


def load_sample_demo_from_gdrive(
    demo_key: str, preview_limit: int = 10
) -> Optional[Demo]:
    """
    Downloads and parses a demo file from Google Drive using awpy.
    Only creates a temp file for preview (not required for parsing).
    """
    try:
        if demo_key not in GOOGLE_DRIVE_FILES:
            st.error(f"❌ Unknown demo key: {demo_key}")
            return None

        demo_config = GOOGLE_DRIVE_FILES[demo_key]
        filename = demo_config["filename"]
        print(filename)

        # Step 1: Download the file
        downloaded_path = download_demo_file(demo_config["file_id"], filename)
        print(downloaded_path)
        if not downloaded_path:
            return None

        st.success("✅ Demo downloaded successfully!")

        # Step 2: (Optional) File validation
        if os.path.getsize(downloaded_path) < 1072:
            st.error("❌ File too small to be a valid demo file")
            return None
        print("file big enough")

        # Step 3: Parse directly using awpy (or your Demo class)
        demo = Demo(downloaded_path)  # ✅ Direct path — no extra temp file
        print("demo created")

        st.success("✅ Demo file parsed successfully!")

        if preview_limit > 0:
            st.write(f"📄 Demo file: {filename}")
            st.write(f"📊 Size: {os.path.getsize(downloaded_path):,} bytes")

        return demo

    except Exception as e:
        st.error(f"❌ Error loading demo file: {e}")
        return None
