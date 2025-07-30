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
    This handles virus scan warnings automatically.

    Args:
        file_id (str): Google Drive file ID
        filename (str): Original filename for reference

    Returns:
        str: Path to downloaded temporary file, or None if failed
    """
    try:
        st.info(f"🔄 Downloading {filename} using gdown...")

        # Validate file_id
        if not file_id or len(file_id) < 10:
            st.error(f"❌ Invalid Google Drive file ID: {file_id}")
            return None

        # Create temporary file with proper naming
        temp_fd, temp_path = tempfile.mkstemp(suffix=".dem", prefix="demo_")
        os.close(temp_fd)  # Close file descriptor but keep the file

        # Set proper permissions
        os.chmod(temp_path, 0o644)

        st.write(f"📁 Temp file: {temp_path}")
        st.write(f"🔗 File ID: {file_id}")

        # Construct Google Drive URL
        url = f"https://drive.google.com/uc?id={file_id}"
        st.write(f"📡 Download URL: {url}")

        # Use gdown to download with progress bar
        try:
            # Create a progress placeholder
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            # Configure gdown options
            # fuzzy=True helps with file ID extraction
            # quiet=False shows progress (we'll capture it)
            success = gdown.download(
                url=url,
                output=temp_path,
                quiet=False,  # Show progress
                fuzzy=True,  # Handle different URL formats
            )

            # Clear progress placeholders
            progress_placeholder.empty()
            status_placeholder.empty()

            if not success:
                st.error("❌ gdown reported download failure")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None

        except Exception as gdown_error:
            st.error(f"❌ gdown error: {gdown_error}")

            # Try alternative gdown method
            st.warning("🔄 Trying alternative gdown method...")
            try:
                # Sometimes the direct file ID works better
                gdown.download(id=file_id, output=temp_path, quiet=False)

            except Exception as alt_error:
                st.error(f"❌ Alternative method also failed: {alt_error}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None

        # Verify the download
        if not os.path.exists(temp_path):
            st.error("❌ Downloaded file doesn't exist")
            return None

        file_size = os.path.getsize(temp_path)
        st.write(f"📊 Downloaded size: {file_size:,} bytes")

        # Validate file size
        if file_size == 0:
            st.error("❌ Downloaded file is empty")
            os.unlink(temp_path)
            return None

        if file_size < 1072:
            st.error(f"❌ File too small ({file_size} bytes) - likely not a valid demo")
            os.unlink(temp_path)
            return None

        # Test file readability
        try:
            with open(temp_path, "rb") as test_file:
                header = test_file.read(32)
                st.write(f"🔍 File header: {header[:16].hex()}")

                # Basic demo file validation (optional)
                if header[:4] == b"HLVD":  # Half-Life/Source engine demo header
                    st.success("✅ Valid Source engine demo file detected")
                else:
                    st.warning(
                        "⚠️ Unusual file header - may not be a standard demo file"
                    )

        except Exception as read_error:
            st.error(f"❌ Cannot read downloaded file: {read_error}")
            os.unlink(temp_path)
            return None

        st.success(f"✅ Download successful: {file_size:,} bytes")
        return temp_path

    except ImportError:
        st.error("❌ gdown library not installed")
        st.error("💡 Add 'gdown>=4.6.0' to your requirements.txt file")
        return None

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None


def load_sample_demo_from_gdrive(
    demo_key: str, preview_limit: int = 10
) -> Optional[object]:
    """
    Downloads and parses a demo file from Google Drive using awpy.
    Includes comprehensive error handling and file validation.
    """
    try:
        st.write(f"🎯 Loading demo: {demo_key}")

        # Validate demo key
        if demo_key not in GOOGLE_DRIVE_FILES:
            st.error(f"❌ Unknown demo key: {demo_key}")
            st.error(f"Available keys: {list(GOOGLE_DRIVE_FILES.keys())}")
            return None

        demo_config = GOOGLE_DRIVE_FILES[demo_key]
        filename = demo_config["filename"]

        st.write(f"📋 Loading: {filename}")
        print(f"Loading demo: {filename}")

        # Check session cache first
        cache_key = f"demo_path_{demo_key}"
        downloaded_path = None

        if cache_key in st.session_state:
            cached_path = st.session_state[cache_key]
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 1072:
                st.success("✅ Using cached demo file")
                downloaded_path = cached_path
            else:
                st.warning("⚠️ Cache invalid, re-downloading...")
                del st.session_state[cache_key]

        # Download if not cached
        if not downloaded_path:
            downloaded_path = download_demo_file(demo_config["file_id"], filename)

            if not downloaded_path:
                st.error("❌ Download failed")
                return None

            # Cache the successful download
            st.session_state[cache_key] = downloaded_path
            st.success("✅ Demo downloaded successfully!")

        print(f"Downloaded path: {downloaded_path}")

        # Additional file validation before Demo creation
        if not os.path.exists(downloaded_path):
            st.error(f"❌ File doesn't exist: {downloaded_path}")
            return None

        file_size = os.path.getsize(downloaded_path)
        if file_size < 1072:
            st.error(f"❌ File too small: {file_size} bytes")
            return None

        st.write(f"📊 File size: {file_size:,} bytes")
        print(f"File size: {file_size} bytes")

        # Test file accessibility
        try:
            with open(downloaded_path, "rb") as test_file:
                header = test_file.read(32)
                st.write(f"🔍 File header: {header[:16].hex()}")
                print(f"File header: {header[:16].hex()}")
        except Exception as read_error:
            st.error(f"❌ Cannot read file: {read_error}")
            return None

        print("File validation passed, creating Demo object...")

        # Create Demo object with error handling
        try:
            # Import your Demo class here
            # from awpy import Demo

            st.info("🔄 Creating Demo object...")

            # For testing - replace this with your actual Demo import
            demo = Demo(downloaded_path)

            # Placeholder - replace with actual Demo creation

            print("Demo object created successfully")
            st.success("✅ Demo file parsed successfully!")

        except Exception as demo_error:
            st.error(f"❌ Error creating Demo object: {demo_error}")
            st.error(f"Error type: {type(demo_error).__name__}")

            # Show more details for debugging
            import traceback

            st.error("Full traceback:")
            st.code(traceback.format_exc())

            return None

        # Show preview information
        if preview_limit > 0:
            st.write(f"📄 Demo file: {filename}")
            st.write(f"📊 Size: {file_size:,} bytes")
            st.write(f"📁 Path: {downloaded_path}")

        return demo

    except Exception as e:
        st.error(f"❌ Error in load_sample_demo_from_gdrive: {e}")

        # Detailed error information
        import traceback

        st.error("Full error traceback:")
        st.code(traceback.format_exc())

        # Debug information
        st.error("🔍 Debug Information:")
        st.write(f"- Demo key: {demo_key}")
        st.write(f"- Available keys: {list(GOOGLE_DRIVE_FILES.keys())}")

        if demo_key in GOOGLE_DRIVE_FILES:
            config = GOOGLE_DRIVE_FILES[demo_key]
            st.write(f"- Filename: {config['filename']}")
            st.write(
                f"- File ID: {config['file_id'][:10]}..."
                if config["file_id"]
                else "None"
            )

        return None


# Utility functions
def clear_demo_cache():
    """Clear all cached demo files"""
    cleared = 0
    for key in list(st.session_state.keys()):
        if key.startswith("demo_path_"):
            file_path = st.session_state[key]
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                del st.session_state[key]
                cleared += 1
            except Exception as e:
                st.error(f"Error clearing {key}: {e}")

    st.success(f"✅ Cleared {cleared} cached files")


def test_file_download(demo_key: str):
    """Test just the download part without Demo creation"""
    if demo_key not in GOOGLE_DRIVE_FILES:
        st.error("Invalid demo key")
        return

    config = GOOGLE_DRIVE_FILES[demo_key]
    downloaded_path = download_demo_file(config["file_id"], config["filename"])

    if downloaded_path:
        st.success(f"✅ Download test successful: {downloaded_path}")
        st.write(f"File size: {os.path.getsize(downloaded_path):,} bytes")
    else:
        st.error("❌ Download test failed")
