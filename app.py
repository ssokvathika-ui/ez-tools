import streamlit as st
import subprocess
import os
import shutil
import json
import tempfile
import logging
from datetime import datetime
from typing import Dict, List
import sqlite3
import hashlib
import re
import platform
import requests
from urllib.parse import urlparse
import traceback
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def get_exe_directory():
    """Get the directory where the .exe file is located"""
    if getattr(sys, 'frozen', False):
        # If running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # If running as script
        return os.path.dirname(os.path.abspath(__file__))

# Modify your Config class to use exe directory
class Config:
    # Get the directory where the exe is located
    EXE_DIR = get_exe_directory()
    OUTPUT_DIR = os.path.join(EXE_DIR, "silent_echo_output")
    FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"
    YT_DLP_PATH = shutil.which("yt-dlp") or "yt-dlp"
    MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB limit
    CACHE_TTL = 3600  # 1 hour

    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "cache"), exist_ok=True)

        

# Initialize configuration
Config.setup_directories()

# Database Setup
def init_database():
    """Initialize SQLite database for user history with proper schema"""
    conn = sqlite3.connect(os.path.join(Config.OUTPUT_DIR, "user_history.db"))
    cursor = conn.cursor()

    # Create table with title column
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS download_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        url TEXT,
        audio_format TEXT,
        video_format TEXT,
        filename TEXT,
        title TEXT
    )
    ''')

    # Check if title column exists (for backward compatibility)
    try:
        cursor.execute("PRAGMA table_info(download_history)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'title' not in columns:
            # Add title column if it doesn't exist
            cursor.execute("ALTER TABLE download_history ADD COLUMN title TEXT")
    except Exception as e:
        logger.error(f"Database migration error: {e}")

    conn.commit()
    conn.close()

# Initialize database
init_database()

def is_valid_youtube_url(url: str) -> bool:
    """Validate YouTube URL format"""
    pattern = r'^https?://(www\.)?(youtube\.com|youtu\.be)/.+'
    return re.match(pattern, url) is not None

class YouTubeDownloader:
    """Enhanced YouTube downloader with caching and error handling"""

    def __init__(self, yt_dlp_path: str, ffmpeg_path: str):
        self.yt_dlp_path = yt_dlp_path
        self.ffmpeg_path = ffmpeg_path

    def get_video_info(self, url: str) -> Dict:
        """Get video information with caching and safe subprocess call"""
        if not url or not is_valid_youtube_url(url):
            raise ValueError(f"Invalid URL: {url}")

        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(Config.OUTPUT_DIR, "cache", f"{cache_key}.json")

        # Check cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        # Fetch fresh data
        try:
            cmd = [
                self.yt_dlp_path,
                "-j",
                url,
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0 AppleWebKit/605.1.15",
                "--no-cache-dir"
            ]

            logger.info(f"Executing: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False  # ← Critical: prevents shell injection & misparsing
            )

            stdout, stderr = process.communicate(timeout=30)

            if process.returncode != 0:
                error = stderr.decode('utf-8', errors='ignore').strip()
                raise Exception(f"yt-dlp failed: {error}")

            info = json.loads(stdout.decode('utf-8', errors='ignore'))

            # Save to cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            return info
        except Exception as e:
            logger.error(f"Error fetching video info: {e}\n{traceback.format_exc()}")
            raise

    def get_formats(self, url: str) -> Dict[str, List[Dict]]:
        """Get audio and video formats"""
        try:
            info = self.get_video_info(url)
            formats = info.get("formats", [])

            audio_only = [f for f in formats if f.get("vcodec") == "none"]
            video_only = [f for f in formats if f.get("acodec") == "none"]

            return {
                "audio_only": audio_only,
                "video_only": video_only,
                "title": info.get("title", "unknown_title")
            }
        except Exception as e:
            logger.error(f"Error getting formats: {e}\n{traceback.format_exc()}")
            raise

    def get_fresh_direct_url(self, url: str, format_id: str) -> str:
        """Get fresh direct download URL for a specific format (bypasses caching issues)"""
        try:
            # Force fresh URL by adding timestamp to avoid cached URLs
            timestamp = str(int(time.time()))

            # Use Popen with binary mode to avoid encoding issues
            process = subprocess.Popen([
                self.yt_dlp_path,
                "-f", format_id,
                "--get-url",
                "--add-header", "Referer:https://www.youtube.com/",
                "--add-header", "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "--no-cache-dir",  # Don't use cached data
                f"--add-header", "Cache-Control:no-cache",
                url
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)

            stdout, stderr = process.communicate(timeout=30)

            if process.returncode != 0:
                raise Exception(f"yt-dlp failed: {stderr.decode('utf-8', errors='ignore')}")

            direct_url = stdout.decode('utf-8', errors='ignore').strip()

            # Validate URL
            if not direct_url.startswith("http"):
                raise Exception("Invalid URL received from yt-dlp")

            return direct_url
        except Exception as e:
            logger.error(f"Error getting fresh direct URL: {e}\n{traceback.format_exc()}")
            raise

    def sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')
        # Limit length to 100 characters
        if len(filename) > 100:
            filename = filename[:100]
        return filename

    def download_stream_with_progress(self, url: str, output_path: str, progress_text) -> bool:
        """Download a stream directly with progress indication"""
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_text.text(f"📥 Downloading... {progress:.1f}%")

            return True
        except Exception as e:
            logger.error(f"Error downloading stream: {e}\n{traceback.format_exc()}")
            return False

    def merge_streams_direct_fast(self, video_url: str, audio_url: str, title: str, original_youtube_url: str) -> tuple:
        """Fast merge using FFmpeg to download and merge in one step"""
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            # reivse 
            st.session_state["temp_dirs"] = st.session_state.get("temp_dirs", []) + [temp_dir]
            # Sanitize the title for filename use
            safe_title = self.sanitize_filename(title)
            base_name = f"{safe_title}_sanctuary"
            output_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}.webm")

            # Handle duplicate filenames
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}_{counter}.webm")
                counter += 1

            # Get FRESH direct URLs to avoid 403 errors
            st.info("🔄 Getting fresh stream URLs...")
            fresh_video_url = self.get_fresh_direct_url(original_youtube_url,
                                                        st.session_state.get("video_id", ""))
            fresh_audio_url = self.get_fresh_direct_url(original_youtube_url,
                                                        st.session_state.get("audio_id", ""))

            # Use FFmpeg to download and merge in one step (much faster!)
            st.info("🔄 Streaming merge with FFmpeg (faster)...")

            # Build command with proper URL handling
            cmd = [
                self.ffmpeg_path,
                "-i", fresh_video_url,
                "-i", fresh_audio_url,
                "-c", "copy",
                "-f", "webm",
                "-y",
                output_path
            ]

            # Run FFmpeg with Popen to avoid threading issues
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=600)  # 10 minutes timeout
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise Exception("FFmpeg timed out after 10 minutes")

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown FFmpeg error"
                logger.error(f"FFmpeg fast merge failed: {error_msg}")

                # Check for specific 403 errors
                if "403" in error_msg or "Forbidden" in error_msg:
                    return False, output_path, "YouTube blocked access (403 error) - trying alternative method"

                # Return failure with error details
                return False, output_path, error_msg

            return True, output_path, "Success"

        except Exception as e:
            logger.error(f"Error in fast merge: {e}\n{traceback.format_exc()}")
            return False, "", str(e)
        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.error(f"Failed to cleanup temp directory {temp_dir}: {e}")

    def merge_streams_direct(self, video_url: str, audio_url: str, title: str, original_youtube_url: str) -> str:
        """Merge video and audio streams using direct URLs and FFmpeg"""
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            #revise
            st.session_state["temp_dirs"] = st.session_state.get("temp_dirs", []) + [temp_dir]

            # Sanitize the title for filename use
            safe_title = self.sanitize_filename(title)
            base_name = f"{safe_title}_sanctuary"

            # Define temp file paths
            temp_video = os.path.join(temp_dir, "temp_video.webm")
            temp_audio = os.path.join(temp_dir, "temp_audio.webm")
            output_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}.webm")

            # Handle duplicate filenames
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(Config.OUTPUT_DIR, f"{base_name}_{counter}.webm")
                counter += 1

            # Get FRESH direct URLs to avoid 403 errors
            st.info("🔄 Getting fresh stream URLs for download...")
            fresh_video_url = self.get_fresh_direct_url(original_youtube_url,
                                                        st.session_state.get("video_id", ""))
            fresh_audio_url = self.get_fresh_direct_url(original_youtube_url,
                                                        st.session_state.get("audio_id", ""))

            # Download streams with progress
            progress_text = st.empty()
            progress_text.text("📥 Downloading video stream...")

            if not self.download_stream_with_progress(fresh_video_url, temp_video, progress_text):
                raise Exception("Failed to download video stream")

            progress_text.text("📥 Downloading audio stream...")
            if not self.download_stream_with_progress(fresh_audio_url, temp_audio, progress_text):
                raise Exception("Failed to download audio stream")

            # Merge using FFmpeg
            progress_text.text("🔄 Merging streams with FFmpeg...")

            cmd = [
                self.ffmpeg_path,
                "-i", temp_video,
                "-i", temp_audio,
                "-c", "copy",
                "-f", "webm",
                "-y",
                output_path
            ]

            # Run FFmpeg with Popen to avoid threading issues
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False
            )

            # Wait for completion
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown FFmpeg error"
                logger.error(f"FFmpeg merge failed: {error_msg}")
                raise Exception(f"FFmpeg merge failed: {error_msg}")

            return output_path

        except Exception as e:
            logger.error(f"Error merging streams: {e}\n{traceback.format_exc()}")
            raise
        finally:
            # Cleanup temp directory and all downloaded files
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory and files: {temp_dir}")
                except Exception as e:
                    logger.error(f"Failed to cleanup temp directory {temp_dir}: {e}")

# Initialize downloader with fallbacks
try:
    downloader = YouTubeDownloader(Config.YT_DLP_PATH, Config.FFMPEG_PATH)
except Exception as e:
    st.error(f"Failed to initialize downloader: {e}")
    st.stop()

# UI Components
def show_header():
    """Display application header"""
    st.title("🧘‍♂️ សម្លេងស្ងប់ស្ងាត់ — Silent Echo Studio")
    st.markdown("""
    Whispers your link. Let the streams reveal themselves.
    Choose your sanctuary for meditative content creation.

    🌿 *Crafted with care for peaceful creators*
    """)

def show_output_directories():
    """Display current output directories"""
    st.subheader("📁 Output Directories")
    st.markdown(f"**Main Output Directory:** `{Config.OUTPUT_DIR}`")
    cache_dir = os.path.join(Config.OUTPUT_DIR, "cache")
    st.markdown(f"**Cache Directory:** `{cache_dir}`")
    
    # List files in output directory
    if os.path.exists(Config.OUTPUT_DIR):
        files = os.listdir(Config.OUTPUT_DIR)
        if files:
            st.markdown("**Files in Output Directory:**")
            for file in files:
                file_path = os.path.join(Config.OUTPUT_DIR, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    st.markdown(f"- `{file}` ({file_size} bytes)")
                else:
                    st.markdown(f"- `{file}/` (directory)")


def show_input_section():
    """Display input section"""
    video_url = st.text_input("📥 Paste your YouTube link here:", key="video_url")

    if video_url:
        if not is_valid_youtube_url(video_url):
            st.error("⚠️ Invalid YouTube URL.")
            return

        if st.button("🔍 Reveal Stream Spirits", key="reveal_button"):
            with st.spinner("🕊️ Gathering stream spirits..."):
                try:
                    formats = downloader.get_formats(video_url)

                    # Store all data in session state
                    st.session_state["audio_only"] = formats["audio_only"]
                    st.session_state["video_only"] = formats["video_only"]
                    st.session_state["formats_revealed"] = True
                    st.session_state["current_url"] = video_url
                    st.session_state["video_title"] = formats["title"]

                    st.success("🌿 Stream spirits revealed!")

                except Exception as e:
                    error_msg = str(e)
                    if "403" in error_msg or "Forbidden" in error_msg:
                        st.error("⚠️ YouTube is blocking this request. Try again later or use a different video.")
                    else:
                        st.error(f"⚠️ Technical error: Could not extract formats.\n{error_msg}")

def show_format_selection():
    """Display format selection interface"""
    if not st.session_state.get("formats_revealed", False):
        return

    video_url = st.session_state.get("current_url", "")
    title = st.session_state.get("video_title", "Unknown Title")

    st.subheader("🎵 Audio Streams")
    selected_audio = st.selectbox(
        "Choose audio stream:",
        st.session_state["audio_only"],
        format_func=lambda f: f"{f['format_id']} — {f['ext']} — {f.get('abr', 'N/A')}kbps",
        key="audio_select"
    )

    st.subheader("🎬 Video Streams")
    selected_video = st.selectbox(
        "Choose video stream:",
        st.session_state["video_only"],
        format_func=lambda f: f"{f['format_id']} — {f['ext']} — {f.get('height', 'N/A')}p",
        key="video_select"
    )

    # Display current title
    st.markdown(f"📖 **Current Video Title:** {title}")

    # Store selections
    st.session_state["audio_id"] = selected_audio["format_id"]
    st.session_state["video_id"] = selected_video["format_id"]

    # Generate and display direct download links
    if "audio_id" in st.session_state and "video_id" in st.session_state:
        with st.spinner("🔄 Generating download links..."):
            try:
                audio_url = downloader.get_fresh_direct_url(video_url, st.session_state["audio_id"])
                video_url_dl = downloader.get_fresh_direct_url(video_url, st.session_state["video_id"])

                st.session_state["direct_audio_url"] = audio_url
                st.session_state["direct_video_url"] = video_url_dl

                st.markdown("🔗 **Direct Download Links:**")
                st.markdown(f"🎧 [Download Audio Stream]({audio_url})")
                st.markdown(f"🎬 [Download Video Stream]({video_url_dl})")

            except Exception as e:
                st.error(f"⚠️ Failed to generate download links: {str(e)}")

        st.markdown("💫 Choose your streams. Let the merge begin when your heart is ready.")

def show_merge_section():
    """Display merge section"""
    if "audio_id" not in st.session_state or "video_id" not in st.session_state:
        return

    title = st.session_state.get("video_title", "Unknown Title")
    original_url = st.session_state.get("current_url", "")

    # Add option for fast merge
    use_fast_merge = st.checkbox("⚡ Use Fast Merge (recommended)", value=True)

    if st.button("🌺 Merge into Sanctuary", key="merge_button"):
        with st.spinner("🎧 Merging with silent clarity..."):
            try:
                if use_fast_merge:
                    # Use fast merge (streaming) - NO temporary files created
                    success, merged_path, error_msg = downloader.merge_streams_direct_fast(
                        st.session_state["direct_video_url"],
                        st.session_state["direct_audio_url"],
                        title,
                        original_url
                    )

                    if not success:
                        st.warning(f"⚡ Fast merge failed: {error_msg}")
                        if "403" in error_msg or "Forbidden" in error_msg:
                            st.info("🔄 YouTube blocked streaming access. Trying download method...")
                        else:
                            st.info("🔄 Trying alternative download-then-merge method...")

                            # Fallback to download-then-merge approach
                            merged_path = downloader.merge_streams_direct(
                                st.session_state["direct_video_url"],
                                st.session_state["direct_audio_url"],
                                title,
                                original_url
                            )
                else:
                    # Use download-then-merge approach
                    merged_path = downloader.merge_streams_direct(
                        st.session_state["direct_video_url"],
                        st.session_state["direct_audio_url"],
                        title,
                        original_url
                    )

                # Save to history
                conn = sqlite3.connect(os.path.join(Config.OUTPUT_DIR, "user_history.db"))
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO download_history (timestamp, url, audio_format, video_format, filename, title)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                    datetime.now().isoformat(),
                    st.session_state["current_url"],
                    st.session_state["audio_id"],
                    st.session_state["video_id"],
                    os.path.basename(merged_path),
                    title
                ))
                conn.commit()
                conn.close()

                
                with open(merged_path, "rb") as f:
                    st.success("🌕 Sanctuary complete!")

                    # For desktop apps, we will show the file path and provide a direct download
                    file_name = os.path.basename(merged_path)
                    st.markdown(f"📁 **File saved to:** `{file_name}`")
                    st.markdown(f"📍 **File Location:** `{merged_path}`")

                    # Create a direct download link that works in the desktop apps
                    with open(merged_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Download Sanctuary (.webm)",
                            data=file,
                            file_name=os.path.basename(merged_path),
                            mime="video/webm",
                            key="download_button"
                        )
                    
                    # Show the output filename
                    st.markdown(f"🪷 *Let this merged stream be a vessel of peace. You've woven sound and vision into stillness.*")
                    st.markdown(f"📝 **Output filename:** `{os.path.basename(merged_path)}`")

            except Exception as e:
                error_msg = str(e)
                if "FFmpeg not found" in error_msg:
                    st.error("⚠️ FFmpeg is required but not available in this environment.")
                    st.info("This issue occurs because Streamlit Cloud doesn't have FFmpeg pre-installed.")
                    st.markdown("💡 **Workaround:** Use direct download links instead of merging.")
                elif "403" in error_msg or "Forbidden" in error_msg:
                    st.error("⚠️ YouTube is blocking this request. Try again later.")
                else:
                    st.error(f"⚠️ Merge failed: {error_msg}")
                    st.markdown("🌧️ Even the quietest rituals may falter. Try again with care.")

def show_history():
    """Display user download history"""
    try:
        conn = sqlite3.connect(os.path.join(Config.OUTPUT_DIR, "user_history.db"))
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, url, audio_format, video_format, filename, title
            FROM download_history
            ORDER BY timestamp DESC
            LIMIT 10
            ''')
        history = cursor.fetchall()
        conn.close()

        if history:
            st.subheader("📜 Recent Downloads")
            for row in history:
                st.markdown(f"""
                    - **{datetime.fromisoformat(row[0]).strftime('%Y-%m-%d %H:%M')}**
                    Title: `{row[5][:50]}...`
                    File: `{row[4]}`
                    Audio: `{row[2]}` | Video: `{row[3]}`
                    """)
        else:
            st.info("No download history yet.")
    except Exception as e:
        logger.error(f"Error loading history: {e}\n{traceback.format_exc()}")
        st.error(f"Error loading history: {e}")

def show_footer():
    """Display footer"""
    st.markdown("---")
    st.markdown("""
        🌙 **Production-Ready Silent Echo Studio**
        Crafted with care for meditative creators. Let each merge be a moment of peace.

        🔧 Features:
        - Caching for faster format discovery
        - Direct download links
        - Download history tracking
        - Title-based output naming
        - Error handling and logging
        - Secure temporary file management
        - Fresh URL generation to avoid YouTube blocks
        - Fast streaming merge option with automatic fallback
        """)

# Main Application Flow
def main():
    """Main application function"""
    show_header()

    # Add a clear way to reset session
    if st.button("🔄 Start New Session"):
        st.session_state.clear()
        st.rerun()

    if st.button("🗑️ Clear All Temporary Files"):
        dirs = st.session_state.get("temp_dirs", [])

        # search for additional directories with prefix "silent_echo_"
        temp_root = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp")
         # adjust this path to where your temp dirs are created
        for entry in os.listdir(temp_root):
            full_path = os.path.join(temp_root, entry)
            if entry.startswith("silent_echo_"):
                if os.path.isdir(full_path) and full_path not in dirs:
                    dirs.append(full_path)
        # clean up all collected directories
        for d in dirs:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                    logger.info(f"Manually cleaned up temp dir: {d}")
                except Exception as e:
                    logger.error(f"Failed to remove temp dir {d}: {e}")
        st.session_state["temp_dirs"] = []
        st.success("🧹 All temporary files cleared.")
    # Display output directory
    # show_output_directories() 
    # Your clean flow - exactly right!
    show_input_section()
    show_format_selection()
    show_merge_section()
    show_history()
    show_footer()

if __name__ == "__main__":
    main()
