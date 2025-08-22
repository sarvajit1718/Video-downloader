import os
import random
import time
from typing import Callable, Dict, Any, Optional
import yt_dlp


# ------------------------- Helpers -------------------------

def _human_wait(multiplier: float = 1.0) -> None:
    """Sleep a random short time to avoid hammering servers."""
    low, high = 0.2, 1.2
    time.sleep(random.uniform(low, high) * multiplier)


def _format_bytes(size_bytes: Optional[int]) -> str:
    """Convert bytes into human-readable format."""
    if not size_bytes:
        return "Unknown"
    size = float(size_bytes)
    units = ("B", "KB", "MB", "GB")
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _default_progress_hook(d: Dict[str, Any]) -> None:
    """Default download progress hook."""
    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
        done = d.get('downloaded_bytes', 0)
        pct = done / total * 100
        print(
            f"🛸 Downloading... {pct:.1f}% "
            f"({_format_bytes(done)}/{_format_bytes(total)})",
            end='\r'
        )
    elif d['status'] == 'finished':
        print(f"\n✅ Finished: {d.get('filename', 'Unknown file')}")


# ------------------------- Downloader -------------------------

class Downloader:
    def __init__(self, output_dir: str = "downloads"):
        self.video_dir = os.path.join(output_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def download_video(
        self,
        url: str,
        format_selector: str = "bv*+ba/best",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Download a video with yt-dlp (no metadata, no thumbnail).
        - format_selector: "bv*+ba/best" means bestvideo+audio, fallback to best.
        """
        progress_hooks = [progress_callback or _default_progress_hook]

        outtmpl = os.path.join(self.video_dir, "%(title)s.%(ext)s")

        ydl_opts = {
            "format": format_selector,
            "outtmpl": outtmpl,
            "progress_hooks": progress_hooks,
            "quiet": True,
            "noprogress": True,
            "merge_output_format": "mp4",  # ensure mp4 output
            "postprocessors": [],          # 🚫 no metadata / no thumbnails
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as e:
            print(f"❌ Download failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


# ------------------------- CLI Example -------------------------

if __name__ == "__main__":
    url = input("Enter YouTube URL: ").strip()
    downloader = Downloader()
    downloader.download_video(url)
