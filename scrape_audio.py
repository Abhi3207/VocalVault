"""
YouTube Audio Scraper
=====================
Downloads audio from YouTube videos using yt-dlp.
Supports single videos, playlists, and batch URL lists.

Usage:
    # Single video
    python scrape_audio.py "https://www.youtube.com/watch?v=VIDEO_ID"

    # Multiple videos
    python scrape_audio.py "URL1" "URL2" "URL3"

    # Playlist
    python scrape_audio.py "https://www.youtube.com/playlist?list=PLAYLIST_ID"

    # Custom output format (wav instead of mp3)
    python scrape_audio.py --format wav "URL"

    # Custom output directory
    python scrape_audio.py --output ./my_audio "URL"
"""

import argparse
import os
import sys
from pathlib import Path

import yt_dlp


def download_audio(
    urls: list[str],
    output_dir: str = "./downloaded_audio",
    audio_format: str = "mp3",
    audio_quality: str = "192",  # kbps
    verbose: bool = False,
):
    """
    Download audio from YouTube videos.

    Args:
        urls: List of YouTube URLs (videos or playlists).
        output_dir: Directory to save downloaded audio files.
        audio_format: Output audio format (mp3, wav, flac, aac, opus, m4a).
        audio_quality: Audio quality in kbps (for lossy formats).
        verbose: Show detailed download progress.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # FFmpeg location (installed via winget)
    ffmpeg_path = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
    )

    # yt-dlp configuration
    ydl_opts = {
        # FFmpeg path
        "ffmpeg_location": ffmpeg_path,
        # Extract audio only
        "format": "bestaudio/best",
        # Post-processing: convert to desired format
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": audio_quality,
            }
        ],
        # Output template: saves as "Title - Channel.ext"
        "outtmpl": str(output_path / "%(title)s - %(channel)s.%(ext)s"),
        # Avoid re-downloading
        "download_archive": str(output_path / ".download_archive.txt"),
        # Metadata
        "addmetadata": True,
        # Quiet mode unless verbose
        "quiet": not verbose,
        "no_warnings": not verbose,
        # Progress hooks
        "progress_hooks": [_progress_hook],
        # Restrict filenames to avoid OS issues
        "restrictfilenames": False,
        "windowsfilenames": True,
    }

    print(f"\n{'='*60}")
    print(f"  YouTube Audio Scraper")
    print(f"  Format: {audio_format.upper()} @ {audio_quality}kbps")
    print(f"  Output: {output_path.resolve()}")
    print(f"{'='*60}\n")

    successful = 0
    failed = 0

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                print(f"[*] Processing: {url}")
                # Extract info first to show what we're downloading
                info = ydl.extract_info(url, download=False)

                if info.get("_type") == "playlist":
                    count = len(info.get("entries", []))
                    print(f"    Playlist: {info.get('title', 'Unknown')} ({count} videos)")
                else:
                    print(f"    Title: {info.get('title', 'Unknown')}")
                    duration = info.get("duration", 0)
                    if duration:
                        mins, secs = divmod(duration, 60)
                        print(f"    Duration: {int(mins)}:{int(secs):02d}")

                # Now download
                ydl.download([url])
                successful += 1
                print(f"    ✓ Done!\n")

            except yt_dlp.utils.DownloadError as e:
                failed += 1
                print(f"    ✗ Failed: {e}\n")
            except Exception as e:
                failed += 1
                print(f"    ✗ Unexpected error: {e}\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"  Successful: {successful}")
    print(f"  Failed:     {failed}")
    print(f"  Files saved to: {output_path.resolve()}")
    print(f"{'='*60}\n")

    return successful, failed


def _progress_hook(d):
    """Show download progress."""
    if d["status"] == "downloading":
        percent = d.get("_percent_str", "N/A")
        speed = d.get("_speed_str", "N/A")
        eta = d.get("_eta_str", "N/A")
        print(f"\r    Downloading: {percent} | Speed: {speed} | ETA: {eta}  ", end="", flush=True)
    elif d["status"] == "finished":
        print(f"\r    Download complete, converting...                          ")


def list_available_formats(url: str):
    """List all available audio/video formats for a URL."""
    ydl_opts = {"listformats": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=False)


def get_video_info(url: str) -> dict:
    """Get metadata for a video without downloading."""
    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title"),
            "channel": info.get("channel"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
            "upload_date": info.get("upload_date"),
            "description": info.get("description", "")[:200],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --format wav --output ./audio "URL1" "URL2"
  %(prog)s --format flac --quality 320 "PLAYLIST_URL"
  %(prog)s --info "URL"
        """,
    )
    parser.add_argument("urls", nargs="*", help="YouTube video or playlist URLs")
    parser.add_argument(
        "--format", "-f",
        default="mp3",
        choices=["mp3", "wav", "flac", "aac", "opus", "m4a"],
        help="Output audio format (default: mp3)",
    )
    parser.add_argument(
        "--quality", "-q",
        default="192",
        help="Audio quality in kbps (default: 192)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./downloaded_audio",
        help="Output directory (default: ./downloaded_audio)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Show video info without downloading",
    )
    parser.add_argument(
        "--list-formats", "-l",
        action="store_true",
        help="List available formats for a URL",
    )

    args = parser.parse_args()

    if not args.urls:
        parser.print_help()
        sys.exit(1)

    # Info mode
    if args.info:
        for url in args.urls:
            info = get_video_info(url)
            print(f"\nTitle:    {info['title']}")
            print(f"Channel:  {info['channel']}")
            if info["duration"]:
                m, s = divmod(info["duration"], 60)
                print(f"Duration: {int(m)}:{int(s):02d}")
            print(f"Views:    {info['view_count']:,}")
            print(f"Uploaded: {info['upload_date']}")
            print(f"Desc:     {info['description']}...")
        sys.exit(0)

    # List formats mode
    if args.list_formats:
        for url in args.urls:
            list_available_formats(url)
        sys.exit(0)

    # Download mode
    download_audio(
        urls=args.urls,
        output_dir=args.output,
        audio_format=args.format,
        audio_quality=args.quality,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
