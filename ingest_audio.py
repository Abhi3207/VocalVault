"""
Ingest audio files into the vector DB.

Usage:
    # Ingest all audio from downloaded_audio folder
    python ingest_audio.py

    # Ingest a specific file
    python ingest_audio.py "path/to/audio.mp3"

    # Ingest from a custom folder
    python ingest_audio.py --folder ./my_audio
"""

import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config
from rag.pipeline import get_pipeline


def ingest_folder(folder: Path):
    """Ingest all supported audio files from a folder."""
    extensions = config.audio.supported_extensions
    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not files:
        print(f"No audio files found in {folder}")
        print(f"Supported formats: {extensions}")
        return

    print(f"\nFound {len(files)} audio file(s) in {folder}:\n")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}  ({size_mb:.1f} MB)")

    pipeline = get_pipeline()

    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Ingesting: {f.name}")
        print("  Loading and chunking...")
        result = pipeline.ingest(f)
        print(f"  Done! {result['segments']} segments in {result['seconds']:.1f}s")

    total = pipeline.document_count()
    sources = pipeline.list_sources()
    print(f"\n{'='*50}")
    print(f"Knowledge base: {len(sources)} file(s), {total} segments")
    print(f"{'='*50}")


def ingest_file(path: Path):
    """Ingest a single audio file."""
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"\nIngesting: {path.name}")
    pipeline = get_pipeline()
    result = pipeline.ingest(path)
    print(f"Done! {result['segments']} segments in {result['seconds']:.1f}s")
    print(f"Knowledge base: {pipeline.document_count()} total segments")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest audio into vector DB")
    parser.add_argument("file", nargs="?", help="Specific audio file to ingest")
    parser.add_argument("--folder", default="./downloaded_audio",
                        help="Folder of audio files (default: ./downloaded_audio)")
    args = parser.parse_args()

    if args.file:
        ingest_file(Path(args.file))
    else:
        ingest_folder(Path(args.folder))
