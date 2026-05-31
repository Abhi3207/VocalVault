"""
Query the audio knowledge base.

Usage:
    # Query with an audio file
    python query_audio.py "path/to/query.wav"

    # Query with more results
    python query_audio.py --top-k 10 "path/to/query.wav"

    # Record from microphone (5 seconds)
    python query_audio.py --record 5
"""

import sys
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config
from rag.pipeline import get_pipeline


def query_with_file(audio_path: str, top_k: int = 5):
    """Query the knowledge base with an audio file."""
    path = Path(audio_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    pipeline = get_pipeline()
    count = pipeline.document_count()

    if count == 0:
        print("Knowledge base is empty. Run ingest_audio.py first.")
        return

    print(f"\nKnowledge base: {count} segments")
    print(f"Query file: {path.name}")
    print(f"Searching top {top_k} results...\n")

    results = pipeline.query(str(path), top_k=top_k)

    if not results:
        print("No matching segments found.")
        return

    print(f"Found {len(results)} matching segment(s):\n")
    print(f"{'Rank':<6} {'Source File':<40} {'Time Range':<18} {'Distance':<10} {'Snippet'}")
    print("-" * 120)

    for r in results:
        time_range = f"{r.start_time:.1f}s - {r.end_time:.1f}s"
        snippet = r.snippet_path.name if r.snippet_path else "N/A"
        source = r.source_file[:38] if len(r.source_file) > 38 else r.source_file
        print(f"#{r.rank:<5} {source:<40} {time_range:<18} {r.distance:<10.4f} {snippet}")

    print(f"\nSnippets saved to: {config.paths.snippets_dir}")
    print("You can play any snippet file with your audio player.")


def record_and_query(duration: int = 5, top_k: int = 5):
    """Record from microphone and query."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("Install sounddevice for mic recording: pip install sounddevice")
        print("\nAlternatively, record audio with any app and pass the file:")
        print('  python query_audio.py "path/to/recording.wav"')
        return

    sr = config.audio.sample_rate
    print(f"\nRecording {duration} seconds from microphone...")
    print("Speak now!")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()

    query_path = config.paths.data_dir / "mic_query.wav"
    sf.write(str(query_path), audio, sr)
    print(f"Saved recording to {query_path}\n")

    query_with_file(str(query_path), top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the audio knowledge base")
    parser.add_argument("file", nargs="?", help="Audio file to use as query")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--record", "-r", type=int, metavar="SECONDS",
                        help="Record from microphone for N seconds")
    args = parser.parse_args()

    if args.record:
        record_and_query(duration=args.record, top_k=args.top_k)
    elif args.file:
        query_with_file(args.file, top_k=args.top_k)
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python query_audio.py "path/to/audio_clip.wav"')
        print("  python query_audio.py --record 5")
        print('  python query_audio.py --top-k 10 "query.mp3"')
