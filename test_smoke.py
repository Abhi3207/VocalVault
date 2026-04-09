"""
Smoke test for the Audio RAG pipeline.

Generates synthetic audio tones, ingests them, and verifies that a query
tone retrieves the correct source — all without real audio files.
"""

import sys
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config


def generate_tone(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    """Generate a pure sine wave at the given frequency."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def main():
    import soundfile as sf

    sr = config.audio.sample_rate
    data_dir = config.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate synthetic audio files ──────────────────────────────
    print("🔊 Generating synthetic test audio …")

    tone_a = generate_tone(440.0, 15.0, sr)   # A4 note, 15 seconds
    tone_b = generate_tone(880.0, 15.0, sr)   # A5 note, 15 seconds

    path_a = data_dir / "tone_440hz.wav"
    path_b = data_dir / "tone_880hz.wav"
    sf.write(str(path_a), tone_a, sr)
    sf.write(str(path_b), tone_b, sr)
    print(f"  → Saved {path_a.name} and {path_b.name}")

    # ── 2. Ingest both into the pipeline ───────────────────────────────
    from rag.pipeline import get_pipeline

    pipeline = get_pipeline()
    pipeline.clear_all()

    print("\n⚡ Ingesting tone_440hz.wav …")
    r1 = pipeline.ingest(path_a)
    print(f"  → {r1['segments']} segments in {r1['seconds']}s")

    print("⚡ Ingesting tone_880hz.wav …")
    r2 = pipeline.ingest(path_b)
    print(f"  → {r2['segments']} segments in {r2['seconds']}s")

    print(f"\n📊 Knowledge base: {pipeline.document_count()} total segments")

    # ── 3. Query with a short 440 Hz tone ──────────────────────────────
    print("\n🔍 Querying with a 3-second 440 Hz tone …")

    query_tone = generate_tone(440.0, 3.0, sr)
    query_path = data_dir / "query_440hz.wav"
    sf.write(str(query_path), query_tone, sr)

    results = pipeline.query(query_path, top_k=3)

    print(f"\n📋 Results ({len(results)} matches):")
    for r in results:
        print(
            f"  #{r.rank}  {r.source_file}  "
            f"[{r.start_time:.1f}s – {r.end_time:.1f}s]  "
            f"distance={r.distance:.4f}  "
            f"snippet={r.snippet_path}"
        )

    # ── 4. Verify ──────────────────────────────────────────────────────
    if results and results[0].source_file == "tone_440hz.wav":
        print("\n✅ PASS — top result correctly matches the 440 Hz source!")
        return 0
    else:
        print("\n❌ FAIL — top result did not match expected source.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
