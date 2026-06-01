"""
Create sample query audio files from existing ingested data.

Extracts short clips to demonstrate what a good query input looks like.
Run: python create_sample_query.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
import librosa

from config import config

SAMPLE_DIR = config.paths.data_dir.parent / "sample_queries"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def extract_clip(source_path: Path, start_sec: float, duration_sec: float, output_name: str):
    """Extract a short clip from a source audio file."""
    sr = config.audio.sample_rate
    waveform, _ = librosa.load(str(source_path), sr=sr, mono=True)
    
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    clip = waveform[start_sample:end_sample].astype(np.float32)
    
    out_path = SAMPLE_DIR / output_name
    sf.write(str(out_path), clip, sr)
    
    print(f"  ✅ {output_name}")
    print(f"     Source: {source_path.name}")
    print(f"     Range: {start_sec:.1f}s – {start_sec + duration_sec:.1f}s")
    print(f"     Duration: {duration_sec:.1f}s  |  Samples: {len(clip)}  |  SR: {sr} Hz")
    print(f"     Size: {out_path.stat().st_size / 1024:.1f} KB")
    print()
    return out_path


def generate_tone_query(freq_hz: float, duration_sec: float, output_name: str):
    """Generate a synthetic tone as a query sample."""
    sr = config.audio.sample_rate
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    
    out_path = SAMPLE_DIR / output_name
    sf.write(str(out_path), waveform, sr)
    
    print(f"  ✅ {output_name}")
    print(f"     Synthetic {freq_hz:.0f} Hz tone")
    print(f"     Duration: {duration_sec:.1f}s  |  Samples: {len(waveform)}  |  SR: {sr} Hz")
    print(f"     Size: {out_path.stat().st_size / 1024:.1f} KB")
    print()
    return out_path


def main():
    print("=" * 60)
    print("  🎵 Creating Sample Query Audio Files")
    print("=" * 60)
    print()
    
    samples_created = []
    
    # ── Sample 1: Extract a 5s clip from the Garikapati speech ──────
    speech_files = list(config.paths.data_dir.glob("*.mp3"))
    if speech_files:
        speech = speech_files[0]
        print("📢 Sample 1 — Speech clip (from ingested data)")
        print("-" * 50)
        
        # Extract from ~30s in (past any intro silence)
        path = extract_clip(speech, start_sec=30.0, duration_sec=5.0, 
                           output_name="sample_speech_clip_30s.wav")
        samples_created.append(path)
        
        # Another clip from a different part
        print("📢 Sample 2 — Speech clip (different section)")
        print("-" * 50)
        path = extract_clip(speech, start_sec=120.0, duration_sec=5.0,
                           output_name="sample_speech_clip_120s.wav")
        samples_created.append(path)
    
    # ── Sample 3: Synthetic 440 Hz tone query ──────────────────────
    print("🎹 Sample 3 — Synthetic 440 Hz tone query")
    print("-" * 50)
    path = generate_tone_query(440.0, 3.0, "sample_tone_440hz.wav")
    samples_created.append(path)
    
    # ── Sample 4: Synthetic 880 Hz tone query ──────────────────────
    print("🎹 Sample 4 — Synthetic 880 Hz tone query")
    print("-" * 50)
    path = generate_tone_query(880.0, 3.0, "sample_tone_880hz.wav")
    samples_created.append(path)
    
    # ── Summary ────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  📁 {len(samples_created)} sample queries saved to:")
    print(f"     {SAMPLE_DIR}")
    print()
    print("  HOW TO USE:")
    print("  1. Launch the app:  python app.py")
    print("  2. On the right panel, click 'Upload' under query audio")
    print("  3. Pick any file from sample_queries/")
    print("  4. Click '🔍 Search Similar Audio'")
    print()
    print("  WHAT MAKES A GOOD QUERY:")
    print("  • Duration: 2–10 seconds (short enough for fast embedding)")
    print("  • Format: .wav, .mp3, .flac, .ogg, .m4a")
    print("  • Content: Should sound similar to something in your KB")
    print("  • Quality: Clean audio works best (less background noise)")
    print("=" * 60)


if __name__ == "__main__":
    main()
