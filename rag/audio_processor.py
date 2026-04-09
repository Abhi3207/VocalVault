"""
Audio loading and segmentation.

Handles:
  • Loading audio files in any common format (wav, mp3, flac, ogg, m4a)
  • Resampling to the target sample rate
  • Splitting into fixed-duration overlapping segments
  • Exporting individual segments to WAV files
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AudioSegment:
    """A single audio chunk with provenance metadata."""
    waveform: np.ndarray        # 1-D float32 array
    sample_rate: int
    start_time: float           # seconds from start of source file
    end_time: float
    source_file: str            # original filename
    chunk_index: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def uid(self) -> str:
        """Unique identifier for this segment."""
        stem = Path(self.source_file).stem
        return f"{stem}__chunk{self.chunk_index:04d}"


# ═══════════════════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_audio(
    path: str | Path,
    sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to *sr* (defaults to config sample rate).

    Returns (waveform, sample_rate) where waveform is a 1-D float32 array.
    """
    sr = sr or config.audio.sample_rate
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in config.audio.supported_extensions:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {config.audio.supported_extensions}"
        )

    logger.info("Loading audio: %s  (target sr=%d)", path.name, sr)
    waveform, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    waveform = waveform.astype(np.float32)

    logger.info(
        "  → %s loaded: %.2f s, %d samples",
        path.name,
        len(waveform) / actual_sr,
        len(waveform),
    )
    return waveform, actual_sr


# ═══════════════════════════════════════════════════════════════════════════
#  Chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_audio(
    waveform: np.ndarray,
    sr: int,
    source_file: str,
    segment_duration: float | None = None,
    overlap: float | None = None,
) -> list[AudioSegment]:
    """
    Split a waveform into fixed-duration overlapping segments.

    Parameters
    ----------
    waveform : 1-D float32 array
    sr : sample rate
    source_file : name of the original file (for metadata)
    segment_duration : length of each chunk in seconds (default from config)
    overlap : overlap between consecutive chunks in seconds (default from config)

    Returns
    -------
    List of AudioSegment instances.
    """
    seg_dur = segment_duration or config.audio.segment_duration
    seg_overlap = overlap or config.audio.segment_overlap
    step = seg_dur - seg_overlap

    total_duration = len(waveform) / sr
    logger.info(
        "Chunking '%s': %.2f s → segments of %.1f s (overlap %.1f s)",
        source_file, total_duration, seg_dur, seg_overlap,
    )

    segments: list[AudioSegment] = []
    start_sample = 0
    chunk_idx = 0

    while start_sample < len(waveform):
        end_sample = start_sample + int(seg_dur * sr)
        chunk_waveform = waveform[start_sample:end_sample]

        # Skip very short trailing segments (< 1 second)
        if len(chunk_waveform) < sr:
            break

        start_time = start_sample / sr
        end_time = min(start_time + seg_dur, total_duration)

        segments.append(AudioSegment(
            waveform=chunk_waveform,
            sample_rate=sr,
            start_time=round(start_time, 3),
            end_time=round(end_time, 3),
            source_file=source_file,
            chunk_index=chunk_idx,
        ))

        start_sample += int(step * sr)
        chunk_idx += 1

    logger.info("  → Produced %d segments from '%s'", len(segments), source_file)
    return segments


# ═══════════════════════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════════════════════

def save_segment(segment: AudioSegment, output_dir: str | Path | None = None) -> Path:
    """
    Save an AudioSegment to a WAV file.

    Returns the path to the written file.
    """
    output_dir = Path(output_dir or config.paths.snippets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{segment.uid}.wav"
    out_path = output_dir / filename

    sf.write(str(out_path), segment.waveform, segment.sample_rate)
    logger.debug("Saved segment → %s", out_path)
    return out_path
