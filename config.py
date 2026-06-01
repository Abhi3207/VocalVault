"""
Central configuration for the Audio RAG pipeline.
All paths, model names, and hyperparameters are defined here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PathsConfig:
    """File-system paths used across the project."""
    data_dir: Path = PROJECT_ROOT / "data"
    vectordb_dir: Path = PROJECT_ROOT / "vectordb"
    models_dir: Path = PROJECT_ROOT / "models"
    snippets_dir: Path = PROJECT_ROOT / "snippets"   # extracted result snippets
    queries_dir: Path = PROJECT_ROOT / "queries"     # saved query input + output pairs

    def __post_init__(self):
        for d in (self.data_dir, self.vectordb_dir, self.models_dir, self.snippets_dir, self.queries_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    """Audio processing parameters."""
    sample_rate: int = 48_000          # CLAP expects 48 kHz
    segment_duration: float = 10.0     # seconds per chunk
    segment_overlap: float = 2.0       # overlap between consecutive chunks
    supported_extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


@dataclass
class EmbeddingConfig:
    """CLAP audio embedding model settings."""
    model_name: str = "laion/larger_clap_music_and_speech"
    dimension: int = 512
    batch_size: int = 8
    normalize: bool = True
    device: str = "cpu"                # "cuda" if GPU available


@dataclass
class RetrieverConfig:
    """Retrieval settings."""
    top_k: int = 5
    distance_metric: str = "cosine"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)

    # Gradio
    server_host: str = "0.0.0.0"
    server_port: int = 7861
    share: bool = False


# ── Singleton instance ──────────────────────────────────────────────────────
config = AppConfig()
