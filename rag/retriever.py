"""
Audio retriever — embeds a query waveform and retrieves the most
similar audio segments from the vector store, then extracts the
actual audio snippets from the original files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import config
from rag.audio_processor import load_audio, save_segment, AudioSegment
from rag.embedder import AudioEmbedder
from rag.vector_store import AudioVectorStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """One matched segment with its similarity score and audio path."""
    rank: int
    source_file: str
    start_time: float
    end_time: float
    distance: float
    snippet_path: Path | None = None   # path to extracted WAV snippet


# ═══════════════════════════════════════════════════════════════════════════
#  Retriever
# ═══════════════════════════════════════════════════════════════════════════

class AudioRetriever:
    """
    Given a query audio, find and return the most similar segments
    from the knowledge base.
    """

    def __init__(
        self,
        embedder: AudioEmbedder,
        store: AudioVectorStore,
    ):
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query_waveform: np.ndarray,
        sr: int,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Embed the query audio and search the vector store.

        Returns a list of RetrievalResult sorted by relevance (best first).
        """
        k = top_k or config.retriever.top_k

        # 1. Embed the query
        logger.info("Embedding query audio (%.2f s) …", len(query_waveform) / sr)
        query_emb = self._embedder.embed(query_waveform, sr)

        # 2. Search
        raw = self._store.query(query_emb, top_k=k)

        if not raw["ids"]:
            logger.warning("No results found in vector store")
            return []

        # 3. Build results
        results: list[RetrievalResult] = []
        for rank, (uid, dist, meta) in enumerate(
            zip(raw["ids"], raw["distances"], raw["metadatas"]),
            start=1,
        ):
            results.append(RetrievalResult(
                rank=rank,
                source_file=meta["source_file"],
                start_time=meta["start_time"],
                end_time=meta["end_time"],
                distance=round(dist, 4),
            ))

        logger.info(
            "Retrieved %d results (distances: %s)",
            len(results),
            [r.distance for r in results],
        )
        return results

    def extract_snippets(
        self,
        results: list[RetrievalResult],
        data_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> list[RetrievalResult]:
        """
        For each retrieval result, load the original audio and extract
        the relevant time range, saving it as a WAV file.

        Mutates each result's `snippet_path` in place and returns the list.
        """
        data_dir = Path(data_dir or config.paths.data_dir)
        output_dir = Path(output_dir or config.paths.snippets_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            source_path = data_dir / r.source_file
            if not source_path.exists():
                logger.warning("Source file missing: %s", source_path)
                continue

            waveform, sr = load_audio(source_path)

            start_sample = int(r.start_time * sr)
            end_sample = int(r.end_time * sr)
            snippet_waveform = waveform[start_sample:end_sample]

            seg = AudioSegment(
                waveform=snippet_waveform,
                sample_rate=sr,
                start_time=r.start_time,
                end_time=r.end_time,
                source_file=r.source_file,
                chunk_index=r.rank,
            )
            r.snippet_path = save_segment(seg, output_dir)

        return results
