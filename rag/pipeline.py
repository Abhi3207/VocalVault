"""
Audio RAG Pipeline — orchestration layer.

Composes:
  • AudioProcessor  — load & chunk audio files
  • AudioEmbedder   — CLAP-based embeddings
  • AudioVectorStore — ChromaDB persistence
  • AudioRetriever  — similarity search & snippet extraction

Provides two main entry points:
  • ingest(audio_path)          — add audio to the knowledge base
  • query(query_audio_path)     — retrieve similar audio snippets
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from config import config
from rag.audio_processor import load_audio, chunk_audio, clear_audio_cache
from rag.embedder import AudioEmbedder
from rag.vector_store import AudioVectorStore
from rag.retriever import AudioRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# ── Module-level singleton ─────────────────────────────────────────────────
_pipeline = None  # type: AudioRAGPipeline | None

# Type alias for progress callbacks: fn(current_step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


class AudioRAGPipeline:
    """End-to-end audio RAG pipeline."""

    def __init__(self):
        self._embedder = AudioEmbedder()
        self._store = AudioVectorStore()
        self._retriever = AudioRetriever(self._embedder, self._store)

    # ═══════════════════════════════════════════════════════════════════
    #  Ingestion
    # ═══════════════════════════════════════════════════════════════════

    def ingest(
        self,
        audio_path: str | Path,
        progress: ProgressCallback | None = None,
    ) -> dict:
        """
        Load an audio file, chunk it, embed all segments, and store.

        Parameters
        ----------
        audio_path : path to any supported audio file
        progress : optional callback ``fn(step, total, message)`` for UI
                   progress tracking.

        Returns
        -------
        dict with keys: file, segments, seconds, skipped
        """
        t0 = time.perf_counter()
        audio_path = Path(audio_path)

        def _progress(step: int, total: int, msg: str):
            if progress:
                progress(step, total, msg)

        _progress(0, 4, f"Checking '{audio_path.name}'…")

        # ── Duplicate detection ────────────────────────────────────────
        existing_sources = self._store.list_sources()
        if audio_path.name in existing_sources:
            logger.info(
                "File '%s' already in knowledge base — skipping ingestion",
                audio_path.name,
            )
            _progress(4, 4, f"'{audio_path.name}' already ingested")
            return {
                "file": audio_path.name,
                "segments": 0,
                "seconds": 0.0,
                "skipped": True,
            }

        # 1. Copy to data dir so snippets can be extracted later
        _progress(1, 4, f"Copying '{audio_path.name}' to data dir…")
        dest = config.paths.data_dir / audio_path.name
        if not dest.exists():
            import shutil
            shutil.copy2(str(audio_path), str(dest))
            logger.info("Copied '%s' → data dir", audio_path.name)

        # 2. Load & chunk
        _progress(2, 4, f"Loading and chunking '{audio_path.name}'…")
        waveform, sr = load_audio(dest)
        segments = chunk_audio(waveform, sr, source_file=audio_path.name)

        if not segments:
            logger.warning("No segments produced from '%s'", audio_path.name)
            return {
                "file": audio_path.name,
                "segments": 0,
                "seconds": 0.0,
                "skipped": False,
            }

        # 3. Embed
        _progress(3, 4, f"Embedding {len(segments)} segments…")
        embeddings = self._embedder.embed_batch(segments)

        # 4. Store
        _progress(4, 4, "Storing in vector database…")
        ids = [seg.uid for seg in segments]
        metadatas = [
            {
                "source_file": seg.source_file,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "chunk_index": seg.chunk_index,
                "duration": seg.duration,
            }
            for seg in segments
        ]
        self._store.add_segments(ids, embeddings, metadatas)

        elapsed = round(time.perf_counter() - t0, 2)
        logger.info(
            "Ingested '%s': %d segments in %.2f s",
            audio_path.name, len(segments), elapsed,
        )
        return {
            "file": audio_path.name,
            "segments": len(segments),
            "seconds": elapsed,
            "skipped": False,
        }

    def ingest_bytes(self, filename: str, audio_bytes: bytes) -> dict:
        """Ingest from raw bytes (e.g. Gradio upload)."""
        dest = config.paths.data_dir / filename
        dest.write_bytes(audio_bytes)
        return self.ingest(dest)

    # ═══════════════════════════════════════════════════════════════════
    #  Query
    # ═══════════════════════════════════════════════════════════════════

    def query(
        self,
        query_audio_path: str | Path,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Given a query audio file, retrieve the most similar segments.

        Returns a list of RetrievalResult (each with a .snippet_path
        pointing to the extracted WAV file).
        """
        query_audio_path = Path(query_audio_path)
        waveform, sr = load_audio(query_audio_path)

        results = self._retriever.retrieve(waveform, sr, top_k=top_k)

        if results:
            results = self._retriever.extract_snippets(results)

        return results

    def query_text(
        self,
        text: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Search the audio knowledge base using a text description.

        Uses CLAP's text encoder to embed the query into the same
        vector space, enabling cross-modal retrieval.

        Parameters
        ----------
        text : natural-language description (e.g. "piano music")
        top_k : max results to return

        Returns
        -------
        list of RetrievalResult with extracted snippets.
        """
        results = self._retriever.retrieve_text(text, top_k=top_k)

        if results:
            results = self._retriever.extract_snippets(results)

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  Management
    # ═══════════════════════════════════════════════════════════════════

    def document_count(self) -> int:
        return self._store.document_count()

    def list_sources(self) -> list[str]:
        return self._store.list_sources()

    def delete_source(self, source_file: str) -> int:
        """
        Remove a single source file from the knowledge base.

        Deletes all of its segments from the vector store and removes
        the data file if it exists.

        Returns the number of segments deleted.
        """
        count = self._store.delete_source(source_file)

        # Also remove the data file copy
        data_path = config.paths.data_dir / source_file
        if data_path.exists():
            data_path.unlink()
            logger.info("Removed data file: %s", data_path)

        # Clear audio cache since the file is gone
        clear_audio_cache()

        return count

    def clear_all(self):
        """Clear vector store and snippet cache."""
        self._store.clear_all()
        clear_audio_cache()
        # Clean up snippets
        snippets = config.paths.snippets_dir
        if snippets.exists():
            for f in snippets.iterdir():
                if f.is_file():
                    f.unlink()
        logger.info("Pipeline cleared — store and snippets removed")


def get_pipeline() -> AudioRAGPipeline:
    """Return (or create) the global pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AudioRAGPipeline()
    return _pipeline
