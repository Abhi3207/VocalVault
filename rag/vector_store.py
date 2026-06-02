"""
ChromaDB-backed vector store for audio segment embeddings.

Stores CLAP embeddings alongside metadata (source file, start/end time,
chunk index) so that retrieved results can be mapped back to their
original audio regions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import numpy as np

from config import config

logger = logging.getLogger(__name__)


class AudioVectorStore:
    """Persistent vector storage for audio embeddings."""

    COLLECTION_NAME = "audio_segments"

    def __init__(self, persist_dir: str | Path | None = None):
        self._persist_dir = str(persist_dir or config.paths.vectordb_dir)
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ── Lazy initialisation ────────────────────────────────────────────

    def _ensure_ready(self):
        if self._collection is not None:
            return

        logger.info("Opening ChromaDB at %s", self._persist_dir)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": config.retriever.distance_metric},
        )
        logger.info(
            "Collection '%s' ready — %d existing documents",
            self.COLLECTION_NAME,
            self._collection.count(),
        )

    # ── Write ──────────────────────────────────────────────────────────

    def add_segments(
        self,
        ids: list[str],
        embeddings: list[np.ndarray],
        metadatas: list[dict],
    ) -> int:
        """
        Upsert audio segment embeddings into the store.

        Parameters
        ----------
        ids : unique IDs (one per segment)
        embeddings : list of numpy arrays (512-dim each)
        metadatas : list of dicts with keys like
            {"source_file", "start_time", "end_time", "chunk_index"}

        Returns
        -------
        Number of segments added.
        """
        self._ensure_ready()

        # ChromaDB accepts plain lists
        emb_lists = [e.tolist() for e in embeddings]

        self._collection.upsert(
            ids=ids,
            embeddings=emb_lists,
            metadatas=metadatas,
        )

        logger.info("Upserted %d segments into vector store", len(ids))
        return len(ids)

    # ── Read ───────────────────────────────────────────────────────────

    def query(
        self,
        embedding: np.ndarray,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> dict:
        """
        Find the closest audio segments to the given embedding.

        Parameters
        ----------
        embedding : query vector (512-dim)
        top_k : number of results to return
        where : optional ChromaDB metadata filter, e.g.
            {"source_file": "my_audio.wav"}

        Returns a dict with keys:
            ids, distances, metadatas  (each a list of length ≤ top_k).
        """
        self._ensure_ready()
        k = top_k or config.retriever.top_k
        count = self._collection.count()

        if count == 0:
            return {"ids": [], "distances": [], "metadatas": []}

        query_kwargs = dict(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, count),
            include=["distances", "metadatas"],
        )
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        }

    # ── Management ─────────────────────────────────────────────────────

    def document_count(self) -> int:
        self._ensure_ready()
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Return unique source filenames in the store."""
        self._ensure_ready()
        if self._collection.count() == 0:
            return []

        all_meta = self._collection.get(include=["metadatas"])
        sources = sorted({m["source_file"] for m in all_meta["metadatas"]})
        return sources

    def clear_all(self):
        """Delete the entire collection and recreate it."""
        self._ensure_ready()
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": config.retriever.distance_metric},
        )
        logger.info("Vector store cleared")
