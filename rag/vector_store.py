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
        try:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": config.retriever.distance_metric},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at '{self._persist_dir}': {exc}"
            ) from exc

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

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=emb_lists,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to upsert {len(ids)} segments into ChromaDB: {exc}"
            ) from exc

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

        try:
            results = self._collection.query(**query_kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return {"ids": [], "distances": [], "metadatas": []}

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

    def delete_source(self, source_file: str) -> int:
        """
        Delete all segments belonging to a specific source file.

        Returns the number of segments deleted.
        """
        self._ensure_ready()

        # Find all IDs for this source
        all_data = self._collection.get(
            where={"source_file": source_file},
            include=["metadatas"],
        )
        ids_to_delete = all_data["ids"]

        if not ids_to_delete:
            logger.info("No segments found for source '%s'", source_file)
            return 0

        self._collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d segments for source '%s'",
            len(ids_to_delete), source_file,
        )
        return len(ids_to_delete)

    def clear_all(self):
        """Delete the entire collection and recreate it."""
        self._ensure_ready()
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": config.retriever.distance_metric},
        )
        logger.info("Vector store cleared")
