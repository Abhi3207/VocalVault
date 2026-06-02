"""
Audio retriever — embeds a query waveform and retrieves the most
similar audio segments from the vector store, then extracts the
actual audio snippets from the original files.

Improvements over naive top-k retrieval:
  • Over-fetch & deduplicate overlapping segments from the same file
  • Merge adjacent time ranges into contiguous spans
  • Filter results by maximum distance threshold
  • Multi-scale query embedding for long audio queries
  • Cached file loading during snippet extraction
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import config
from rag.audio_processor import load_audio, save_segment, AudioSegment, chunk_audio
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
    merged_count: int = 1              # how many raw segments were merged


# ═══════════════════════════════════════════════════════════════════════════
#  Retriever
# ═══════════════════════════════════════════════════════════════════════════

class AudioRetriever:
    """
    Given a query audio, find and return the most similar segments
    from the knowledge base.

    Features:
      • Over-fetches candidates, then deduplicates overlapping segments
      • Merges adjacent/overlapping time ranges from the same source file
      • Filters results below a relevance threshold
      • Supports multi-scale query embedding for longer audio
    """

    def __init__(
        self,
        embedder: AudioEmbedder,
        store: AudioVectorStore,
    ):
        self._embedder = embedder
        self._store = store

    # ── Main retrieval entry point ─────────────────────────────────────

    def retrieve(
        self,
        query_waveform: np.ndarray,
        sr: int,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Embed the query audio and search the vector store.

        For short queries (≤ segment_duration), embeds as a single vector.
        For longer queries, chunks and performs multi-scale retrieval,
        then merges and deduplicates all candidates.

        Returns a list of RetrievalResult sorted by relevance (best first).
        """
        k = top_k or config.retriever.top_k
        fetch_k = k * config.retriever.over_fetch_factor
        query_duration = len(query_waveform) / sr

        logger.info("Embedding query audio (%.2f s) …", query_duration)

        # Decide single-scale vs multi-scale
        if query_duration > config.audio.segment_duration:
            candidates = self._multi_scale_retrieve(query_waveform, sr, fetch_k)
        else:
            candidates = self._single_retrieve(query_waveform, sr, fetch_k)

        if not candidates:
            logger.warning("No results found in vector store")
            return []

        # Post-processing pipeline
        candidates = self._filter_by_threshold(candidates)
        candidates = self._merge_overlapping(candidates)
        candidates = self._assign_ranks(candidates, k)

        logger.info(
            "Retrieved %d results after dedup (distances: %s)",
            len(candidates),
            [r.distance for r in candidates],
        )
        return candidates

    def retrieve_text(
        self,
        text: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve audio segments matching a text description.

        Uses CLAP's text encoder to embed the query into the same
        vector space as audio embeddings.
        """
        k = top_k or config.retriever.top_k
        fetch_k = k * config.retriever.over_fetch_factor

        logger.info("Embedding text query: '%s'", text)
        query_emb = self._embedder.embed_text(text)

        raw = self._store.query(query_emb, top_k=fetch_k)
        candidates = self._raw_to_results(raw)

        if not candidates:
            logger.warning("No results found for text query")
            return []

        candidates = self._filter_by_threshold(candidates)
        candidates = self._merge_overlapping(candidates)
        candidates = self._assign_ranks(candidates, k)

        logger.info(
            "Text query retrieved %d results (distances: %s)",
            len(candidates),
            [r.distance for r in candidates],
        )
        return candidates

    # ── Single-scale retrieval ─────────────────────────────────────────

    def _single_retrieve(
        self, waveform: np.ndarray, sr: int, fetch_k: int,
    ) -> list[RetrievalResult]:
        """Embed the entire waveform and search."""
        query_emb = self._embedder.embed(waveform, sr)
        raw = self._store.query(query_emb, top_k=fetch_k)
        return self._raw_to_results(raw)

    # ── Multi-scale retrieval ──────────────────────────────────────────

    def _multi_scale_retrieve(
        self, waveform: np.ndarray, sr: int, fetch_k: int,
    ) -> list[RetrievalResult]:
        """
        For long queries: chunk into segments matching the ingestion
        config, embed each chunk, search independently, and merge
        all candidates.
        """
        segments = chunk_audio(waveform, sr, source_file="__query__")
        logger.info(
            "Multi-scale query: %.2f s → %d chunks",
            len(waveform) / sr, len(segments),
        )

        all_candidates: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        for seg in segments:
            query_emb = self._embedder.embed(seg.waveform, sr)
            raw = self._store.query(query_emb, top_k=fetch_k)

            for uid, dist, meta in zip(
                raw["ids"], raw["distances"], raw["metadatas"]
            ):
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    all_candidates.append(RetrievalResult(
                        rank=0,
                        source_file=meta["source_file"],
                        start_time=meta["start_time"],
                        end_time=meta["end_time"],
                        distance=round(dist, 4),
                    ))
                else:
                    # If we've seen this segment before, keep the best distance
                    for c in all_candidates:
                        if (c.source_file == meta["source_file"]
                                and c.start_time == meta["start_time"]):
                            c.distance = min(c.distance, round(dist, 4))
                            break

        # Sort by distance (best first)
        all_candidates.sort(key=lambda r: r.distance)
        return all_candidates

    # ── Post-processing helpers ────────────────────────────────────────

    @staticmethod
    def _raw_to_results(raw: dict) -> list[RetrievalResult]:
        """Convert raw vector store response to RetrievalResult list."""
        results: list[RetrievalResult] = []
        for uid, dist, meta in zip(
            raw["ids"], raw["distances"], raw["metadatas"]
        ):
            results.append(RetrievalResult(
                rank=0,  # assigned later after dedup
                source_file=meta["source_file"],
                start_time=meta["start_time"],
                end_time=meta["end_time"],
                distance=round(dist, 4),
            ))
        return results

    @staticmethod
    def _filter_by_threshold(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Remove results with distance above the configured threshold."""
        max_dist = config.retriever.max_distance
        before = len(results)
        filtered = [r for r in results if r.distance <= max_dist]
        dropped = before - len(filtered)
        if dropped:
            logger.info(
                "Threshold filter: dropped %d/%d results (max_distance=%.2f)",
                dropped, before, max_dist,
            )
        return filtered

    @staticmethod
    def _merge_overlapping(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Merge overlapping or adjacent segments from the same source file.

        Two segments are merged if they overlap or are within
        `merge_overlap_sec` of each other. The merged result keeps
        the best (lowest) distance and spans the combined time range.
        """
        if not results:
            return []

        gap = config.retriever.merge_overlap_sec

        # Group by source file
        by_file: dict[str, list[RetrievalResult]] = defaultdict(list)
        for r in results:
            by_file[r.source_file].append(r)

        merged: list[RetrievalResult] = []

        for source_file, file_results in by_file.items():
            # Sort by start time
            file_results.sort(key=lambda r: r.start_time)

            current = file_results[0]
            merge_count = 1

            for nxt in file_results[1:]:
                # Check if next segment overlaps or is within the gap
                if nxt.start_time <= current.end_time + gap:
                    # Merge: extend time range, keep best distance
                    current = RetrievalResult(
                        rank=0,
                        source_file=source_file,
                        start_time=current.start_time,
                        end_time=max(current.end_time, nxt.end_time),
                        distance=min(current.distance, nxt.distance),
                        merged_count=merge_count + 1,
                    )
                    merge_count += 1
                else:
                    # No overlap — emit current, start new
                    current.merged_count = merge_count
                    merged.append(current)
                    current = nxt
                    merge_count = 1

            current.merged_count = merge_count
            merged.append(current)

        # Sort by distance (best first)
        merged.sort(key=lambda r: r.distance)

        if len(merged) < len(results):
            logger.info(
                "Merge: %d raw segments → %d deduplicated results",
                len(results), len(merged),
            )
        return merged

    @staticmethod
    def _assign_ranks(
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Trim to top_k and assign sequential rank numbers."""
        results = results[:top_k]
        for i, r in enumerate(results, start=1):
            r.rank = i
        return results

    # ── Snippet extraction ─────────────────────────────────────────────

    def extract_snippets(
        self,
        results: list[RetrievalResult],
        data_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> list[RetrievalResult]:
        """
        For each retrieval result, load the original audio and extract
        the relevant time range, saving it as a WAV file.

        Groups results by source file to avoid redundant file loads.
        Mutates each result's `snippet_path` in place and returns the list.
        """
        data_dir = Path(data_dir or config.paths.data_dir)
        output_dir = Path(output_dir or config.paths.snippets_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group by source file to load each file only once
        by_file: dict[str, list[RetrievalResult]] = defaultdict(list)
        for r in results:
            by_file[r.source_file].append(r)

        for source_file, file_results in by_file.items():
            source_path = data_dir / source_file
            if not source_path.exists():
                logger.warning("Source file missing: %s", source_path)
                continue

            # Load once for all snippets from this file
            waveform, sr = load_audio(source_path)

            for r in file_results:
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
