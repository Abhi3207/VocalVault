"""
CLAP-based audio embedding — no speech-to-text.

Uses the CLAP (Contrastive Language-Audio Pretraining) model to produce
dense vector representations from raw audio waveforms.  The encoder maps
audio to a 512-dimensional embedding space where similar sounds cluster
together, enabling audio-to-audio retrieval via cosine similarity.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import ClapProcessor, ClapModel

from config import config

if TYPE_CHECKING:
    from rag.audio_processor import AudioSegment

logger = logging.getLogger(__name__)


class AudioEmbedder:
    """Thin wrapper around CLAP for audio → embedding conversion."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        self._model_name = model_name or config.embedding.model_name
        self._device = device or config.embedding.device
        self._model: ClapModel | None = None
        self._processor: ClapProcessor | None = None

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"AudioEmbedder(model={self._model_name!r}, "
            f"device={self._device!r}, loaded={loaded})"
        )

    # ── Lazy initialisation ────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return

        logger.info("Loading CLAP model: %s  (device=%s)", self._model_name, self._device)
        self._processor = ClapProcessor.from_pretrained(self._model_name)
        self._model = ClapModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        logger.info("CLAP model loaded ✓")

    # ── Public API ─────────────────────────────────────────────────────

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Embed a single waveform.

        Parameters
        ----------
        waveform : 1-D float32 array
        sr : sample rate of the waveform

        Returns
        -------
        np.ndarray of shape (512,) — L2-normalised embedding.

        Raises
        ------
        ValueError
            If the waveform is empty or too short to embed.
        """
        if waveform is None or len(waveform) == 0:
            raise ValueError("Cannot embed an empty waveform")
        if len(waveform) < sr * 0.1:
            raise ValueError(
                f"Waveform too short to embed ({len(waveform) / sr:.3f}s). "
                "Minimum ~0.1s required."
            )

        self._ensure_loaded()

        inputs = self._processor(
            audio=waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            feats = self._model.get_audio_features(**inputs)

        embedding = feats.squeeze(0).cpu().numpy().astype(np.float32)

        if config.embedding.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def embed_batch(self, segments: list[AudioSegment]) -> list[np.ndarray]:
        """
        Embed a list of AudioSegments in batches.

        Returns a list of 512-dim numpy arrays (one per segment).
        """
        if not segments:
            return []

        self._ensure_loaded()
        batch_size = config.embedding.batch_size
        all_embeddings: list[np.ndarray] = []
        t0 = time.perf_counter()

        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            waveforms = [seg.waveform for seg in batch]
            sr = batch[0].sample_rate

            inputs = self._processor(
                audio=waveforms,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.inference_mode():
                feats = self._model.get_audio_features(**inputs)

            embeddings = feats.cpu().numpy().astype(np.float32)

            # Free GPU memory between batches to prevent OOM on large ingests
            if self._device == "cuda":
                torch.cuda.empty_cache()

            if config.embedding.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                embeddings = embeddings / norms

            all_embeddings.extend(embeddings)

            logger.debug(
                "Embedded batch %d–%d / %d",
                i, min(i + batch_size, len(segments)), len(segments),
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Embedded %d segments in %.2f s (%.1f seg/s)",
            len(segments), elapsed,
            len(segments) / elapsed if elapsed > 0 else 0,
        )
        return all_embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text description into the same 512-dim space as audio.

        CLAP was trained on audio–text pairs, so text embeddings are
        directly comparable to audio embeddings via cosine similarity.

        Parameters
        ----------
        text : natural-language description (e.g. "piano music", "Telugu speech")

        Returns
        -------
        np.ndarray of shape (512,) — L2-normalised embedding.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        self._ensure_loaded()

        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            feats = self._model.get_text_features(**inputs)

        embedding = feats.squeeze(0).cpu().numpy().astype(np.float32)

        if config.embedding.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    @property
    def dimension(self) -> int:
        return config.embedding.dimension
