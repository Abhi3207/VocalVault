"""
Gradio-based UI for Audio RAG.

Features:
  • Multi-file audio upload with ingestion progress
  • Microphone / file-upload query input
  • Top-K audio snippet results with playback
  • Knowledge-base management panel
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from config import config
from rag.pipeline import get_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════════════════

def _ingest(files) -> str:
    """Handle audio file upload and ingestion."""
    if not files:
        return "⚠️ No files selected."

    pipeline = get_pipeline()
    messages = []

    for file_path in files:
        path = file_path if isinstance(file_path, str) else file_path.name
        filename = Path(path).name

        result = pipeline.ingest(path)
        messages.append(
            f"✅ **{filename}** — {result['segments']} segments "
            f"in {result['seconds']}s"
        )

    total_docs = pipeline.document_count()
    total_sources = len(pipeline.list_sources())
    summary = (
        "### Ingestion Complete\n\n"
        + "\n".join(messages)
        + f"\n\n**Knowledge base:** {total_sources} file(s), {total_docs} segments"
    )
    return summary


def _query(audio_file) -> list:
    """Handle an audio query — retrieve similar segments."""
    pipeline = get_pipeline()

    if pipeline.document_count() == 0:
        # Return empty outputs
        return [
            "📭 No audio ingested yet. Upload audio files first.",
            *[gr.update(visible=False, value=None) for _ in range(5)],
            *[gr.update(visible=False, value="") for _ in range(5)],
        ]

    if audio_file is None:
        return [
            "⚠️ No query audio provided. Record or upload an audio snippet.",
            *[gr.update(visible=False, value=None) for _ in range(5)],
            *[gr.update(visible=False, value="") for _ in range(5)],
        ]

    results = pipeline.query(audio_file, top_k=config.retriever.top_k)

    if not results:
        return [
            "🔍 No matching segments found.",
            *[gr.update(visible=False, value=None) for _ in range(5)],
            *[gr.update(visible=False, value="") for _ in range(5)],
        ]

    status = f"### 🔊 Found {len(results)} matching segment(s)\n"

    audio_updates = []
    label_updates = []
    for i in range(5):
        if i < len(results):
            r = results[i]
            audio_updates.append(gr.update(
                visible=True,
                value=str(r.snippet_path) if r.snippet_path else None,
            ))
            label_updates.append(gr.update(
                visible=True,
                value=(
                    f"**#{r.rank}** · `{r.source_file}` · "
                    f"{r.start_time:.1f}s – {r.end_time:.1f}s · "
                    f"distance: {r.distance:.4f}"
                ),
            ))
        else:
            audio_updates.append(gr.update(visible=False, value=None))
            label_updates.append(gr.update(visible=False, value=""))

    return [status, *audio_updates, *label_updates]


def _get_status() -> str:
    """Return current knowledge base status."""
    pipeline = get_pipeline()
    count = pipeline.document_count()
    sources = pipeline.list_sources()
    if not sources:
        return "📭 No audio files ingested."
    source_list = "\n".join(f"  • {s}" for s in sources)
    return f"**{count} segments** from **{len(sources)} file(s):**\n{source_list}"


def _clear_all() -> str:
    """Clear the entire knowledge base."""
    get_pipeline().clear_all()
    return "🗑️ All audio data cleared."


# ═══════════════════════════════════════════════════════════════════════════
#  UI Layout
# ═══════════════════════════════════════════════════════════════════════════

_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}
.result-card {
    border: 1px solid #333;
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.03);
}
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""
    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(theme=theme, css=_CSS, title="Audio RAG") as app:
        gr.Markdown(
            "# 🎵 Audio RAG\n",
            elem_classes=["main-title"],
        )
        gr.Markdown(
            "Upload audio files as your knowledge base, then query with "
            "an audio snippet — retrieval powered by CLAP embeddings, "
            "**no speech-to-text**.",
            elem_classes=["subtitle"],
        )

        with gr.Row(equal_height=False):
            # ── Left: Knowledge Base Management ────────────────────────
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### 📁 Knowledge Base")

                upload = gr.File(
                    label="Upload Audio Files",
                    file_count="multiple",
                    file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                    type="filepath",
                )
                ingest_btn = gr.Button(
                    "⚡ Ingest Audio", variant="primary", size="lg"
                )
                ingest_output = gr.Markdown(label="Ingestion Status")

                gr.Markdown("---")

                status_btn = gr.Button("🔍 Refresh Status", size="sm")
                status_output = gr.Markdown(label="KB Status")

                clear_btn = gr.Button(
                    "🗑️ Clear Knowledge Base", variant="stop", size="sm"
                )
                clear_output = gr.Markdown()

            # ── Right: Query & Results ─────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 🎙️ Query with Audio")

                query_audio = gr.Audio(
                    label="Record or upload your query audio",
                    sources=["microphone", "upload"],
                    type="filepath",
                )
                search_btn = gr.Button(
                    "🔍 Search Similar Audio", variant="primary", size="lg"
                )

                gr.Markdown("---")

                results_status = gr.Markdown()

                # Pre-create 5 result slots
                result_audios = []
                result_labels = []
                for i in range(5):
                    with gr.Group(visible=False, elem_classes=["result-card"]) as _:
                        lbl = gr.Markdown(visible=False)
                        aud = gr.Audio(
                            label=f"Result #{i+1}",
                            interactive=False,
                            visible=False,
                        )
                    result_labels.append(lbl)
                    result_audios.append(aud)

        # ── Wire events ────────────────────────────────────────────────
        ingest_btn.click(fn=_ingest, inputs=upload, outputs=ingest_output)
        status_btn.click(fn=_get_status, outputs=status_output)
        clear_btn.click(fn=_clear_all, outputs=clear_output)

        search_btn.click(
            fn=_query,
            inputs=query_audio,
            outputs=[results_status, *result_audios, *result_labels],
        )

    return app


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name=config.server_host,
        server_port=config.server_port,
        share=config.share,
    )
