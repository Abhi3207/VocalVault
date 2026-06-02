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
import shutil
from datetime import datetime
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


def _save_query_io(query_audio_path: str, results: list) -> Path:
    """Save query input and output snippets into a timestamped folder."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    query_dir = config.paths.queries_dir / timestamp
    query_dir.mkdir(parents=True, exist_ok=True)

    # Save input audio
    src = Path(query_audio_path)
    dest_input = query_dir / f"query_input{src.suffix}"
    shutil.copy2(str(src), str(dest_input))
    logger.info("Saved query input → %s", dest_input)

    # Save output snippets
    for r in results:
        if r.snippet_path and Path(r.snippet_path).exists():
            dest_name = f"result_{r.rank}_{r.source_file}_{r.start_time:.1f}s-{r.end_time:.1f}s.wav"
            # Sanitise filename (replace problematic characters)
            dest_name = dest_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            shutil.copy2(str(r.snippet_path), str(query_dir / dest_name))

    logger.info("Saved %d result snippets → %s", len(results), query_dir)
    return query_dir


def _format_results(results: list, status_prefix: str) -> list:
    """Format retrieval results into Gradio output updates."""
    audio_updates = []
    label_updates = []
    for i in range(5):
        if i < len(results):
            r = results[i]
            merge_info = f" · merged {r.merged_count} segments" if r.merged_count > 1 else ""
            audio_updates.append(gr.update(
                visible=True,
                value=str(r.snippet_path) if r.snippet_path else None,
            ))
            label_updates.append(gr.update(
                visible=True,
                value=(
                    f"**#{r.rank}** · `{r.source_file}` · "
                    f"{r.start_time:.1f}s – {r.end_time:.1f}s · "
                    f"distance: {r.distance:.4f}{merge_info}"
                ),
            ))
        else:
            audio_updates.append(gr.update(visible=False, value=None))
            label_updates.append(gr.update(visible=False, value=""))

    return [status_prefix, *audio_updates, *label_updates]


def _empty_results(message: str) -> list:
    """Return empty result outputs with a status message."""
    return [
        message,
        *[gr.update(visible=False, value=None) for _ in range(5)],
        *[gr.update(visible=False, value="") for _ in range(5)],
    ]


def _query(audio_file) -> list:
    """Handle an audio query — retrieve similar segments."""
    pipeline = get_pipeline()

    if pipeline.document_count() == 0:
        return _empty_results("📭 No audio ingested yet. Upload audio files first.")

    if audio_file is None:
        return _empty_results("⚠️ No query audio provided. Record or upload an audio snippet.")

    results = pipeline.query(audio_file, top_k=config.retriever.top_k)

    if not results:
        return _empty_results("🔍 No matching segments found.")

    # Save query input + output together
    saved_dir = _save_query_io(audio_file, results)

    status = (
        f"### 🔊 Found {len(results)} matching segment(s)\n"
        f"💾 Saved to `{saved_dir.name}/`"
    )
    return _format_results(results, status)


def _query_text(text_query: str) -> list:
    """Handle a text query — search audio KB by description."""
    pipeline = get_pipeline()

    if pipeline.document_count() == 0:
        return _empty_results("📭 No audio ingested yet. Upload audio files first.")

    if not text_query or not text_query.strip():
        return _empty_results("⚠️ Enter a text description to search.")

    results = pipeline.query_text(text_query.strip(), top_k=config.retriever.top_k)

    if not results:
        return _empty_results(f'🔍 No matching segments found for "{text_query}".')

    status = (
        f"### 🔊 Found {len(results)} segment(s) matching "
        f'"{text_query}"'
    )
    return _format_results(results, status)


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
            "an audio snippet or text description — retrieval powered by "
            "CLAP embeddings, **no speech-to-text**.",
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
                with gr.Tabs():
                    with gr.TabItem("🎙️ Audio Query"):
                        query_audio = gr.Audio(
                            label="Record or upload your query audio",
                            sources=["microphone", "upload"],
                            type="filepath",
                        )
                        search_btn = gr.Button(
                            "🔍 Search Similar Audio", variant="primary", size="lg"
                        )

                    with gr.TabItem("✏️ Text Query"):
                        query_text_input = gr.Textbox(
                            label="Describe the audio you're looking for",
                            placeholder='e.g. "piano music", "someone speaking", "birds chirping"',
                            lines=2,
                        )
                        text_search_btn = gr.Button(
                            "🔍 Search by Description", variant="primary", size="lg"
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

        all_outputs = [results_status, *result_audios, *result_labels]

        search_btn.click(
            fn=_query,
            inputs=query_audio,
            outputs=all_outputs,
        )
        text_search_btn.click(
            fn=_query_text,
            inputs=query_text_input,
            outputs=all_outputs,
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
