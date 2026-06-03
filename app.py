"""
Gradio-based UI for Audio RAG.

Features:
  • Multi-file audio upload with ingestion progress
  • Microphone / file-upload query input
  • Top-K audio snippet results with playback & similarity %
  • Knowledge-base management panel with per-source delete
  • Premium dark UI with glassmorphism and micro-animations
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

def _ingest(files, progress=gr.Progress(track_tqdm=False)) -> str:
    """Handle audio file upload and ingestion."""
    if not files:
        return "⚠️ No files selected."

    pipeline = get_pipeline()
    messages = []
    total_files = len(files)

    for idx, file_path in enumerate(files):
        path = file_path if isinstance(file_path, str) else file_path.name
        filename = Path(path).name

        progress((idx) / total_files, desc=f"Processing {filename}…")

        result = pipeline.ingest(path)

        if result.get("skipped"):
            messages.append(f"⏭️ **{filename}** — already in knowledge base")
        else:
            messages.append(
                f"✅ **{filename}** — {result['segments']} segments "
                f"in {result['seconds']}s"
            )

    progress(1.0, desc="Done!")

    total_docs = pipeline.document_count()
    total_sources = len(pipeline.list_sources())
    summary = (
        "### ✨ Ingestion Complete\n\n"
        + "\n".join(messages)
        + f"\n\n📊 **Knowledge base:** {total_sources} file(s), {total_docs} segments"
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
            merge_info = f"  ·  merged {r.merged_count} segments" if r.merged_count > 1 else ""
            sim_pct = r.similarity_pct

            # Build a styled label with similarity bar
            bar_width = max(0, min(100, sim_pct))
            if sim_pct >= 70:
                bar_color = "#8b5cf6"
            elif sim_pct >= 40:
                bar_color = "#6366f1"
            else:
                bar_color = "#64748b"

            label_html = (
                f"**#{r.rank}**  ·  `{r.source_file}`  ·  "
                f"{r.start_time:.1f}s – {r.end_time:.1f}s  ·  "
                f"**{sim_pct:.0f}% match**{merge_info}\n\n"
                f'<div style="background:rgba(100,100,100,0.2);border-radius:6px;'
                f'height:6px;width:100%;overflow:hidden;">'
                f'<div style="background:{bar_color};height:100%;'
                f'width:{bar_width}%;border-radius:6px;'
                f'transition:width 0.6s ease;"></div></div>'
            )

            audio_updates.append(gr.update(
                visible=True,
                value=str(r.snippet_path) if r.snippet_path else None,
            ))
            label_updates.append(gr.update(
                visible=True,
                value=label_html,
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
        return (
            '<div style="text-align:center;padding:2rem;color:#94a3b8;">'
            "<p style='font-size:2rem;'>📭</p>"
            "<p>No audio files ingested yet.</p>"
            "<p style='font-size:0.85rem;'>Upload audio files to get started.</p>"
            "</div>"
        )
    source_list = "\n".join(f"  • `{s}`" for s in sources)
    return (
        f"**{count} segments** from **{len(sources)} file(s):**\n\n"
        f"{source_list}"
    )


def _get_source_choices() -> gr.update:
    """Return list of ingested sources for the delete dropdown."""
    pipeline = get_pipeline()
    sources = pipeline.list_sources()
    return gr.update(choices=sources, value=None)


def _delete_source(source_file: str) -> tuple[str, gr.update]:
    """Delete a single source from the knowledge base."""
    if not source_file:
        return "⚠️ No file selected.", gr.update()

    pipeline = get_pipeline()
    count = pipeline.delete_source(source_file)

    # Refresh the dropdown
    sources = pipeline.list_sources()
    dropdown_update = gr.update(choices=sources, value=None)

    return (
        f"🗑️ Deleted **{source_file}** ({count} segments removed).",
        dropdown_update,
    )


def _clear_all() -> str:
    """Clear the entire knowledge base."""
    get_pipeline().clear_all()
    return "🗑️ All audio data cleared."


# ═══════════════════════════════════════════════════════════════════════════
#  UI Layout
# ═══════════════════════════════════════════════════════════════════════════

_CSS = """
/* ── Global overrides ──────────────────────────────────────────── */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto;
}

/* ── Header ────────────────────────────────────────────────────── */
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-header h1 {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.subtitle {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* ── Glassmorphism cards ───────────────────────────────────────── */
.glass-card {
    background: rgba(30, 30, 45, 0.5) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(139, 92, 246, 0.15) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(139, 92, 246, 0.35) !important;
    box-shadow: 0 0 24px rgba(139, 92, 246, 0.08);
}

/* ── Result cards ──────────────────────────────────────────────── */
.result-card {
    background: rgba(30, 30, 45, 0.4) !important;
    border: 1px solid rgba(100, 116, 139, 0.2) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important;
    transition: all 0.3s ease;
}
.result-card:hover {
    border-color: rgba(139, 92, 246, 0.4) !important;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.1);
    transform: translateY(-1px);
}

/* ── Section headers ───────────────────────────────────────────── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #c4b5fd;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Stats bar ─────────────────────────────────────────────────── */
.stats-bar {
    display: flex;
    gap: 1.5rem;
    padding: 0.6rem 1rem;
    background: rgba(139, 92, 246, 0.08);
    border-radius: 10px;
    border: 1px solid rgba(139, 92, 246, 0.12);
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: #c4b5fd;
}

/* ── Buttons ───────────────────────────────────────────────────── */
.primary-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(124, 58, 237, 0.25) !important;
}
.primary-btn:hover {
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(-1px) !important;
}
.danger-btn {
    background: rgba(239, 68, 68, 0.15) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    color: #fca5a5 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
.danger-btn:hover {
    background: rgba(239, 68, 68, 0.25) !important;
    border-color: rgba(239, 68, 68, 0.5) !important;
}

/* ── Pulse animation for processing ────────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.3); }
    50% { box-shadow: 0 0 16px 4px rgba(139, 92, 246, 0.15); }
}
.processing {
    animation: pulse-glow 2s infinite;
}

/* ── Footer ────────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 1rem 0 0.5rem;
    color: #64748b;
    font-size: 0.8rem;
    border-top: 1px solid rgba(100, 116, 139, 0.15);
    margin-top: 1.5rem;
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
    ).set(
        body_background_fill="linear-gradient(135deg, #0f0a1a 0%, #1a1025 40%, #0f172a 100%)",
        body_text_color="#e2e8f0",
        block_background_fill="rgba(15, 15, 25, 0.6)",
        block_border_color="rgba(100, 116, 139, 0.15)",
        block_border_width="1px",
        block_label_text_color="#a78bfa",
        block_title_text_color="#c4b5fd",
        input_background_fill="rgba(30, 30, 50, 0.8)",
        input_border_color="rgba(100, 116, 139, 0.25)",
        button_primary_background_fill="linear-gradient(135deg, #7c3aed 0%, #6366f1 100%)",
        button_primary_text_color="white",
        button_secondary_background_fill="rgba(100, 116, 139, 0.15)",
        button_secondary_text_color="#c4b5fd",
    )

    with gr.Blocks(theme=theme, css=_CSS, title="Audio RAG — VocalVault") as app:
        # ── Header ─────────────────────────────────────────────────────
        gr.HTML(
            '<div class="main-header">'
            "<h1>🎵 VocalVault</h1>"
            "</div>"
        )
        gr.Markdown(
            "Upload audio files as your knowledge base, then query with "
            "an audio snippet or text description — retrieval powered by "
            "**CLAP embeddings**, no speech-to-text.",
            elem_classes=["subtitle"],
        )

        with gr.Row(equal_height=False):
            # ══════════════════════════════════════════════════════════
            #  LEFT COLUMN — Knowledge Base Management
            # ══════════════════════════════════════════════════════════
            with gr.Column(scale=1, min_width=360):

                # ── Upload Section ─────────────────────────────────────
                with gr.Group(elem_classes=["glass-card"]):
                    gr.Markdown(
                        '<span class="section-title">📁 Knowledge Base</span>'
                    )

                    upload = gr.File(
                        label="Upload Audio Files",
                        file_count="multiple",
                        file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                        type="filepath",
                    )
                    ingest_btn = gr.Button(
                        "⚡ Ingest Audio",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"],
                    )
                    ingest_output = gr.Markdown(label="Ingestion Status")

                # ── Status Section ─────────────────────────────────────
                with gr.Group(elem_classes=["glass-card"]):
                    gr.Markdown(
                        '<span class="section-title">📊 Status</span>'
                    )
                    status_btn = gr.Button(
                        "🔄 Refresh Status",
                        size="sm",
                        variant="secondary",
                    )
                    status_output = gr.Markdown(label="KB Status")

                # ── Management Section ─────────────────────────────────
                with gr.Group(elem_classes=["glass-card"]):
                    gr.Markdown(
                        '<span class="section-title">🛠️ Manage</span>'
                    )

                    with gr.Row():
                        delete_dropdown = gr.Dropdown(
                            label="Select file to remove",
                            choices=[],
                            interactive=True,
                            scale=3,
                        )
                        delete_btn = gr.Button(
                            "🗑️",
                            size="sm",
                            elem_classes=["danger-btn"],
                            scale=1,
                        )

                    delete_output = gr.Markdown()

                    clear_btn = gr.Button(
                        "🗑️ Clear Entire Knowledge Base",
                        variant="stop",
                        size="sm",
                        elem_classes=["danger-btn"],
                    )
                    clear_output = gr.Markdown()

            # ══════════════════════════════════════════════════════════
            #  RIGHT COLUMN — Query & Results
            # ══════════════════════════════════════════════════════════
            with gr.Column(scale=2):

                with gr.Group(elem_classes=["glass-card"]):
                    with gr.Tabs():
                        with gr.TabItem("🎙️ Audio Query"):
                            query_audio = gr.Audio(
                                label="Record or upload your query audio",
                                sources=["microphone", "upload"],
                                type="filepath",
                            )
                            search_btn = gr.Button(
                                "🔍 Search Similar Audio",
                                variant="primary",
                                size="lg",
                                elem_classes=["primary-btn"],
                            )

                        with gr.TabItem("✏️ Text Query"):
                            query_text_input = gr.Textbox(
                                label="Describe the audio you're looking for",
                                placeholder=(
                                    'e.g. "piano music", "someone speaking in Telugu", '
                                    '"birds chirping", "rain sounds"'
                                ),
                                lines=2,
                            )
                            text_search_btn = gr.Button(
                                "🔍 Search by Description",
                                variant="primary",
                                size="lg",
                                elem_classes=["primary-btn"],
                            )

                # ── Results ────────────────────────────────────────────
                with gr.Group(elem_classes=["glass-card"]):
                    gr.Markdown(
                        '<span class="section-title">🎯 Results</span>'
                    )
                    results_status = gr.Markdown()

                    # Pre-create 5 result slots
                    result_audios = []
                    result_labels = []
                    for i in range(5):
                        with gr.Group(
                            visible=False,
                            elem_classes=["result-card"],
                        ) as _:
                            lbl = gr.Markdown(visible=False)
                            aud = gr.Audio(
                                label=f"Result #{i+1}",
                                interactive=False,
                                visible=False,
                            )
                        result_labels.append(lbl)
                        result_audios.append(aud)

        # ── Footer ─────────────────────────────────────────────────────
        gr.HTML(
            '<div class="app-footer">'
            "Powered by <strong>CLAP</strong> embeddings  ·  "
            "<strong>ChromaDB</strong> vector store  ·  "
            "Built with <strong>Gradio</strong>"
            "</div>"
        )

        # ── Wire events ────────────────────────────────────────────────
        ingest_btn.click(fn=_ingest, inputs=upload, outputs=ingest_output)
        status_btn.click(fn=_get_status, outputs=status_output)
        clear_btn.click(fn=_clear_all, outputs=clear_output)

        # Delete source flow
        delete_dropdown.focus(fn=_get_source_choices, outputs=delete_dropdown)
        delete_btn.click(
            fn=_delete_source,
            inputs=delete_dropdown,
            outputs=[delete_output, delete_dropdown],
        )

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
