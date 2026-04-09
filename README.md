# 🎵 Audio RAG — Pure Audio Embedding Retrieval

A Retrieval-Augmented Generation system that operates **entirely in the audio domain** — no speech-to-text anywhere in the pipeline. Audio files are chunked, embedded using [CLAP](https://huggingface.co/laion/larger_clap_music_and_speech), and retrieved via cosine similarity.

## How It Works

```
┌─────────────┐      ┌──────────┐      ┌──────────────┐      ┌─────────┐
│  Audio File │ ───▶ │  Chunk   │ ───▶ │ CLAP Encode  │ ───▶ │ChromaDB │
│  (KB Input) │      │ (10s seg)│      │ (512-dim emb)│      │  Store  │
└─────────────┘      └──────────┘      └──────────────┘      └─────────┘

┌─────────────┐      ┌──────────────┐      ┌──────────┐      ┌─────────────┐
│ Query Audio │ ───▶ │ CLAP Encode  │ ───▶ │ Cosine   │ ───▶ │ Audio       │
│  (Snippet)  │      │ (512-dim emb)│      │ Search   │      │ Snippets    │
└─────────────┘      └──────────────┘      └──────────┘      └─────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the smoke test (generates synthetic tones & verifies retrieval)
python test_smoke.py

# 3. Launch the Gradio UI
python app.py
```

The UI opens at `http://localhost:7861` with:
- **Left panel** — Upload audio files (wav, mp3, flac, ogg, m4a) to build your knowledge base
- **Right panel** — Record from mic or upload a query audio → get top matching segments as playable audio

## Project Structure

```
audio-rag/
├── app.py                  # Gradio web UI
├── config.py               # All configuration (paths, model, audio params)
├── requirements.txt        # Python dependencies
├── test_smoke.py           # Automated smoke test with synthetic tones
├── rag/
│   ├── audio_processor.py  # Load audio, chunk into segments, export WAV
│   ├── embedder.py         # CLAP model wrapper (audio → 512-dim vector)
│   ├── vector_store.py     # ChromaDB storage & similarity search
│   ├── retriever.py        # Query embedding + snippet extraction
│   └── pipeline.py         # Orchestrates ingest() and query()
├── data/                   # Ingested audio files (auto-created)
├── vectordb/               # ChromaDB persistent storage (auto-created)
└── snippets/               # Extracted result snippets (auto-created)
```

## Configuration

All settings in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `audio.segment_duration` | `10.0` s | Length of each audio chunk |
| `audio.segment_overlap` | `2.0` s | Overlap between chunks |
| `audio.sample_rate` | `48000` Hz | CLAP's expected sample rate |
| `embedding.model_name` | `laion/larger_clap_music_and_speech` | HuggingFace CLAP model |
| `retriever.top_k` | `5` | Number of results to return |

## Key Design Decisions

- **No speech-to-text** — Works purely with audio embeddings. Useful for music, environmental sounds, speaker similarity, and any domain where transcription is unwanted.
- **CLAP embeddings** — Contrastive Language-Audio Pretraining model maps audio to a shared embedding space. While it supports text too, we use **only the audio encoder**.
- **Fixed-duration chunking** — Simple and robust. Audio is split into 10-second segments with 2-second overlap. Trailing segments shorter than 1 second are discarded.
- **ChromaDB** — Lightweight, persistent vector store. No external database required.
