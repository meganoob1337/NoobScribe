# Attribution

## NoobScribe

NoobScribe is developed by **meganoob1337**, building on the open-source project below.

## Upstream project

This project is derived from **[parakeet-diarized](https://github.com/jfgonsalves/parakeet-diarized)** by **jfgonsalves** (Filipe Gonsalves).

Thank you to Filipe Gonsalves and contributors to the upstream project for providing a strong open-source foundation for this work.

The last upstream commit used as a reference for file provenance is **`6abadfdca15ca439ad7877db073caa7397499782`** (“readme update”).

Upstream license (project): **MIT** (see upstream repository).

Speech recognition models (e.g. **[NVIDIA NeMo](https://github.com/NVIDIA/NeMo)** checkpoints such as **[Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2)**) may be subject to **separate** terms (e.g. **CC-BY-4.0** or other model licenses). See the model card on Hugging Face for the checkpoint you use.

## File provenance (summary)

| Status | Files |
|--------|--------|
| **Unmodified** from upstream at `6abadfd` | `main.py`, `run.sh`, `tests/test_api.py`, `tests/test_chunking.py` (content was upstream; banners/strings may be updated for NoobScribe branding) |
| **Modified** from upstream | `audio.py`, `transcription.py`, `diarization/__init__.py`, `models.py`, `config.py`, `api.py`, `.gitignore` |
| **Replaced** (docs/deps evolved) | `README.md`, `requirements.txt` |
| **New** in NoobScribe (no upstream equivalent) | `database/`, `routers/`, `services/`, `webui/`, `Dockerfile`, `docker-compose.yml`, `docker-compose.traefik.yaml`, `env.example`, `DOCKER_README.md`, `API_DOCUMENTATION.md`, `AGENTS.md`, `.dockerignore`, `requirements.lock.txt` (pinned deps for Docker), etc. |

Per-file SPDX-style notices appear at the top of Python and shell files that trace to the upstream repo.

## Third-party components

| Component | Role | Links |
|-----------|------|--------|
| **NVIDIA NeMo** | ASR toolkit | [NeMo (GitHub)](https://github.com/NVIDIA/NeMo) |
| **Canary 1B v2** | Default speech recognition model | [Model card (Hugging Face)](https://huggingface.co/nvidia/canary-1b-v2) |
| **Pyannote.audio** | Speaker diarization | [Project (GitHub)](https://github.com/pyannote/pyannote-audio) · [speaker-diarization-3.1 (HF)](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| **SpeechBrain** | Spoken language identification (optional) | [SpeechBrain](https://speechbrain.github.io/) · [lang-id model (HF)](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) |
| **Chroma** | Vector store for speaker embeddings | [Chroma](https://www.trychroma.com/) · [chroma-core/chroma (GitHub)](https://github.com/chroma-core/chroma) |
| **SQLAlchemy** | Async ORM / SQLite for recordings | [SQLAlchemy](https://www.sqlalchemy.org/) |
| **Alpine.js** | WebUI frontend reactivity | [Alpine.js](https://alpinejs.dev/) · [alpinejs/alpine (GitHub)](https://github.com/alpinejs/alpine) |

Other runtime libraries are listed in **`requirements.txt`** / **`requirements.lock.txt`**. Audio I/O relies on **FFmpeg** (system dependency).
