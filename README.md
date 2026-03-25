# NoobScribe

**NoobScribe** is a FastAPI service that exposes an **OpenAI Whisper–compatible** HTTP API for speech-to-text, powered by **[NVIDIA Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2)** via **[NVIDIA NeMo](https://github.com/NVIDIA/NeMo)** and optional **[Pyannote.audio](https://github.com/pyannote/pyannote-audio)** speaker diarization ([default pipeline on Hugging Face](https://huggingface.co/pyannote/speaker-diarization-3.1)).

Use case: run a zero-install transcription workflow from the browser - upload meeting audio or record live from mic/tab audio, then review and manage speaker-labeled transcripts in one place.

> **Attribution:** Based on **[parakeet-diarized](https://github.com/jfgonsalves/parakeet-diarized)** by [jfgonsalves](https://github.com/jfgonsalves). See **[ATTRIBUTION.md](ATTRIBUTION.md)** for file-level provenance and licensing notes.

**Author:** [meganoob1337](https://github.com/meganoob1337)

**Development note:** This project was developed using Cursor as an assistive tool, with direction from a human and additional manual edits by the author.

## Features

- Whisper-compatible `POST /v1/audio/transcriptions` (json, text, srt, vtt, verbose_json)
- Segment timestamps; optional word timestamps and diarization
- **Speaker embedding memory** ([**Chroma**](https://www.trychroma.com/)) with display names and re-matching across stored transcripts
- **Recording library** ([**SQLAlchemy**](https://www.sqlalchemy.org/) + SQLite + on-disk audio) and **Web UI** at `/ui` (transcript view: per-segment play/pause via `GET /v1/audio/snippet`, streams only on play)
- Optional **spoken language ID** ([**SpeechBrain**](https://speechbrain.github.io/) · [VoxLingua107 ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)) when `language` is omitted
- Offline options: `MODEL_PATH` ([NeMo](https://github.com/NVIDIA/NeMo) `.nemo`), `DIARIZATION_MODEL_PATH` (local [**Pyannote**](https://github.com/pyannote/pyannote-audio) pipeline)

Full endpoint reference: **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)**

## Core functionality

NoobScribe is built around a Whisper-compatible transcription API with persistent storage and speaker-aware post-processing:

- Accept audio uploads and transcribe them through NVIDIA Canary (NeMo) using a familiar Whisper-style endpoint.
- Return transcript output in multiple formats (`json`, `text`, `srt`, `vtt`, `verbose_json`) with segment timestamps.
- Optionally run speaker diarization and attach speaker embeddings to transcript results.
- Match detected speakers against a stored speaker memory (Chroma) to reuse display names across recordings.
- Store recordings and transcription metadata in SQLite, with audio files persisted on disk.
- Support language auto-detection (when `language` is omitted) and offline model paths for ASR/diarization.

## Implemented Web UI features

The built-in UI at `/ui` is a single-page app for managing recordings, transcripts, and speaker identities:

- **Recordings library:** paginated recordings list, responsive table/cards, and quick navigation to details.
- **Upload flow:** upload audio file with optional recording name and immediate redirect to the created recording.
- **Browser recording:** capture microphone and/or tab/window audio, live level meters, preview, upload, and discard.
- **Recording detail:** rename recordings, play original audio, run transcription jobs with diarization/word timestamp options, and review prior transcription runs.
- **Transcript viewer:** formatted transcript display, per-segment snippet playback, `.txt` download, and save detected speaker embeddings.
- **Speaker management:** list/delete speakers, add embeddings from snippet extraction or manual vector input, inspect embedding snippets, and delete individual embeddings.

## Quick start (Docker)

Prerequisites: [Docker](https://docs.docker.com/get-docker/), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU.

**Pre-built image:** CI publishes **`ghcr.io/meganoob1337/noobscribe`** (tags such as `latest` on the default branch and semver on `v*` git tags). Compose uses that image by default so you can start without a local build:

```bash
docker compose up -d
```

To **build from source** instead (or after changing the app or Dockerfile):

```bash
docker compose build
docker compose up -d
```

Set **`NOOBSCRIBE_IMAGE`** in `.env` if you use a fork or a pinned digest/tag (see **[DOCKER_README.md](DOCKER_README.md)**).

- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`
- Web UI: `http://localhost:8000/ui` (or open `http://localhost:8000/` — redirects to `/ui`)

Data persists under `./data` (mounted in Compose). Hugging Face cache can live under `./huggingface`.

Copy **[env.example](env.example)** to `.env` and adjust values (Compose loads `.env` automatically; never commit secrets).

See **[DOCKER_README.md](DOCKER_README.md)** for environment variables, **Traefik + TLS**, and manual `docker run` examples.

**Traefik:** Use **[docker-compose.traefik.yaml](docker-compose.traefik.yaml)** with a pre-existing Traefik stack and external Docker network. Set `NOOBSCRIBE_HOST` and `TRAEFIK_NETWORK` in `.env` (see `env.example`). Full steps in **DOCKER_README.md**.

## Local development (without Docker)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set `HUGGINGFACE_ACCESS_TOKEN` if you use Hugging Face for diarization (or set `DIARIZATION_MODEL_PATH` for offline pyannote). **Pyannote-audio** models on Hugging Face are **gated**: sign in, open each model card you need (for example the default [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) pipeline), and **accept the access terms** on the card before downloads will succeed with your token.

```bash
./run.sh --hf-token "your_token"    # optional
# or
./run.sh --port 8000
```

## Configuration (high level)

| Variable | Default | Notes |
|----------|---------|--------|
| `MODEL_ID` | `nvidia/canary-1b-v2` | [Canary](https://huggingface.co/nvidia/canary-1b-v2) / [NeMo](https://github.com/NVIDIA/NeMo) from Hugging Face |
| `MODEL_PATH` | _(unset)_ | Local [NeMo](https://github.com/NVIDIA/NeMo) `.nemo` checkpoint (offline ASR) |
| `DIARIZATION_MODEL_PATH` | _(unset)_ | Local [Pyannote](https://github.com/pyannote/pyannote-audio) pipeline dir |
| `HUGGINGFACE_ACCESS_TOKEN` | _(unset)_ | For [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) when not offline; Pyannote models are **gated** on Hugging Face — you must accept the terms on each model card (see note below) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/noobscribe.db` | Recordings metadata |
| `CHROMADB_PATH` | `./data/speakers` | [Chroma](https://www.trychroma.com/) speaker embedding store |
| `RECORDINGS_PATH` | `./data/recordings` | Uploaded audio |
| `TEMP_DIR` | `/tmp/noobscribe` | Temp transcoding |
| `CHUNK_DURATION` | `20` | Chunk length (seconds) for long files |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.7` | Cosine similarity for speaker match |

**Gated Pyannote models:** The **pyannote-audio** checkpoints used for diarization (including the default pipeline and any sub-models it pulls) are hosted on Hugging Face under **gated** repositories. Create a [Hugging Face access token](https://huggingface.co/settings/tokens), then visit each required model page while logged in and **accept the user conditions** (e.g. “Agree and access repository”). Until you do, `huggingface_hub` may return access errors even with a valid token.

## Testing

```bash
./venv/bin/python -m pip install -r requirements.txt pytest
./venv/bin/python -m pytest tests/ -q
./venv/bin/python tests/test_api.py --file /path/to/audio.wav --url http://localhost:8000
```

## License

NoobScribe project code is licensed under the **MIT License**. See **[LICENSE](LICENSE)**.

For provenance details and upstream attribution, see **[ATTRIBUTION.md](ATTRIBUTION.md)**.

**ASR and diarization models** have their own terms — see **[Canary](https://huggingface.co/nvidia/canary-1b-v2)** and **[Pyannote diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)** model cards (Canary is commonly **CC-BY-4.0**; confirm on each card you use).
