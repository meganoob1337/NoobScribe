# NoobScribe

**NoobScribe** is a FastAPI service that exposes an **OpenAI Whisper–compatible** HTTP API for speech-to-text, powered by **[NVIDIA Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2)** via **[NVIDIA NeMo](https://github.com/NVIDIA/NeMo)** and optional **[Pyannote.audio](https://github.com/pyannote/pyannote-audio)** speaker diarization ([default pipeline on Hugging Face](https://huggingface.co/pyannote/speaker-diarization-3.1)).

> **Attribution:** Based on **[parakeet-diarized](https://github.com/jfgonsalves/parakeet-diarized)** by [jfgonsalves](https://github.com/jfgonsalves). See **[ATTRIBUTION.md](ATTRIBUTION.md)** for file-level provenance and licensing notes.

**Author:** [meganoob1337](https://github.com/meganoob1337)

## Features

- Whisper-compatible `POST /v1/audio/transcriptions` (json, text, srt, vtt, verbose_json)
- Segment timestamps; optional word timestamps and diarization
- **Speaker embedding memory** ([**Chroma**](https://www.trychroma.com/)) with display names and re-matching across stored transcripts
- **Recording library** ([**SQLAlchemy**](https://www.sqlalchemy.org/) + SQLite + on-disk audio) and **Web UI** at `/ui` (transcript view: per-segment play/pause via `GET /v1/audio/snippet`, streams only on play)
- Optional **spoken language ID** ([**SpeechBrain**](https://speechbrain.github.io/) · [VoxLingua107 ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)) when `language` is omitted
- Offline options: `MODEL_PATH` ([NeMo](https://github.com/NVIDIA/NeMo) `.nemo`), `DIARIZATION_MODEL_PATH` (local [**Pyannote**](https://github.com/pyannote/pyannote-audio) pipeline)

Full endpoint reference: **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)**

## Quick start (Docker)

Prerequisites: [Docker](https://docs.docker.com/get-docker/), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU.

```bash
docker compose build
docker compose up -d
```

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

Set `HUGGINGFACE_ACCESS_TOKEN` if you use Hugging Face for diarization (or set `DIARIZATION_MODEL_PATH` for offline pyannote).

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
| `HUGGINGFACE_ACCESS_TOKEN` | _(unset)_ | For [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) when not offline |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/noobscribe.db` | Recordings metadata |
| `CHROMADB_PATH` | `./data/speakers` | [Chroma](https://www.trychroma.com/) speaker embedding store |
| `RECORDINGS_PATH` | `./data/recordings` | Uploaded audio |
| `TEMP_DIR` | `/tmp/noobscribe` | Temp transcoding |
| `CHUNK_DURATION` | `20` | Chunk length (seconds) for long files |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.7` | Cosine similarity for speaker match |

## Testing

```bash
./venv/bin/python -m pip install -r requirements.txt pytest
./venv/bin/python -m pytest tests/ -q
./venv/bin/python tests/test_api.py --file /path/to/audio.wav --url http://localhost:8000
```

## License

Project code: follow **[ATTRIBUTION.md](ATTRIBUTION.md)** and upstream **MIT** terms where they apply. **ASR and diarization models** have their own terms — see **[Canary](https://huggingface.co/nvidia/canary-1b-v2)** and **[Pyannote diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)** model cards (Canary is commonly **CC-BY-4.0**; confirm on each card you use).
