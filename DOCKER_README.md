# Running NoobScribe with Docker

GPU-backed Docker setup for **NoobScribe** (NeMo ASR + optional pyannote diarization).

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/)
2. [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) for GPU
3. Hugging Face token for **remote** diarization, **or** `DIARIZATION_MODEL_PATH` for offline pyannote

## Build and run (Compose)

From the repository root:

```bash
docker compose build
docker compose up -d
```

Stop:

```bash
docker compose down
```

The default [docker-compose.yml](docker-compose.yml) publishes **`8000:8000`** and uses **`gpus: all`**. The Traefik file pins one GPU via **`GPU_DEVICE_ID`** (see [env.example](env.example)).

## Manual image build

```bash
docker build -t noobscribe-api .
docker run --gpus all -p 8000:8000 \
  -e HUGGINGFACE_ACCESS_TOKEN=your_token \
  -v ./data:/app/data \
  noobscribe-api
```

Pinned dependencies for the image are in **`requirements.lock.txt`** (installed in the Dockerfile).

## Environment variables

| Variable | Description |
|----------|-------------|
| `DEBUG` | `0` or `1` |
| `ENABLE_DIARIZATION` | `true` / `false` |
| `INCLUDE_DIARIZATION_IN_TEXT` | `true` / `false` |
| `MODEL_ID` | NeMo model id (default `nvidia/canary-1b-v2`) |
| `MODEL_PATH` | Local NeMo `.nemo` (offline ASR) |
| `DIARIZATION_MODEL_PATH` | Local pyannote pipeline directory |
| `TEMPERATURE` | Sampling temperature |
| `CHUNK_DURATION` | Chunk length in seconds |
| `HUGGINGFACE_ACCESS_TOKEN` | HF token (diarization / downloads) |
| `RECORDINGS_PATH`, `DATABASE_URL`, `CHROMADB_PATH` | Defaults match `./data` → `/app/data` in Compose |

## Volumes

Compose mounts **`./data:/app/data`**, which stores:

- SQLite (`noobscribe.db` by default)
- `./data/recordings/` — library audio
- `./data/speakers/` — Chroma speaker index

Ephemeral transcoding uses `TEMP_DIR` (default `/tmp/noobscribe` inside the app config unless overridden).

## API and UI

- API: `http://localhost:8000`
- Web UI: `http://localhost:8000/ui`

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@/path/to/audio.wav \
  -F diarize=true
```

## Traefik (reverse proxy + TLS)

Use **[docker-compose.traefik.yaml](docker-compose.traefik.yaml)** when Traefik already runs on your host (or in another Compose project) and you attach services to a **shared external Docker network**.

### Prerequisites

1. A Traefik instance with entrypoints and (optional) a certificate resolver configured in **Traefik’s static config** (e.g. `websecure` + `letsencrypt`).
2. An **external** Docker network that the Traefik container uses — same name you set in `TRAEFIK_NETWORK`.

   ```bash
   docker network create traefik-global-proxy
   ```

   Use your real network name if Traefik uses something else.

### Configure environment

1. Copy the example file and edit:

   ```bash
   cp env.example .env
   ```

2. Set at least:

   | Variable | Purpose |
   |----------|---------|
   | `NOOBSCRIBE_HOST` | Hostname for Traefik’s `Host(\`...\`)` rule (no `https://`). Must resolve to Traefik (DNS / `/etc/hosts`). |
   | `TRAEFIK_NETWORK` | **Exact** name of the external network Traefik is on (e.g. `traefik-global-proxy`). |
   | `TRAEFIK_ENTRYPOINT` | Traefik HTTPS entrypoint name (default `websecure`). |
   | `TRAEFIK_CERT_RESOLVER` | Your ACME resolver name in Traefik (default `letsencrypt`). |
   | `GPU_DEVICE_ID` | NVIDIA GPU index for this service (`0`, `1`, …). |
   | `HUGGINGFACE_TOKEN` | If you need Hugging Face for diarization (same as default compose). |

   All of these are documented inline in **[env.example](env.example)**.

### Run

From the repo root (Compose v2):

```bash
docker compose -f docker-compose.traefik.yaml --env-file .env up -d --build
```

Stop:

```bash
docker compose -f docker-compose.traefik.yaml --env-file .env down
```

- Traefik routes `https://<NOOBSCRIBE_HOST>/` to the app on container port **8000**.
- Web UI: `https://<NOOBSCRIBE_HOST>/ui`
- Optional direct access: `NOOBSCRIBE_PUBLISH_PORT` (default `8000:8000`) still maps the host for debugging.

If labels don’t match your Traefik setup, adjust `TRAEFIK_ENTRYPOINT`, `TRAEFIK_CERT_RESOLVER`, or the label block in `docker-compose.traefik.yaml` to match your static config.
