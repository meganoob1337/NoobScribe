# Running NoobScribe with Docker

GPU-backed Docker setup for **NoobScribe** (NeMo ASR + optional pyannote diarization).

Transcription is **asynchronous**: each Compose profile runs **Redis**, an **API** container (HTTP only), and a **worker** container that consumes the job queue. See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the full design.

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/)
2. [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) for GPU
3. Hugging Face token for **remote** diarization (accept terms on [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)), **or** `DIARIZATION_MODEL_PATH` for offline pyannote (clone community-1, not 3.1)

## Pre-built image (GitHub Container Registry)

The workflow **[.github/workflows/docker-publish.yml](.github/workflows/docker-publish.yml)** builds and pushes the application image to **GHCR** (used for both API and worker containers) on pushes to the default branch (`main`), on version tags `v*`, and on manual **workflow_dispatch**. It does **not** run on pull requests.

- **Package:** `ghcr.io/meganoob1337/noobscribe` (GitHub lowercases the repository name; the GitHub repo is **NoobScribe**).
- **Tags (GPU image, [Dockerfile](Dockerfile), `linux/amd64` only):** **`latest`** on `main`; git tags like **`v1.2.3`** produce a full semver tag (e.g. **`1.2.3`**). No per-commit SHA tags.
- **Tags (CPU image, [Dockerfile.cpu](Dockerfile.cpu), `linux/amd64` only):** **`latest-cpu`** on `main`; **`v*`** releases produce **`1.2.3-cpu`**. Use **`latest-cpu`** (or a pinned version tag) on machines without an NVIDIA GPU.

**ARM hosts (Apple Silicon Macs, aarch64 Linux):** CI does **not** publish a native `linux/arm64` image. Use the **CPU** image and set **`platform: linux/amd64`** so Docker Desktop (or a setup with QEMU/binfmt) runs the amd64 image under emulation. The `cpu` profiles in **[docker-compose.yml](docker-compose.yml)** already set this. For a manual run:

```bash
docker run --platform linux/amd64 -p 8000:8000 \
  -e FORCE_CPU=1 \
  ghcr.io/meganoob1337/noobscribe:latest-cpu
```

Expect higher CPU use and slower startup than on a native amd64 machine.

Compose sets **`image`** to that registry image and keeps **`build: .`** so you can still build locally. Override the image with **`NOOBSCRIBE_IMAGE`** in `.env` (see **[env.example](env.example)**) — useful for forks (`ghcr.io/<your-user>/<your-repo>:<tag>`) or pinning a specific tag.

If the package is **private**, authenticate before pull:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u USERNAME --password-stdin
```

Use a [personal access token](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry) with `read:packages`. For a public package, no login is required.

## Build and run (Compose)

From the repository root, **pull the pre-built image** (default) or **build locally**:

```bash
cp env.example .env
docker compose up -d
```

Local build from the Dockerfile:

```bash
docker compose build
docker compose up -d
```

**CPU-only (no NVIDIA GPU):** use `--profile cpu` with **[docker-compose.yml](docker-compose.yml)** — default image **`latest-cpu`** (`linux/amd64`), **`platform: linux/amd64`** (needed on ARM so the pre-built image is used with emulation), **`Dockerfile.cpu`** for local builds, **`FORCE_CPU=1`**, no GPU device reservations.

```bash
docker compose --profile cpu up -d
```

Stop:

```bash
docker compose down
```

Each profile starts **`redis`**, the **API** container (`SKIP_MODEL_LOAD=1`, no NeMo in-process), and a matching **worker** container (`python -m workers.run`, loads ASR + diarization). Transcription requests return `202` immediately; the worker updates job status in SQLite and reports chunk progress.

The default [docker-compose.yml](docker-compose.yml) mode uses the `traefik` profile from `.env` (`COMPOSE_PROFILES=traefik`), running **GPU + Traefik** with no host port publish. Use `--profile no-traefik` (GPU) or `--profile cpu-no-traefik` (CPU) for direct host publish via `NOOBSCRIBE_PUBLISH_PORT`.

### Worker architecture (Compose)

| Service | Purpose | GPU | Exposes HTTP |
|---------|---------|-----|--------------|
| `redis` | RQ job queue (`transcription` queue) | No | No |
| `noobscribe-api` / `noobscribe-api-cpu` / `…-no-traefik` | FastAPI + Web UI; enqueues jobs | Optional (API does not run inference) | Yes (Traefik or host port) |
| `noobscribe-worker` / `noobscribe-worker-cpu` / `…-no-traefik` | Runs `python -m workers.run`; ASR + diarization | Yes (GPU profiles) or `FORCE_CPU=1` (CPU profiles) | No |

Both app containers share the **same image** and mount **`./data`** and **`./huggingface`**. Only the API publishes port 8000 (directly or via Traefik). Workers have no HTTP port; check logs with `docker compose logs -f noobscribe-worker` (or the CPU/no-traefik service name for your profile).

**Scaling workers:** start an extra worker service with the same image, env, volumes, and `command: ["/app/venv/bin/python", "-m", "workers.run"]`, or duplicate the worker service in a override file. All workers must share the same `REDIS_URL` and `./data` mount.

**Health:** `GET /health` on the API reports `queue_enabled`, `redis_ok`, `workers_registered`, and `queue_depth`. If `workers_registered` is 0, jobs remain `queued` until a worker starts.

## Manual image build

Prefer **Compose** so Redis and a worker start with the API. A single `docker run` of only the API is not enough for transcription.

GPU example (three containers, simplified — use [docker-compose.yml](docker-compose.yml) for production):

```bash
docker build -t noobscribe:latest .

docker network create noobscribe-net
docker run -d --name noobscribe-redis --network noobscribe-net \
  -v ./data/redis:/data redis:7-alpine redis-server --appendonly yes

docker run -d --name noobscribe-worker --network noobscribe-net --gpus all \
  -e REDIS_URL=redis://noobscribe-redis:6379/0 \
  -e HUGGINGFACE_ACCESS_TOKEN=your_token \
  -v ./data:/app/data -v ./huggingface:/root/.cache/huggingface \
  noobscribe:latest /app/venv/bin/python -m workers.run

docker run -d --name noobscribe-api --network noobscribe-net -p 8000:8000 \
  -e SKIP_MODEL_LOAD=1 \
  -e REDIS_URL=redis://noobscribe-redis:6379/0 \
  -e HUGGINGFACE_ACCESS_TOKEN=your_token \
  -v ./data:/app/data -v ./huggingface:/root/.cache/huggingface \
  noobscribe:latest
```

Pinned dependencies for the **CPU** image are in **`requirements.lock.txt`** ([`Dockerfile.cpu`](Dockerfile.cpu)). The **GPU** image installs from [`requirements.txt`](requirements.txt) and then force-reinstalls the PyTorch cu130 stack ([`Dockerfile`](Dockerfile)).

## Environment variables

| Variable | Description |
|----------|-------------|
| `DEBUG` | `0` or `1` |
| `ENABLE_DIARIZATION` | `true` / `false` |
| `INCLUDE_DIARIZATION_IN_TEXT` | `true` / `false` |
| `USE_API` | `1` / `true` / `yes` to use remote OpenAI-compatible STT instead of local NeMo (requires `STT_BASE_URL`, `STT_API_KEY`) |
| `STT_BASE_URL` | OpenAI-compatible API base URL (e.g. `https://api.openai.com/v1`) |
| `STT_API_KEY` | Bearer token for the STT API |
| `MODEL_ID` | `whisper-1` | **Remote API mode only** (`USE_API=1`): freeform remote `model` name. |
| `ASR_MODEL` | `canary-1b-v2` | **Local mode:** registry key from `config/asr_models.yaml`. Unknown keys fail at startup. |
| `ASR_MODELS_CONFIG` | _(unset)_ | Optional path to a custom ASR model registry YAML file. |
| `MODEL_PATH` | Local NeMo `.nemo` or ONNX directory when the selected registry entry allows override |
| `DIARIZATION_MODEL_PATH` | Local pyannote pipeline directory |
| `TEMPERATURE` | Sampling temperature |
| `CHUNK_MIN_DURATION` | Earliest pause cut after a chunk starts (seconds; default 15) |
| `CHUNK_MAX_DURATION` | Latest cut / force-split length (seconds; default 30) |
| `CHUNK_SILENCE_DB` | Silence detection noise threshold in dB (default -40) |
| `CHUNK_SILENCE_MIN_DURATION` | Minimum pause length for a cut (seconds; default 0.3) |
| `HUGGINGFACE_ACCESS_TOKEN` | HF token (diarization / downloads) |
| `RECORDINGS_PATH`, `DATABASE_URL`, `CHROMADB_PATH` | Defaults match `./data` → `/app/data` in Compose |
| `REDIS_URL` | Redis connection for the transcription queue (Compose: `redis://redis:6379/0`) |
| `QUEUE_TRANSCRIPTION` | `1` (default) enables async jobs; `0` makes enqueue endpoints return `503` |
| `SKIP_MODEL_LOAD` | `1` on API containers — workers load NeMo / diarization models |
| `LOAD_MODELS_ON_DEMAND` | When `1`/`true`/`yes`, workers (or sync API) load ASR only for transcription and unload afterward instead of keeping it warm |
| `SPEAKER_MATCH_MODE` | `nearest` (default) or `centroid`. Set on **both** API and worker. After changing, restart both then rematch transcripts. |

Set queue-related variables on **both** API and worker services when overriding Compose env.

## Volumes

Compose mounts **`./data:/app/data`**, which stores:

- SQLite (`noobscribe.db` by default) — recordings, jobs, transcription results
- `./data/recordings/` — library audio
- `./data/speakers/` — Chroma speaker index
- `./data/redis/` — Redis AOF persistence (Compose `redis` service)

Ephemeral transcoding uses `TEMP_DIR` (default `/tmp/noobscribe` inside the app config unless overridden).

API and worker containers must use the **same** `./data` mount so workers can read recordings and write job results.

## API and UI

- API: `http://localhost:8000`
- Web UI: `http://localhost:8000/ui`

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@/path/to/audio.wav \
  -F diarize=true
# => 202 { "job_id", "status": "queued" } — poll GET /v1/transcription-jobs/{job_id}
```

## Traefik (reverse proxy + TLS)

The default mode in **[docker-compose.yml](docker-compose.yml)** is Traefik-enabled. Use it when Traefik already runs on your host (or in another Compose project) and you attach services to a **shared external Docker network**.

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

From the repo root (Compose v2). Omit **`--build`** if you only want the default **GHCR** image; add **`--build`** to rebuild from the local Dockerfile.

```bash
docker compose --env-file .env up -d
# or: ... up -d --build
```

Stop:

```bash
docker compose --env-file .env down
```

- Traefik routes `https://<NOOBSCRIBE_HOST>/` to the app on container port **8000**.
- Web UI: `https://<NOOBSCRIBE_HOST>/ui`
- Optional direct access: run either `docker compose --profile no-traefik up -d` (GPU) or `docker compose --profile cpu-no-traefik up -d` (CPU), both using `NOOBSCRIBE_PUBLISH_PORT` (default `8000:8000`).

If labels don’t match your Traefik setup, adjust `TRAEFIK_ENTRYPOINT`, `TRAEFIK_CERT_RESOLVER`, or the label block in `docker-compose.yml` to match your static config.
