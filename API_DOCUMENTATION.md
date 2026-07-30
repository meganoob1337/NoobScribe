# API Documentation

## Base URL

The API is typically available at `http://localhost:8000` (default port). The base URL can be configured via the `--port` option when starting the server.

**Root path:** `GET /` returns **302** to `/ui` (Web UI). Not listed in OpenAPI (`/docs`).

## Authentication

Currently, the API does not require authentication.

## Content Types

- **Multipart Form Data**: Used for file uploads (audio transcription)
- **JSON**: Used for JSON request/response bodies (speaker management)

## Async transcription jobs

Transcription work runs in a **background worker** (Redis + [RQ](https://python-rq.org/)), not in the API process. All transcription entry points return **`202 Accepted`** immediately with a job id; poll for status and results.

**Queue a job** → one of:

- `POST /v1/audio/transcriptions`
- `POST /v1/recordings/{recording_id}/transcribe`
- `POST /v1/speakers/extract-from-audio`

**202 response:**

```json
{
  "job_id": "uuid",
  "status": "queued",
  "recording_id": "uuid-or-null"
}
```

**Poll status:** `GET /v1/transcription-jobs/{job_id}`

```json
{
  "job_id": "uuid",
  "job_type": "recording_transcribe",
  "status": "queued",
  "recording_id": "uuid",
  "transcription_id": null,
  "progress_stage": "transcribing",
  "progress_current": 3,
  "progress_total": 10,
  "error_message": null,
  "result": null,
  "created_at": "2026-06-10T12:00:00+00:00",
  "started_at": "2026-06-10T12:00:01+00:00",
  "completed_at": null
}
```

**Status values:** `queued`, `running`, `completed`, `failed`, `cancelled`.

**Progress stages:** `preparing`, `loading_diarization_model`, `diarization`, `loading_transcription_model`, `transcribing`, `completed`, `cancelled`.

- `loading_diarization_model` — pyannote pipeline is being loaded (before inference). Emitted when diarization is enabled for the job.
- `diarization` — speaker diarization inference is running.
- `loading_transcription_model` — local ASR is being loaded. Only emitted when `LOAD_MODELS_ON_DEMAND` is enabled (warm workers skip this stage).
- `transcribing` — ASR chunks are running; `progress_current` / `progress_total` are the chunk index and chunk count.

**TranscriptionJobResponse** also includes:

- `recording_name` (string, optional): display name when `recording_id` is set
- `cancellable` (boolean): `true` when `status` is `queued` or `running`

When `status` is `completed`:

- **Stored recording jobs** (`recording_transcribe`, `speaker_extract`): `transcription_id` is set; `result` contains the verbose transcript payload (same shape as the former synchronous response, including `stored: true` for library transcriptions).
- **OpenAI-style upload** (`openai_transcribe`): `result` contains the formatted transcript for the requested `response_format` (no SQLite row).

**List jobs for a recording:** `GET /v1/recordings/{recording_id}/transcription-jobs`

**List all visible jobs:** `GET /v1/transcription-jobs?limit=50&offset=0`

Returns queued and running jobs, plus completed/failed/cancelled jobs **only while the linked RQ job metadata still exists in Redis** (same TTL as enqueue: 24h by default). Older finished jobs drop off the list automatically.

**Cancel a job:** `POST /v1/transcription-jobs/{job_id}/cancel`

Only `queued` and `running` jobs can be cancelled (`409` otherwise). Sets `status` to `cancelled`. Running jobs stop cooperatively between pipeline chunks (SimpleWorker).

**Reconcile stuck queued jobs:** `POST /v1/transcription-jobs/reconcile?stale_seconds=60`

Idempotent recovery for jobs stuck in `queued` when Redis/RQ work was lost or finished without updating SQLite. Runs automatically on API startup. Query param `stale_seconds` (minimum 30) controls how long a job must remain queued before re-enqueue is attempted.

**Response:** `{ "examined", "re_enqueued", "skipped", "failed" }`

**Health:** `GET /health` includes `queue_enabled`, `redis_ok`, `workers_registered`, and `queue_depth`.

**Troubleshooting stuck jobs:** If jobs stay `queued` with `started_at` null, check `workers_registered` in `/health` (must be ≥ 1). Inspect worker logs for `Job … not found`. Re-run recovery with `POST /v1/transcription-jobs/reconcile`. Redis queue depth: `LLEN rq:queue:transcription`.

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection for the job queue |
| `QUEUE_TRANSCRIPTION` | `1` | When `0`, transcription enqueue returns `503` |
| `SKIP_MODEL_LOAD` | unset (`1` in Docker API) | API container skips loading NeMo; workers load the model |

## Endpoints

### 1. Transcribe Audio

Queue transcription of an uploaded audio file with optional speaker diarization. **ASR backend:** by default a worker uses a **configured local ASR model** (`ASR_MODEL` registry key, default NeMo Canary 1B v2). When **`USE_API=1`**, each audio chunk is sent to a **remote OpenAI-compatible** `POST …/audio/transcriptions` endpoint (`STT_BASE_URL` + `STT_API_KEY`); **`MODEL_ID`** is passed as the remote `model` name. Diarization, language ID, and merging behavior are unchanged in both modes.

**Endpoint:** `POST /v1/audio/transcriptions`

**Status:** `202 Accepted` — returns `{ "job_id", "status": "queued" }`. Poll `GET /v1/transcription-jobs/{job_id}` for the transcript in `result` when `status` is `completed`.

**Content-Type:** `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | The audio file to transcribe (supports common audio formats: wav, mp3, m4a, flac, etc.) |
| `model` | string | No | `"whisper-1"` | Model identifier (accepted for compatibility). **Local mode:** server uses configured **`ASR_MODEL`** registry entry (see `config/asr_models.yaml`). **API mode (`USE_API`):** the form field is still accepted; the server uses **`MODEL_ID`** as the remote transcription `model` parameter. |
| `language` | string | No | `null` | Language code of the audio (e.g., "en", "de", "fr"). If omitted or blank, the service runs [SpeechBrain VoxLingua107 ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) on up to three segments (start, middle, 75%) of up to `LANGUAGE_ID_MAX_AUDIO_SECONDS` each and uses the majority label (first segment on a three-way tie). Shorter clips use fewer samples (one full clip when the file is ≤ segment length or too short for distinct windows). That code is passed as both NeMo `source_lang` and `target_lang`. If identification is disabled or fails, both are left unset (previous behavior). Set `DISABLE_LANGUAGE_ID=1` to skip detection. |
| `prompt` | string | No | `null` | Optional prompt to guide the transcription (accepted for compatibility but currently ignored) |
| `response_format` | string | No | `"json"` | Format of the response. Options: `"json"`, `"text"`, `"srt"`, `"vtt"`, `"verbose_json"` |
| `temperature` | float | No | `0.0` | Temperature for sampling (0.0 to 1.0). Lower values make output more deterministic. |
| `timestamps` | boolean | No | `false` | Whether to include timestamps in the response (for JSON/verbose_json formats) |
| `timestamp_granularities` | array[string] | No | `null` | Timestamp detail level. Accepts `["segment"]` to include segment-level timestamps |
| `vad_filter` | boolean | No | `false` | Voice activity detection filter (accepted for compatibility) |
| `word_timestamps` | boolean | No | `false` | Whether to include word-level timestamps in segments |
| `diarize` | boolean | No | `true` | Enable speaker diarization (requires HuggingFace token **or** local `DIARIZATION_MODEL_PATH`; see Offline models) |
| `include_diarization_in_text` | boolean | No | `null` | Include speaker labels in transcript text. If `null`, uses server configuration default. |

#### Response Formats

##### JSON (default)

**Content-Type:** `application/json`

```json
{
  "text": "Full transcription text goes here",
  "speakers": [
    {
      "id": "SPEAKER_00",
      "display_name": "John Doe",
      "embedding": [0.1, 0.2, 0.3, ...],
      "matched": true,
      "match_similarity": 0.9124
    }
  ]
}
```

**Response Schema:**
- `text` (string): The full transcription text
- `speakers` (array, optional): Array of speaker information (only included if diarization is enabled and speakers are detected)
  - `id` (string): Speaker identifier (e.g., "SPEAKER_00")
  - `display_name` (string, optional): Display name if matched to a stored speaker
  - `embedding` (array[float]): Speaker embedding vector
  - `matched` (boolean): Whether speaker was matched to a stored identity
  - `match_similarity` (float, optional): Cosine similarity to the best stored speaker match (present when `matched` is true)

##### Verbose JSON

**Content-Type:** `application/json`

```json
{
  "text": "Full transcription text goes here",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, this is a transcription segment.",
      "tokens": [1234, 5678, ...],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.1,
      "speaker": "John Doe"
    }
  ],
  "language": "en",
  "task": "transcribe",
  "duration": 120.5,
  "model": "canary-1b-v2",
  "speakers": [
    {
      "id": "SPEAKER_00",
      "display_name": "John Doe",
      "embedding": [0.1, 0.2, 0.3, ...],
      "matched": true,
      "match_similarity": 0.9124
    }
  ]
}
```

**Response Schema:**
- `text` (string): The full transcription text
- `segments` (array): Array of transcription segments
  - `id` (integer): Segment identifier
  - `seek` (integer): Seek offset
  - `start` (float): Start time in seconds
  - `end` (float): End time in seconds
  - `text` (string): Transcribed text for this segment
  - `tokens` (array[integer]): Token IDs
  - `temperature` (float): Temperature used for this segment
  - `avg_logprob` (float): Average log probability
  - `compression_ratio` (float): Compression ratio
  - `no_speech_prob` (float): Probability of no speech
  - `speaker` (string, optional): Speaker label or display name (if diarization enabled)
- `language` (string, optional): Detected or specified language code
- `task` (string): Always "transcribe"
- `duration` (float, optional): Estimated duration in seconds
- `model` (string): Model identifier used
- `speakers` (array, optional): Array of speaker information (same as JSON format)

##### Text

**Content-Type:** `text/plain`

Returns only the transcription text as plain text.

##### SRT (SubRip)

**Content-Type:** `text/plain`

Returns transcription in SRT subtitle format with timestamps.

Example:
```
1
00:00:00,000 --> 00:00:05,200
Hello, this is a transcription segment.

2
00:00:05,200 --> 00:00:10,500
This is the next segment.
```

##### VTT (WebVTT)

**Content-Type:** `text/plain`

Returns transcription in WebVTT subtitle format with timestamps.

Example:
```
WEBVTT

00:00:00.000 --> 00:00:05.200
Hello, this is a transcription segment.

00:00:05.200 --> 00:00:10.500
This is the next segment.
```

#### Example Request (cURL)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@/path/to/your/audio.wav \
  -F model=whisper-1 \
  -F response_format=verbose_json \
  -F timestamps=true \
  -F diarize=true \
  -F include_diarization_in_text=true
```

#### Example Request (Python)

```python
import requests

url = "http://localhost:8000/v1/audio/transcriptions"
files = {"file": open("audio.wav", "rb")}
data = {
    "model": "whisper-1",
    "response_format": "verbose_json",
    "timestamps": "true",
    "diarize": "true",
    "include_diarization_in_text": "true"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Status Codes

- `200 OK`: Transcription successful
- `400 Bad Request`: Invalid request parameters or unsupported response format
- `500 Internal Server Error`: Error during transcription processing
- `503 Service Unavailable`: Model not loaded yet

---

### 2. Health Check

Check the health status of the API and the loaded model.

**Endpoint:** `GET /health`

**Content-Type:** `application/json`

#### Response Schema

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "speaker_db_initialized": true,
  "model_id": "canary-1b-v2",
  "asr_model": "canary-1b-v2",
  "asr_backend": "nemo",
  "force_cpu": false,
  "cuda_available": true,
  "gpu_info": "NVIDIA GeForce RTX 3090",
  "queue_enabled": true,
  "redis_ok": true,
  "workers_registered": 1,
  "queue_depth": 0,
  "config": {
    "model_id": "canary-1b-v2",
    "asr_model": "canary-1b-v2",
    "asr_backend": "nemo",
    "asr_model_display_name": "NVIDIA Canary 1B v2 (NeMo)",
    "available_local_models": ["canary-1b-v2", "canary-1b-v2-onnx-int8", "canary-1b-v2-onnx"],
    "chunk_min_duration": 15.0,
    "chunk_max_duration": 30.0,
    "chunk_silence_db": -40.0,
    "chunk_silence_min_duration": 0.3,
    "temp_dir": "/tmp/noobscribe",
    "chromadb_path": "./data/speakers",
    "model_path": null,
    "diarization_model_path": null,
    "recordings_path": "./data/recordings",
    "database_url": "sqlite+aiosqlite:///./data/noobscribe.db",
    "speaker_similarity_threshold": 0.7,
    "include_diarization_in_text": true,
    "force_cpu": false
  }
}
```

**Response Fields:**
- `status` (string): Always "ok" if the API is running
- `version` (string): API version
- `model_loaded` (boolean): Whether transcription is ready: **true** if the local ASR model is loaded, **`USE_API`** is enabled (remote STT; no local ASR weights), or **`LOAD_MODELS_ON_DEMAND`** is set on a sync API process (model loads per job). With the Docker queue setup (`SKIP_MODEL_LOAD` / workers), readiness is mainly indicated by `workers_registered`.
- `speaker_db_initialized` (boolean): Whether the Chroma speaker embedding store is available. The health check attempts lazy initialization if startup failed; **false** means speaker CRUD, rematch, and nearest-embedding endpoints return **503** (see server logs and `CHROMADB_PATH`).
- `model_id` (string): Active transcription model name — **`ASR_MODEL`** registry key in local mode; **`MODEL_ID`** in remote API mode
- `asr_model` (string, optional): Local registry key when not in API mode; `null` when `USE_API=1`
- `asr_backend` (string, optional): `nemo` or `onnx` for the configured local model; `null` in API mode
- `force_cpu` (boolean): Whether `FORCE_CPU` is set, forcing CPU inference even if a GPU is present
- `cuda_available` (boolean): Whether the server will use CUDA/GPU for inference (false when `force_cpu` is true or no GPU is available)
- `gpu_info` (string, optional): GPU name if CUDA is used for inference, `null` otherwise
- `queue_enabled` (boolean): Whether async transcription enqueue is enabled (`QUEUE_TRANSCRIPTION`)
- `redis_ok` (boolean): Whether Redis accepts a ping at `REDIS_URL`
- `workers_registered` (integer): RQ workers listening on the `transcription` queue
- `queue_depth` (integer): Jobs waiting in the Redis transcription queue
- `config` (object): Current server configuration (includes `force_cpu`, `use_api`, `stt_base_url`, `has_stt_api_key`, `load_models_on_demand`)

#### Example Request

```bash
curl http://localhost:8000/health
```

#### Status Codes

- `200 OK`: API is healthy

---

### 3. List Models

List available models (compatibility endpoint with OpenAI API). In **local mode**, returns one entry per key in `config/asr_models.yaml`. In **`USE_API`** mode, returns the configured remote **`MODEL_ID`**.

**Endpoint:** `GET /v1/models`

**Content-Type:** `application/json`

#### Response Schema

```json
{
  "object": "list",
  "data": [
    {
      "id": "canary-1b-v2",
      "object": "model",
      "created": 1677649963,
      "owned_by": "noobscribe",
      "root": "canary-1b-v2",
      "parent": null,
      "permission": [
        {
          "id": "modelperm-1",
          "object": "model_permission",
          "created": 1677649963,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

**Response Fields:**
- `object` (string): Always "list"
- `data` (array): Array of model information objects
  - `id` (string): Model identifier
  - `object` (string): Always "model"
  - `created` (integer): Unix timestamp of model creation
  - `owned_by` (string): Owner identifier
  - `root` (string): Root model identifier
  - `parent` (string, optional): Parent model identifier
  - `permission` (array): Array of permission objects

#### Example Request

```bash
curl http://localhost:8000/v1/models
```

#### Status Codes

- `200 OK`: Request successful

---

### 4. List Speakers

Get all speakers stored in the database.

**Endpoint:** `GET /v1/speakers`

**Content-Type:** `application/json`

#### Response Schema

```json
{
  "object": "list",
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "display_name": "John Doe",
      "created_at": "2024-01-01T00:00:00",
      "embedding_count": 2
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "display_name": "Jane Smith",
      "created_at": "2024-01-02T00:00:00",
      "embedding_count": 1
    }
  ]
}
```

**Response Fields:**
- `object` (string): Always "list"
- `data` (array): Array of speaker information objects
  - `id` (string): Unique speaker identifier (UUID)
  - `display_name` (string): Display name of the speaker
  - `created_at` (string): ISO 8601 timestamp of creation
  - `embedding_count` (integer): Number of embeddings stored for this speaker

#### Example Request (cURL)

```bash
curl http://localhost:8000/v1/speakers
```

#### Example Request (Python)

```python
import requests

url = "http://localhost:8000/v1/speakers"
response = requests.get(url)
print(response.json())
```

#### Status Codes

- `200 OK`: Request successful
- `500 Internal Server Error`: Failed to retrieve speakers
- **503 Service Unavailable**: Speaker database not initialized

---

### Re-match stored transcriptions

Re-run speaker matching for every stored transcription that has diarization `speakers` JSON. Updates persisted `display_name`, `matched`, and `match_similarity` on each speaker row with an `embedding` (same logic as the side effect after `POST` / `PUT` / `DELETE` on `/v1/speakers`).

**Endpoint:** `POST /v1/speakers/rematch`

**Content-Type:** `application/json`

#### Response Schema

```json
{
  "updated_transcriptions": 3
}
```

**Response Fields:**
- `updated_transcriptions` (integer): Number of transcription rows whose `speakers_json` was changed

#### Status Codes

- `200 OK`: Rematch completed (including when zero rows changed)
- `500 Internal Server Error`: Rematch failed
- `503 Service Unavailable`: Speaker database not initialized

---

### Nearest stored embeddings

Return the nearest stored speaker **embeddings** (individual Chroma rows) to a query vector, regardless of whether they meet `SPEAKER_SIMILARITY_THRESHOLD`.

**Endpoint:** `POST /v1/speakers/nearest-embeddings`

**Content-Type:** `application/json`

#### Request Body Schema

```json
{
  "embedding": [0.1, 0.2, 0.3],
  "limit": 3
}
```

**Request Fields:**
- `embedding` (array[float], required): Query embedding vector (non-empty)
- `limit` (integer, optional): Number of hits to return (default `3`, clamped to 1–10)

#### Response Schema

```json
{
  "threshold": 0.7,
  "data": [
    {
      "speaker_id": "550e8400-e29b-41d4-a716-446655440000",
      "display_name": "John Doe",
      "embedding_index": 0,
      "similarity": 0.9124,
      "meets_threshold": true
    }
  ]
}
```

**Response Fields:**
- `threshold` (float): Configured `SPEAKER_SIMILARITY_THRESHOLD`
- `data` (array): Nearest hits, best match first
  - `speaker_id` (string): Stored speaker UUID
  - `display_name` (string): Enrolled speaker name
  - `embedding_index` (integer): Which enrollment embedding matched
  - `similarity` (float): Cosine similarity (0–1)
  - `meets_threshold` (boolean): Whether `similarity >= threshold`

#### Status Codes

- `200 OK`: Query successful (empty `data` when no enrollments exist)
- `400 Bad Request`: Missing or invalid embedding
- `500 Internal Server Error`: Query failed
- `503 Service Unavailable`: Speaker database not initialized

---

### 5. Extract speaker embeddings from audio

Upload a short audio clip: the server stores it as a **recording** with `hide_in_recordings=true` (so it does **not** appear in `GET /v1/recordings`), then **queues** the same transcription + optional diarization pipeline as stored recording transcribe. When the job completes, a `TranscriptionResult` is persisted and **`result`** contains diarized **speaker rows with `embedding` vectors** (for enrolling via `POST`/`PUT` `/v1/speakers`). Persisted data enables embedding snippet previews (`GET /v1/speakers/{id}/embeddings/{index}/snippets`) the same way as normal library recordings.

**Endpoint:** `POST /v1/speakers/extract-from-audio`

**Status:** `202 Accepted` — `{ "job_id", "status": "queued", "recording_id" }`. Poll the job endpoint for `result` when complete.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Audio file |
| `name` | string | No | Stored recording display name |
| `language` | string | No | Same as `POST /v1/recordings/{id}/transcribe` |
| `response_format` | string | No | Default `verbose_json` (`json`, `text`, `srt`, `vtt`, `verbose_json`) |
| `temperature` | float | No | `0.0` (accepted for API parity) |
| `timestamps` | boolean | No | Default `true` |
| `word_timestamps` | boolean | No | Default `false` |
| `diarize` | boolean | No | Default `true` |
| `include_diarization_in_text` | boolean | No | Optional |

**Completed job `result` (json / verbose_json):** Same fields as a successful stored `POST /v1/recordings/{recording_id}/transcribe` response, plus **`recording_id`** (UUID of the hidden recording).

**Status codes:** `202` (queued), `400`, `500`, `503` (Redis unavailable or queue disabled).

---

### 6. Create Speaker

Create a new speaker with a display name and initial embedding.

**Endpoint:** `POST /v1/speakers`

**Content-Type:** `application/json`

#### Request Body Schema

```json
{
  "display_name": "John Doe",
  "embedding": [0.1, 0.2, 0.3, 0.4, ...]
}
```

**Request Fields:**
- `display_name` (string, required): Human-readable name for the speaker
- `embedding` (array[float], required): Speaker embedding vector (typically 512 or 1024 dimensions)

**Side effect:** After a successful create, the server **re-matches** every stored recording transcription that has a `speakers` array: for each diarized row with an `embedding`, it runs the same similarity check as live transcription (`SPEAKER_SIMILARITY_THRESHOLD`) against the Chroma speaker index and **updates persisted** `display_name`, `matched`, and `match_similarity` in SQLite. No extra request fields are required. This can add latency when many transcripts exist (one vector query per diarized speaker row).

#### Response Schema

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "display_name": "John Doe",
  "created_at": "2024-01-01T00:00:00",
  "embedding_count": 1
}
```

**Response Fields:**
- `id` (string): Unique speaker identifier (UUID)
- `display_name` (string): Display name of the speaker
- `created_at` (string): ISO 8601 timestamp of creation
- `embedding_count` (integer): Number of embeddings stored for this speaker

#### Example Request (cURL)

```bash
curl -X POST http://localhost:8000/v1/speakers \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "John Doe",
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
  }'
```

#### Example Request (Python)

```python
import requests

url = "http://localhost:8000/v1/speakers"
data = {
    "display_name": "John Doe",
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]  # Your embedding vector
}

response = requests.post(url, json=data)
print(response.json())
```

#### Status Codes

- `200 OK`: Speaker created successfully
- `500 Internal Server Error`: Failed to create speaker
- `503 Service Unavailable`: Speaker database not initialized

---

### 7. Update Speaker

Add an additional embedding to an existing speaker. This helps improve speaker matching accuracy by providing more reference embeddings.

**Endpoint:** `PUT /v1/speakers/{speaker_id}`

**Content-Type:** `application/json`

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `speaker_id` | string | Yes | UUID of the speaker to update |

#### Request Body Schema

```json
{
  "embedding": [0.15, 0.25, 0.35, 0.45, ...]
}
```

**Request Fields:**
- `embedding` (array[float], required): Additional speaker embedding vector (must match dimensions of existing embeddings)

**Side effect:** Same global **re-match** of all stored transcription `speakers` metadata as `POST /v1/speakers` (see above).

#### Response Schema

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "display_name": "John Doe",
  "created_at": "2024-01-01T00:00:00",
  "embedding_count": 2
}
```

**Response Fields:**
- `id` (string): Speaker identifier (same as path parameter)
- `display_name` (string): Display name of the speaker
- `created_at` (string): ISO 8601 timestamp of creation
- `embedding_count` (integer): Updated count of embeddings (incremented by 1)

#### Example Request (cURL)

```bash
curl -X PUT http://localhost:8000/v1/speakers/550e8400-e29b-41d4-a716-446655440000 \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.15, 0.25, 0.35, 0.45, 0.55]
  }'
```

#### Example Request (Python)

```python
import requests

speaker_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:8000/v1/speakers/{speaker_id}"
data = {
    "embedding": [0.15, 0.25, 0.35, 0.45, 0.55]  # Additional embedding vector
}

response = requests.put(url, json=data)
print(response.json())
```

#### Status Codes

- `200 OK`: Speaker updated successfully
- `404 Not Found`: Speaker with the given ID not found
- `500 Internal Server Error`: Failed to update speaker
- `503 Service Unavailable`: Speaker database not initialized

---

### 6a. List speaker embeddings

List metadata for each enrolled embedding (Chroma row) for a speaker. Raw vectors are not returned.

**Endpoint:** `GET /v1/speakers/{speaker_id}/embeddings`

#### Path parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `speaker_id` | string | Yes | UUID of the speaker |

#### Response schema

```json
{
  "speaker_id": "550e8400-e29b-41d4-a716-446655440000",
  "display_name": "John Doe",
  "data": [
    { "embedding_index": 0, "created_at": "2024-01-01T00:00:00" },
    { "embedding_index": 1, "created_at": "2024-01-02T00:00:00" }
  ]
}
```

#### Status codes

- `200 OK`: Success
- `404 Not Found`: Speaker not found
- `503 Service Unavailable`: Speaker database not initialized

---

### 6b. List matching transcription snippets for an embedding

For one enrolled embedding, scan stored transcriptions that have diarization `speakers` JSON. For each transcription, pick the diarization speaker row whose **embedding** has the highest cosine similarity to the enrolled vector; if that similarity is ≥ `SPEAKER_SIMILARITY_THRESHOLD`, find the **first** transcript segment whose `speaker` matches that diarization label and return a `preview_url` for on-demand audio (see **6d**).

**Endpoint:** `GET /v1/speakers/{speaker_id}/embeddings/{embedding_index}/snippets`

#### Path parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `speaker_id` | string | Yes | UUID of the speaker |
| `embedding_index` | integer | Yes | Chroma enrollment index (e.g. `0`, `1`, …) |

#### Response schema

```json
{
  "object": "list",
  "data": [
    {
      "transcription_id": "…",
      "recording_id": "…",
      "recording_name": "Meeting.wav",
      "preview_url": "/v1/audio/snippet?recording_id=…&start=1.2&end=4.5",
      "segment_text": "Hello everyone.",
      "start": 1.2,
      "end": 4.5
    }
  ]
}
```

#### Status codes

- `200 OK`: Success (may be an empty `data` array)
- `404 Not Found`: Speaker or embedding index not found
- `503 Service Unavailable`: Speaker database not initialized

---

### 6c. Delete one speaker embedding

**Endpoint:** `DELETE /v1/speakers/{speaker_id}/embeddings/{embedding_index}`

Removes a single enrollment from Chroma. **Cannot** delete the last remaining embedding; use **Delete speaker** instead.

**Side effect:** On success, the server runs the same global **re-match** of stored transcription `speakers` metadata as `POST` / `PUT` `/v1/speakers`.

#### Status codes

- `200 OK`: Embedding removed
- `400 Bad Request`: Would remove the only embedding
- `404 Not Found`: Speaker or embedding index not found
- `503 Service Unavailable`: Speaker database not initialized

---

### 6d. Stream an audio snippet from a recording

**Endpoint:** `GET /v1/audio/snippet`

**Query parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `recording_id` | string | Yes | Library recording UUID |
| `start` | float | Yes | Start time in seconds |
| `end` | float | Yes | End time in seconds |

Returns `audio/wav` (16 kHz mono PCM) for the half-open interval, generated with ffmpeg. Segment length is capped (default **300** seconds). The temporary file is deleted after the response completes.

#### Status codes

- `200 OK` / `206 Partial Content`: Audio body (standard `FileResponse`; clients may request byte ranges)
- `400 Bad Request`: Invalid time range or segment too long
- `404 Not Found`: Recording or file on disk missing
- `500 Internal Server Error`: ffmpeg or server error

---

### 8. Delete Speaker

Delete a speaker and all its embeddings from the database.

**Endpoint:** `DELETE /v1/speakers/{speaker_id}`

**Content-Type:** `application/json`

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `speaker_id` | string | Yes | UUID of the speaker to delete |

#### Response Schema

```json
{
  "message": "550e8400-e29b-41d4-a716-446655440000 deleted successfully"
}
```

**Response Fields:**
- `message` (string): Confirmation message indicating successful deletion

#### Example Request (cURL)

```bash
curl -X DELETE http://localhost:8000/v1/speakers/550e8400-e29b-41d4-a716-446655440000
```

#### Example Request (Python)

```python
import requests

speaker_id = "550e8400-e29b-41d4-a716-446655440000"
url = f"http://localhost:8000/v1/speakers/{speaker_id}"

response = requests.delete(url)
print(response.json())
```

#### Status Codes

- `200 OK`: Speaker deleted successfully
- `404 Not Found`: Speaker with the given ID not found
- `500 Internal Server Error`: Failed to delete speaker
- `503 Service Unavailable`: Speaker database not initialized

---

## Recording management

Persisted audio files, SQLite metadata, and stored transcription/diarization results. Files live under `RECORDINGS_PATH` (default `./data/recordings`).

### 9. List recordings

**Endpoint:** `GET /v1/recordings`

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | `50` | Page size (max 200) |
| `offset` | integer | `0` | Offset |

**Response:** `{ "object": "list", "data": [ RecordingResponse ], "has_more": boolean }`

Only recordings with **`hide_in_recordings` false** are returned (enrollment-only clips from **Extract speaker embeddings from audio** stay hidden).

Each **RecordingResponse** item: `id`, `name`, `original_filename`, `stored_filename`, `duration_seconds`, `file_size_bytes`, `mime_type`, `hide_in_recordings`, `created_at`, `updated_at`, `transcription_count`.

---

### 10. Create recording (upload)

**Endpoint:** `POST /v1/recordings`

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Audio file |
| `name` | string | No | Display name |
| `hide_in_recordings` | boolean | No | Default `false`. If `true`, the recording is omitted from `GET /v1/recordings` but can still be fetched by id, transcribed, and used for audio snippets. |

**Response:** `RecordingResponse` (`201` implicit via `200` with body).

---

### 11. Get recording (with transcription history)

**Endpoint:** `GET /v1/recordings/{recording_id}`

**Response:** **RecordingDetailResponse** — same fields as `RecordingResponse` plus `transcriptions`: array of **TranscriptionResultResponse** (newest first).

---

### 12. Update recording name

**Endpoint:** `PATCH /v1/recordings/{recording_id}`

**Content-Type:** `application/json`

```json
{ "name": "My meeting" }
```

**Response:** `RecordingResponse`-shaped metadata (count only, no embedded list).

---

### 13. Delete recording

**Endpoint:** `DELETE /v1/recordings/{recording_id}`

Deletes the row, cascades stored transcriptions, and removes the audio file from disk.

**Response:** `{ "message": "<id> deleted" }`

---

### 14. Download recording audio

**Endpoint:** `GET /v1/recordings/{recording_id}/audio`

**Response:** Raw file stream. **`Range` is not honored** (always full file, `200 OK`, `Accept-Ranges: none`) so in-browser `<audio>` controls show correct duration. **`GET /v1/audio/snippet`** continues to use normal ranged file delivery (`200` or `206` when clients send `Range`).

---

### 15. Transcribe stored recording

**Endpoint:** `POST /v1/recordings/{recording_id}/transcribe`

**Content-Type:** `multipart/form-data`

**Status:** `202 Accepted` — `{ "job_id", "status": "queued", "recording_id" }`.

Same form fields as `POST /v1/audio/transcriptions` except **no** `file`: `language`, `response_format`, `temperature`, `timestamps`, `word_timestamps`, `diarize`, `include_diarization_in_text`.

When the job completes, the worker persists a **TranscriptionResult** (segments + speakers JSON). For `json` / `verbose_json`, **`result`** includes:

- `transcription_id` (string): ID of the stored row
- `stored` (boolean): `true`

Other formats (`text`, `srt`, `vtt`) place plain text in `result.text` (and `result.format` for srt/vtt).

---

### 15b. List transcription jobs for a recording

**Endpoint:** `GET /v1/recordings/{recording_id}/transcription-jobs`

**Response:** `{ "object": "list", "data": [ TranscriptionJobResponse ] }`

---

### 16. List all transcription jobs

**Endpoint:** `GET /v1/transcription-jobs`

**Query:** `limit` (default 50, max 200), `offset` (default 0)

**Response:** `{ "object": "list", "data": [ TranscriptionJobResponse ] }` — filtered to [visible jobs](#async-transcription-jobs) (active + historic while in Redis).

---

### 17. Get transcription job status

**Endpoint:** `GET /v1/transcription-jobs/{job_id}`

**Response:** `TranscriptionJobResponse` (see [Async transcription jobs](#async-transcription-jobs)).

---

### 18. Cancel transcription job

**Endpoint:** `POST /v1/transcription-jobs/{job_id}/cancel`

**Response:** Updated `TranscriptionJobResponse` with `status: "cancelled"`.

**Status codes:** `200`, `404`, `409` (not cancellable).

---

### 18b. Reconcile stuck transcription jobs

**Endpoint:** `POST /v1/transcription-jobs/reconcile`

**Query:** `stale_seconds` (default 60, minimum 30)

**Response:**

```json
{
  "examined": 2,
  "re_enqueued": 1,
  "skipped": 1,
  "failed": 0
}
```

Re-enqueues `queued` jobs older than `stale_seconds` when the linked RQ job is missing or already finished/failed. Skips jobs still waiting or running in Redis.

---

### 19. List transcriptions for a recording

**Endpoint:** `GET /v1/recordings/{recording_id}/transcriptions`

**Response:** `{ "object": "list", "data": [ TranscriptionResultResponse ] }`

**TranscriptionResultResponse:** `id`, `recording_id`, `full_text`, `segments` (JSON array), `language`, `model_id`, `diarization_enabled`, `speakers` (JSON array), `duration_seconds`, `word_timestamps`, `created_at`.

---

### 20. Get one stored transcription

**Endpoint:** `GET /v1/recordings/{recording_id}/transcriptions/{transcription_id}`

**Response:** `TranscriptionResultResponse`

---

## Web UI

A minimal SPA is served at **`/ui`** (same host as the API). Open `/ui` in a browser to upload recordings, **record in the browser** (mix window/tab or screen capture audio with the microphone, then upload as WebM), run transcription/diarization, inspect segments, save speaker embeddings to `/v1/speakers`, and manage stored speakers. On the **transcript viewer**, diarized **Speakers** use expandable accordion rows (segment count, matched similarity, **≈ nearest** popover for top stored embedding hits vs threshold, per-speaker segment snippets with audio play) and a **Re-match transcripts** control (`POST /v1/speakers/rematch`). On **Speakers**, **From audio snippet** uploads or records a short clip, calls **`POST /v1/speakers/extract-from-audio`** (hidden stored recording + diarization), then lets you **Save as…** for each detected speaker like the transcript view. The **Speakers** page header also offers **Re-match transcripts** and uses expandable rows: each speaker lists **enrollments** (embedding index), matching **transcriptions** with an **Audio snippet** control (loads `GET /v1/audio/snippet` only when clicked), and **Delete embedding** per enrollment. Saving a speaker triggers a **global re-match** of stored transcript diarization metadata so existing transcripts can show updated names and similarity without re-transcribing.

---

## Offline model loading

Local ASR models are selected via **`ASR_MODEL`** (registry key in [`config/asr_models.yaml`](config/asr_models.yaml)). Unknown keys fail at startup.

Without HuggingFace downloads you can load:

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Optional offline path when the selected registry entry has `allow_path_override: true` — NeMo **`.nemo`** checkpoint or ONNX model directory (`.onnx` files). |
| `DIARIZATION_MODEL_PATH` | Path to a local **pyannote** pipeline directory (with `config.yaml` and weights). When set, diarization does not require `HUGGINGFACE_ACCESS_TOKEN`. |
| `ASR_MODELS_CONFIG` | Optional path to a custom ASR model registry YAML file (default: `config/asr_models.yaml`). |

Leave `MODEL_PATH` unset to download from Hugging Face / onnx-asr hub as configured in the registry entry.

Other useful paths:

| Variable | Default | Description |
|----------|---------|-------------|
| `RECORDINGS_PATH` | `./data/recordings` | Stored upload files |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/noobscribe.db` | Async SQLite URL for recordings metadata |
| `CHROMADB_PATH` | `./data/speakers` | Speaker embedding store |
| `LANGUAGE_ID_MODEL_ID` | `speechbrain/lang-id-voxlingua107-ecapa` | Hugging Face model id for spoken language ID when `language` is omitted |
| `LANGUAGE_ID_SAVEDIR` | `{TEMP_DIR}/speechbrain_lang_id` | Download/cache directory for the language-ID model |
| `LANGUAGE_ID_MAX_AUDIO_SECONDS` | `30` | Seconds per identification segment (start, middle, 75%); majority vote picks the final language |
| `DISABLE_LANGUAGE_ID` | unset | If `1` / `true` / `yes`, skip language ID; NeMo `source_lang` / `target_lang` stay unset when no `language` is sent |
| `USE_API` | unset | If `1` / `true` / `yes`, use remote OpenAI-compatible STT instead of loading a local ASR model. Requires **`STT_BASE_URL`** and **`STT_API_KEY`**. |
| `STT_BASE_URL` | _(unset)_ | OpenAI-compatible API base URL (e.g. `https://api.openai.com/v1`). Required when `USE_API` is set. |
| `STT_API_KEY` | _(unset)_ | Bearer token for the STT API. Required when `USE_API` is set. |
| `ASR_MODEL` | `canary-1b-v2` | **Local mode:** registry key from `config/asr_models.yaml`. Unknown keys fail at startup. |
| `MODEL_ID` | `whisper-1` | **Remote API mode only:** freeform remote `model` name (e.g. `whisper-1`). Ignored in local mode. |
| `ASR_MODELS_CONFIG` | _(unset)_ | Optional path to custom ASR model registry YAML. |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URL for the transcription job queue |
| `QUEUE_TRANSCRIPTION` | `1` | Enable async job queue (`0` disables enqueue) |
| `SKIP_MODEL_LOAD` | unset | When `1`, API process does not load NeMo (Docker API sets this; workers load models) |
| `LOAD_MODELS_ON_DEMAND` | unset (`false`) | When `1` / `true` / `yes`, skip ASR preload at startup. The transcription pipeline loads the local ASR model after language ID / diarization and unloads it afterward. Language ID and diarization already load/unload per use. |

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Status Codes

- `400 Bad Request`: Invalid request parameters or malformed request body
- `404 Not Found`: Resource not found (e.g., speaker ID not found)
- `500 Internal Server Error`: Server-side error during processing
- `503 Service Unavailable`: Service not ready (e.g., model not loaded, database not initialized)

---

## Type Definitions

### WhisperSegment

Represents a segment in the transcription.

```typescript
interface WhisperSegment {
  id: number;
  seek: number;
  start: number;  // seconds
  end: number;    // seconds
  text: string;
  tokens: number[];
  temperature: number;
  avg_logprob: number;
  compression_ratio: number;
  no_speech_prob: number;
  speaker?: string;  // Optional speaker label
}
```

### SpeakerInfo

Information about a speaker with embedding.

```typescript
interface SpeakerInfo {
  id: string;                    // e.g., "SPEAKER_00"
  display_name?: string;         // Display name if matched
  embedding: number[];           // Embedding vector
  matched: boolean;              // Whether matched to stored identity
  match_similarity?: number;     // Cosine similarity when matched (0–1)
}
```

### TranscriptionResponse

Response format for transcription (JSON/verbose_json).

```typescript
interface TranscriptionResponse {
  text: string;
  segments?: WhisperSegment[];
  language?: string;
  task: string;                 // Always "transcribe"
  duration?: number;            // seconds
  model?: string;
  speakers?: SpeakerInfo[];
}
```

### SpeakerCreate

Request model for creating a speaker.

```typescript
interface SpeakerCreate {
  display_name: string;
  embedding: number[];
}
```

### SpeakerUpdate

Request model for updating a speaker.

```typescript
interface SpeakerUpdate {
  embedding: number[];
}
```

### SpeakerResponse

Response model for speaker operations.

```typescript
interface SpeakerResponse {
  id: string;                   // UUID
  display_name: string;
  created_at: string;           // ISO 8601 timestamp
  embedding_count: number;
}
```

### SpeakerList

List of speakers response.

```typescript
interface SpeakerList {
  object: string;              // Always "list"
  data: SpeakerResponse[];
}
```

### SpeakerRematchResponse

Result of manual global rematch.

```typescript
interface SpeakerRematchResponse {
  updated_transcriptions: number;
}
```

### SpeakerNearestRequest

Nearest-neighbor query for stored embeddings.

```typescript
interface SpeakerNearestRequest {
  embedding: number[];
  limit?: number;              // default 3, max 10
}
```

### SpeakerNearestMatch

One nearest stored embedding hit.

```typescript
interface SpeakerNearestMatch {
  speaker_id: string;
  display_name: string;
  embedding_index: number;
  similarity: number;
  meets_threshold: boolean;
}
```

### SpeakerNearestResponse

Nearest-neighbor query result.

```typescript
interface SpeakerNearestResponse {
  threshold: number;
  data: SpeakerNearestMatch[];
}
```

### ModelInfo

Information about an available model.

```typescript
interface ModelInfo {
  id: string;
  object: string;              // Always "model"
  created: number;             // Unix timestamp
  owned_by: string;
  permission: Permission[];
  root: string;
  parent?: string;
}
```

### HealthResponse

Health check response.

```typescript
interface HealthResponse {
  status: string;              // Always "ok"
  version: string;
  model_loaded: boolean;
  speaker_db_initialized: boolean;
  model_id: string;
  cuda_available: boolean;
  gpu_info?: string;
  config: {
    model_id: string;
    chunk_min_duration: number;
    chunk_max_duration: number;
    chunk_silence_db: number;
    chunk_silence_min_duration: number;
    temp_dir: string;
    chromadb_path: string;
    speaker_similarity_threshold: number;
    include_diarization_in_text: boolean;
  };
}
```

---

## Notes

### Speaker Diarization

- Diarization uses **`pyannote/speaker-diarization-community-1`** (pyannote.audio 4.x) from HuggingFace **unless** `DIARIZATION_MODEL_PATH` is set to a local pipeline directory (fully offline). Accept the community-1 model terms on Hugging Face before download; a 3.1 offline cache is not compatible.
- Segment/speaker assignment uses **exclusive speaker diarization** from community-1 for better alignment with transcription timestamps.
- With HuggingFace, configure a token via:
  - Environment variable: `HUGGINGFACE_ACCESS_TOKEN`
  - Command-line argument: `--hf-token`
- Speaker embeddings are extracted and can be matched against stored speakers in the database
- If `include_diarization_in_text` is enabled, speaker labels are prepended to segment text

### Audio Processing

- The API automatically converts uploaded audio files to WAV format
- Long audio files are split at speaking pauses into chunks of roughly 15–30 seconds (force-split at 30s if no pause is found)
- Chunk bounds and silence detection are configurable via `CHUNK_MIN_DURATION`, `CHUNK_MAX_DURATION`, `CHUNK_SILENCE_DB`, and `CHUNK_SILENCE_MIN_DURATION`
- Supported audio formats include: WAV, MP3, M4A, FLAC, OGG, and others supported by FFmpeg

### Speaker Database

- The speaker database uses ChromaDB for storing and matching speaker embeddings
- Similarity matching uses cosine similarity
- Default similarity threshold is 0.7 (configurable)
- Multiple embeddings per speaker improve matching accuracy

### Compatibility

- The API is designed to be compatible with the OpenAI Whisper API
- Most Whisper API parameters are accepted for compatibility
- Response formats match OpenAI's Whisper API formats
- The `/v1/models` endpoint provides compatibility for applications expecting model listings

---

## Rate Limiting

Currently, the API does not implement rate limiting. However, transcription requests are computationally intensive, so consider implementing client-side rate limiting for production use.

---

## CORS

The API includes CORS middleware that allows all origins by default. For production deployments, consider restricting allowed origins.
