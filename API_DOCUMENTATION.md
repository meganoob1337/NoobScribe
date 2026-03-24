# API Documentation

## Base URL

The API is typically available at `http://localhost:8000` (default port). The base URL can be configured via the `--port` option when starting the server.

**Root path:** `GET /` returns **302** to `/ui` (Web UI). Not listed in OpenAPI (`/docs`).

## Authentication

Currently, the API does not require authentication.

## Content Types

- **Multipart Form Data**: Used for file uploads (audio transcription)
- **JSON**: Used for JSON request/response bodies (speaker management)

## Endpoints

### 1. Transcribe Audio

Transcribe an audio file using the NVIDIA NeMo ASR (default: Canary 1B v2) with optional speaker diarization.

**Endpoint:** `POST /v1/audio/transcriptions`

**Content-Type:** `multipart/form-data`

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | The audio file to transcribe (supports common audio formats: wav, mp3, m4a, flac, etc.) |
| `model` | string | No | `"whisper-1"` | Model identifier (accepted for compatibility; server uses configured NeMo ASR, default `nvidia/canary-1b-v2`) |
| `language` | string | No | `null` | Language code of the audio (e.g., "en", "de", "fr"). If omitted or blank, the service runs [SpeechBrain VoxLingua107 ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) on the first portion of the audio (see `LANGUAGE_ID_MAX_AUDIO_SECONDS`) and passes that code as both NeMo `source_lang` and `target_lang`. If identification is disabled or fails, both are left unset (previous behavior). Set `DISABLE_LANGUAGE_ID=1` to skip detection. |
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
      "matched": true
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
  "model": "nvidia/canary-1b-v2",
  "speakers": [
    {
      "id": "SPEAKER_00",
      "display_name": "John Doe",
      "embedding": [0.1, 0.2, 0.3, ...],
      "matched": true
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
  "model_id": "nvidia/canary-1b-v2",
  "cuda_available": true,
  "gpu_info": "NVIDIA GeForce RTX 3090",
  "config": {
    "model_id": "nvidia/canary-1b-v2",
    "chunk_duration": 20.0,
    "temp_dir": "/tmp/noobscribe",
    "chromadb_path": "./data/speakers",
    "model_path": null,
    "diarization_model_path": null,
    "recordings_path": "./data/recordings",
    "database_url": "sqlite+aiosqlite:///./data/noobscribe.db",
    "speaker_similarity_threshold": 0.7,
    "include_diarization_in_text": true
  }
}
```

**Response Fields:**
- `status` (string): Always "ok" if the API is running
- `version` (string): API version
- `model_loaded` (boolean): Whether the ASR model is loaded and ready
- `model_id` (string): Identifier of the loaded model
- `cuda_available` (boolean): Whether CUDA/GPU is available
- `gpu_info` (string, optional): GPU name if CUDA is available, `null` otherwise
- `config` (object): Current server configuration

#### Example Request

```bash
curl http://localhost:8000/health
```

#### Status Codes

- `200 OK`: API is healthy

---

### 3. List Models

List available models (compatibility endpoint with OpenAI API).

**Endpoint:** `GET /v1/models`

**Content-Type:** `application/json`

#### Response Schema

```json
{
  "object": "list",
  "data": [
    {
      "id": "whisper-1",
      "object": "model",
      "created": 1677649963,
      "owned_by": "noobscribe",
      "root": "whisper-1",
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
- `503 Service Unavailable`: Speaker database not initialized

---

### 5. Extract speaker embeddings from audio

Upload a short audio clip in one step: the server stores it as a **recording** with `hide_in_recordings=true` (so it does **not** appear in `GET /v1/recordings`), runs the same transcription + optional diarization pipeline as stored recording transcribe, **persists** a `TranscriptionResult`, and returns diarized **speaker rows with `embedding` vectors** (for enrolling via `POST`/`PUT` `/v1/speakers`). Persisted data enables embedding snippet previews (`GET /v1/speakers/{id}/embeddings/{index}/snippets`) the same way as normal library recordings.

**Endpoint:** `POST /v1/speakers/extract-from-audio`

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

**Response (json / verbose_json):** Same fields as a successful stored `POST /v1/recordings/{recording_id}/transcribe` response, plus **`recording_id`** (UUID of the hidden recording).

**Status codes:** `200`, `400`, `500`, `503` (model not loaded).

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

**Side effect:** After a successful create, the server **re-matches** every stored recording transcription that has a `speakers` array: for each diarized row with an `embedding`, it runs the same similarity check as live transcription (`SPEAKER_SIMILARITY_THRESHOLD`) against the Chroma speaker index and **updates persisted** `display_name` / `matched` in SQLite. No extra request fields are required. This can add latency when many transcripts exist (one vector query per diarized speaker row).

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

Same form fields as `POST /v1/audio/transcriptions` except **no** `file`: `language`, `response_format`, `temperature`, `timestamps`, `word_timestamps`, `diarize`, `include_diarization_in_text`.

Persists a **TranscriptionResult** (segments + speakers JSON). For `json` / `verbose_json`, the response body includes extra fields:

- `transcription_id` (string): ID of the stored row
- `stored` (boolean): `true`

Other formats (`text`, `srt`, `vtt`) return plain text as today.

---

### 16. List transcriptions for a recording

**Endpoint:** `GET /v1/recordings/{recording_id}/transcriptions`

**Response:** `{ "object": "list", "data": [ TranscriptionResultResponse ] }`

**TranscriptionResultResponse:** `id`, `recording_id`, `full_text`, `segments` (JSON array), `language`, `model_id`, `diarization_enabled`, `speakers` (JSON array), `duration_seconds`, `word_timestamps`, `created_at`.

---

### 17. Get one stored transcription

**Endpoint:** `GET /v1/recordings/{recording_id}/transcriptions/{transcription_id}`

**Response:** `TranscriptionResultResponse`

---

## Web UI

A minimal SPA is served at **`/ui`** (same host as the API). Open `/ui` in a browser to upload recordings, **record in the browser** (mix window/tab or screen capture audio with the microphone, then upload as WebM), run transcription/diarization, inspect segments, save speaker embeddings to `/v1/speakers`, and manage stored speakers. On **Speakers**, **From audio snippet** uploads or records a short clip, calls **`POST /v1/speakers/extract-from-audio`** (hidden stored recording + diarization), then lets you **Save as…** for each detected speaker like the transcript view. The **Speakers** page also uses expandable rows: each speaker lists **enrollments** (embedding index), matching **transcriptions** with an **Audio snippet** control (loads `GET /v1/audio/snippet` only when clicked), and **Delete embedding** per enrollment. Saving a speaker triggers a **global re-match** of stored transcript diarization metadata so existing transcripts can show updated names without re-transcribing.

---

## Offline model loading

Without HuggingFace downloads you can load:

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Path to a local NeMo **`.nemo`** checkpoint. When set, ASR is loaded with `ASRModel.restore_from` instead of `from_pretrained(MODEL_ID)`. |
| `DIARIZATION_MODEL_PATH` | Path to a local **pyannote** pipeline directory (with `config.yaml` and weights). When set, diarization does not require `HUGGINGFACE_ACCESS_TOKEN`. |

Leave both unset to use HuggingFace / cache as before.

Other useful paths:

| Variable | Default | Description |
|----------|---------|-------------|
| `RECORDINGS_PATH` | `./data/recordings` | Stored upload files |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/noobscribe.db` | Async SQLite URL for recordings metadata |
| `CHROMADB_PATH` | `./data/speakers` | Speaker embedding store |
| `LANGUAGE_ID_MODEL_ID` | `speechbrain/lang-id-voxlingua107-ecapa` | Hugging Face model id for spoken language ID when `language` is omitted |
| `LANGUAGE_ID_SAVEDIR` | `{TEMP_DIR}/speechbrain_lang_id` | Download/cache directory for the language-ID model |
| `LANGUAGE_ID_MAX_AUDIO_SECONDS` | `30` | Seconds of audio (from the start) used for identification |
| `DISABLE_LANGUAGE_ID` | unset | If `1` / `true` / `yes`, skip language ID; NeMo `source_lang` / `target_lang` stay unset when no `language` is sent |

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
  model_id: string;
  cuda_available: boolean;
  gpu_info?: string;
  config: {
    model_id: string;
    chunk_duration: number;
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

- Diarization uses **`pyannote/speaker-diarization-3.1`** from HuggingFace **unless** `DIARIZATION_MODEL_PATH` is set to a local pipeline directory (fully offline).
- With HuggingFace, configure a token via:
  - Environment variable: `HUGGINGFACE_ACCESS_TOKEN`
  - Command-line argument: `--hf-token`
- Speaker embeddings are extracted and can be matched against stored speakers in the database
- If `include_diarization_in_text` is enabled, speaker labels are prepended to segment text

### Audio Processing

- The API automatically converts uploaded audio files to WAV format
- Long audio files are automatically split into chunks for processing
- Chunk duration is configurable (default: 30 seconds)
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
