# NoobScribe — frontend feature inventory

This document describes what the shipped web UI (`webui/`) implements so you can recreate it in another stack. It is behavior-oriented, not a pixel-perfect spec.

## Current stack (reference)

- **Single-page app**: one HTML shell (`index.html`) with **Alpine.js 3** for reactivity and `x-show` / `x-for` views.
- **Styling**: Tailwind CSS via CDN; font **Plus Jakarta Sans** (Google Fonts).
- **Scripts** (load order): `api.js` (fetch client) → `recorder.js` (browser capture) → `app.js` (app state + routing) → Alpine (deferred).
- **Hosting assumption**: UI is served from the **same origin** as the API (`api.js` uses relative URLs, empty base). Paths like `/health`, `/v1/...`, and `/docs` are expected on that host.

## Routing (hash-based)

Navigation uses `location.hash` (no server-side router). `hashchange` triggers parsing.

| Hash pattern | View / route name | Purpose |
|--------------|-------------------|---------|
| `#/`, empty, or unknown | `list` | Recordings library |
| `#/upload` | `upload` | File upload to create a recording |
| `#/record` | `record` | In-browser capture + upload |
| `#/speakers` | `speakers` | Speaker database + enrollment |
| `#/r/:recordingId` | `detail` | One recording: audio, transcribe, transcriptions list |
| `#/r/:recordingId/t/:transcriptionId` | `transcript` | Read transcript, segments, save embeddings |

**`go(path)`** sets `location.hash` to `#` + path and closes the mobile drawer.

### Lifecycle on navigation

- Leaving **`speakers`**: tears down mic enrollment capture (`MediaRecorder` for extract flow), clears extract UI state.
- Leaving **`record`**: stops level polling, record timer, revokes preview object URL, destroys `NoobScribeRecorder`.
- Leaving **`transcript`** (or any non-transcript route): stops segment snippet audio playback and clears the hidden `Audio` element’s source.

## Global chrome

- **Layout**: Fixed **sidebar** on large screens; **slide-out drawer** on small screens with backdrop, hamburger in sticky header, Escape closes drawer.
- **Nav items**: Recordings (active for list + detail + transcript), Upload, Record, Speakers; **API docs** link to `/docs` (full navigation, not hash).
- **Health** (`GET /health`): Fetched once on init. Sidebar (and mobile header subtitle) show model readiness (`model_loaded`) and `config.model_id`. Errors surface as amber “API: …” text.

## Feature: recordings list (`list`)

- **Data**: `GET /v1/recordings?limit=50&offset=…` — paginated; first load resets list; “Load more” increases offset by 50 when `has_more` is true.
- **Display**: Mobile = card list; `sm+` = table (name, created, duration seconds rounded, transcription count).
- **Actions**: Row/card navigates to `#/r/:id`. CTAs for “New upload” and “Record in browser”.
- **States**: Loading empty state, inline list error.

## Feature: upload recording (`upload`)

- Optional display name; file input accepts common audio types (`accept` includes `audio/*` and extensions).
- **Submit**: `POST /v1/recordings` multipart (`file`, optional `name`) → on success navigates to `#/r/:id`.
- **States**: Busy button, validation error if no file chosen.

## Feature: record in browser (`record`)

Implemented by **`NoobScribeRecorder`** (`recorder.js`):

- **Window/tab audio**: `getDisplayMedia` with audio + video; video tracks stopped after connect; requires an audio track or shows an error.
- **Microphone**: `getUserMedia({ audio: true })`.
- **Mixing**: Web Audio API — sources → gain → analyser (levels) + `MediaStreamDestination`; **MediaRecorder** on the mixed stream (prefers WebM/Opus).
- **UI**: Toggle window vs mic; **level meters** (RMS 0–1, updated via `requestAnimationFrame` while on route); states `idle` | `capturing` | `recording` | `recorded`.
- **Recording**: Start requires at least one live source; **elapsed timer** while recording; stop produces a **Blob**, **object URL** for `<audio controls>` preview.
- **Upload**: Builds `File` named `recording.webm` from blob → `POST /v1/recordings` → navigates to detail; then tears down recorder state.
- **Discard**: Revokes URL, clears blob.

Rebuild note: permissions UX, codec support, and “tab vs window” audio behavior are browser-dependent; the UI includes short copy explaining tab/screen share for system audio.

## Feature: recording detail (`detail`)

- **Load**: `GET /v1/recordings/:id`.
- **Rename**: Editable name + “Save name” → `PATCH /v1/recordings/:id` JSON `{ name }`, then reload recording.
- **Audio playback**: `<audio>` src = `GET /v1/recordings/:id/audio` (URL builder in client).
- **Transcribe panel**:
  - Options bound to state: optional **language** string; checkboxes **diarize**, **word_timestamps**, **include_diarization_in_text**.
  - **Run** sends `POST /v1/recordings/:id/transcribe` as **FormData** with: `language` (if set), `response_format: verbose_json`, `timestamps: true`, `word_timestamps`, `diarize`, `include_diarization_in_text`, `temperature: 0`.
  - After success, reloads recording to refresh transcriptions list.
- **Transcriptions list**: Each item shows created time and “· diarized” if `diarization_enabled`; click → `#/r/:id/t/:transcriptionId`.
- **Delete recording**: Confirm dialog → `DELETE /v1/recordings/:id` → navigate home.

## Feature: transcript viewer (`transcript`)

- **Load**: `GET /v1/recordings/:rid/transcriptions/:tid`.
- **Formatted body**: If `segments[]` exists, builds text with **one segment per paragraph**: `[resolved_label] body`, where label comes from `transcription.speakers[]` (`display_name` or id). **Strips** a leading duplicate speaker prefix from segment text (e.g. `SPEAKER_00: `) when it matches the segment’s diarization id (case-insensitive). Unknown speaker id still shows body only. If no segments, falls back to `full_text`.
- **Download**: Client-side **Blob** download as `transcription-<sanitized-id>.txt` from `formattedTranscript()` output.
- **Speakers block**: For each entry in `transcription.speakers` with a non-empty `embedding`, show id, optional `display_name`, “matched” badge, **Save as…** → opens the same “save embedding” flow as on the Speakers page (load speakers list, choose existing or new name, POST/PUT).
- **Segments list**: Per segment: time range (`start`–`end`, one decimal), resolved speaker label, stripped text, and optional **play** control if `start`/`end` are valid numbers and `end > start`.
  - **Snippet playback**: Single shared `Audio` element, `preload="none"`. Play uses `GET /v1/audio/snippet?recording_id=&start=&end=`. Tapping same segment while playing pauses; switching segment sets new `src` and plays.

## Feature: speakers (`speakers`)

### List and CRUD

- **Load**: `GET /v1/speakers` → list with `display_name`, `id`, `embedding_count`.
- **Delete speaker**: Confirm → `DELETE /v1/speakers/:id`; clears expanded row state and cached snippet panels for that speaker.

### Collapsible “Add or update speaker embedding”

Two tabs:

1. **From audio snippet**
   - Optional label field (`extractSnippetName`).
   - Source: **file** input or **in-page mic record** (simple `MediaRecorder` on mic-only stream, WebM blob + preview URL).
   - **Extract**: `POST /v1/speakers/extract-from-audio` multipart with file, optional `name`, `response_format: verbose_json`, `timestamps: true`, `word_timestamps: false`, `diarize: true`, `include_diarization_in_text: false`, `temperature: 0`.
   - Shows optional `recording_id` note (hidden enrollment recording), formatted transcript preview (`extractFormattedTranscript()` — same bracket/strip rules as main transcript), and **Detected speakers** with **Save as…** per speaker with embedding.
   - If no speaker has a non-empty `embedding` array, sets a helpful **error message** (diarization/embeddings unavailable, clip quality, etc.).

2. **Manual embedding**
   - Parse **JSON array of numbers** from textarea; validate `Array.isArray`.
   - Target: new speaker name **or** existing speaker from select → `POST /v1/speakers` or `PUT /v1/speakers/:id` with `{ embedding }` (PUT adds embedding to existing).

### “Save embedding” modal state (shared with transcript view)

- Fields: optional **existing speaker** select (`speakers` list), or **new name** when “New speaker…” selected.
- **Confirm**: If existing id → `PUT` add embedding; else `POST` create speaker with embedding. Refreshes speaker list and any expanded speakers’ embeddings; purges snippet cache for affected speaker.

### Per-speaker accordion

- **Expand row**: `GET /v1/speakers/:id/embeddings` — list of `{ embedding_index, created_at, … }`.
- **Per embedding**: Nested accordion loads **snippets** on first open: `GET /v1/speakers/:id/embeddings/:index/snippets`.
  - Each snippet: recording label, transcription id prefix, time range, `segment_text`, button **Audio snippet** sets `snip._audioSrc = snip.preview_url` and shows `<audio controls>`.
- **Delete embedding**: Confirm → `DELETE /v1/speakers/:id/embeddings/:index` → refresh embeddings for that speaker and speaker list; purge snippet cache for that speaker.

## API client behaviors (`api.js`) worth mirroring

- **Errors**: Read body as text; if JSON, prefer `detail` for message; throw `Error` with that string.
- **Success**: Parse JSON if body non-empty; empty body → `null`.
- **`createRecording`**: Supports optional `hide_in_recordings` form field — **not** used by current UI paths shown in `index.html`, but the client exposes it for parity with the API.

## Data shapes the UI expects (informal)

Rebuilders should align with the live OpenAPI/docs, but the UI assumes roughly:

- **Recording list item**: `id`, `name`, `created_at`, `duration_seconds?`, `transcription_count`.
- **Recording detail**: above plus `transcriptions[]` with `id`, `created_at`, `diarization_enabled`.
- **Transcription**: `id`, `recording_id`, `segments[]` with `start`, `end`, `text`, `speaker?`, `speakers[]` with `id`, `display_name?`, `matched?`, `embedding?`, `full_text` fallback.
- **Extract result**: `recording_id?`, `segments` or `text`, `speakers[]` as above.
- **Speaker**: `id`, `display_name`, `embedding_count`.
- **Snippet item**: `preview_url`, `recording_id`, `recording_name?`, `transcription_id`, `start`, `end`, `segment_text`.

## Accessibility and UX touches

- Sidebar `aria-label`; mobile menu button `aria-label="Open menu"`; some expandable rows use `aria-expanded`.
- Segment play buttons use dynamic `aria-label` (play vs pause).
- `x-cloak` CSS hides Alpine-bound markup until initialized.

## Out of scope in the current UI

- No login or multi-tenant UI.
- No inline editing of transcript text.
- No `listTranscriptions` as a standalone call — transcriptions come embedded on the recording resource.
- No settings page; transcription options are per-run on the detail screen only.

When the API contract changes, keep this file and `API_DOCUMENTATION.md` in sync per project conventions.
