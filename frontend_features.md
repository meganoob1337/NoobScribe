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
| `#/jobs` | `jobs` | All visible transcription jobs (queue/history) |
| `#/r/:recordingId` | `detail` | One recording: audio, transcribe, transcriptions list |
| `#/r/:recordingId/t/:transcriptionId` | `transcript` | Read transcript, segments, save embeddings |

**`go(path)`** sets `location.hash` to `#` + path and closes the mobile drawer.

### Lifecycle on navigation

- Leaving **`speakers`**: tears down mic enrollment capture (`MediaRecorder` for extract flow), clears extract UI state.
- Leaving **`record`**: stops level polling, record timer, revokes preview object URL, destroys `NoobScribeRecorder`.
- Leaving **`transcript`** (or any non-transcript route): stops segment snippet audio playback and clears the hidden `Audio` element’s source.
- Leaving **`detail`**: stops recording-jobs polling interval.
- Leaving **`jobs`**: stops global jobs polling interval.

## Global chrome

- **Layout**: Fixed **sidebar** on large screens; **slide-out drawer** on small screens with backdrop, hamburger in sticky header, Escape closes drawer.
- **Nav items**: Recordings (active for list + detail + transcript), Upload, Record, Speakers, Jobs; **API docs** link to `/docs` (full navigation, not hash).
- **Health** (`GET /health`): Fetched once on init. Sidebar (and mobile header subtitle) show model readiness (`model_loaded`) and `config.asr_model` / `config.model_id`. When the queue is enabled and Redis is up but `workers_registered` is 0, the sidebar shows an amber warning that jobs will stay queued. Errors surface as amber “API: …” text.

## Feature: recordings list (`list`)

- **Data**: `GET /v1/recordings?limit=50&offset=…` — paginated; first load resets list; “Load more” increases offset by 50 when `has_more` is true.
- **Display**: Mobile = card list; `sm+` = table (name, created, duration seconds rounded, transcription count).
- **Actions**: Row/card navigates to `#/r/:id`. CTAs for “New upload” and “Record in browser”.
- **States**: Loading empty state, inline list error.

## Feature: upload recording (`upload`)

- Optional display name; file input accepts common audio and video types (`accept` includes `audio/*`, `video/*`, and extensions such as `.mp4`, `.webm`, `.mov`).
- **Submit**: `POST /v1/recordings` multipart (`file`, optional `name`) → on success navigates to `#/r/:id`.
- **States**: Busy button, validation error if no file chosen.

## Feature: record in browser (`record`)

Implemented by **`NoobScribeRecorder`** (`recorder.js`):

- **Window/tab audio**: `getDisplayMedia` with audio + video and explicit Chrome hints (`systemAudio: "include"`, `windowAudio: "window"`, plus `selfBrowserSurface: "exclude"`, `monitorTypeSurfaces: "include"`, `surfaceSwitching: "include"`); falls back to `{ audio: true, video: true }` on older browsers that reject unknown options. Video tracks stopped after connect; requires an audio track or shows an error. Users must enable **Share tab/system audio** in the browser picker (Chrome 141+).
- **Microphone**: `getUserMedia({ audio: true })`.
- **Mixing**: Web Audio API — sources → gain → analyser (levels) + `MediaStreamDestination`; **MediaRecorder** on the mixed stream (prefers WebM/Opus).
- **UI**: Toggle window vs mic; **level meters** (RMS 0–1, updated via `requestAnimationFrame` while on route); states `idle` | `capturing` | `recording` | `recorded`.
- **Recording**: Start requires at least one live source; **elapsed timer** while recording; stop produces a **Blob**, **object URL** for `<audio controls>` preview.
- **Floating controls (Chrome / Edge)**: After **Start recording**, the app requests **Document Picture-in-Picture** so a small OS-floating window shows elapsed time, **Stop**, **Mute mic**, and **Mute window** (only buttons for sources that were live at start). The main page mirrors the same mute/stop actions. PiP receives timer/state updates via `postMessage` from the main window; button clicks in the PiP DOM are wired in the main script and use `window.postMessage` back to the app. Leaving the Record route closes PiP and tears down capture as before.
- **In-recording mute**: Implemented by lowering Web Audio **gain** on the mic and/or window path (does not stop MediaRecorder or media tracks). Mutes reset when a take finishes (`stopRecording` success) or on discard.
- **Upload**: Builds `File` named `recording.webm` from blob → `POST /v1/recordings` → navigates to detail; then tears down recorder state.
- **Discard**: Revokes URL, clears blob.

Rebuild note: permissions UX, codec support, and “tab vs window” audio behavior are browser-dependent; the UI includes short copy explaining tab/screen share and the share-audio checkbox in Chrome/Edge.

## Feature: recording detail (`detail`)

- **Load**: `GET /v1/recordings/:id`.
- **Rename**: Editable name + “Save name” → `PATCH /v1/recordings/:id` JSON `{ name }`, then reload recording.
- **Audio playback**: `<audio>` src = `GET /v1/recordings/:id/audio` (URL builder in client).
- **Transcribe panel**:
  - Options bound to state: optional **language** string; checkboxes **diarize**, **word_timestamps**, **include_diarization_in_text**.
  - **Run** sends `POST /v1/recordings/:id/transcribe` as **FormData** (returns `202` + `job_id`), then refreshes the **Active jobs** panel via `GET /v1/recordings/:id/transcription-jobs`. The button stays enabled so multiple jobs can be queued with different options.
  - Queue/API errors show inline in the transcribe panel; job progress and cancel are not shown here.
- **Active jobs panel**: Shows **queued** and **running** jobs for this recording only. Polls every ~2s while any are active. Each row: status badge, job type, live progress from `progress_stage` / `progress_current` / `progress_total` (e.g. “Loading diarization model…”, “Running diarization…”, “Loading transcription model…”, “Transcribing chunk 3/10…”), **Cancel** → `POST /v1/transcription-jobs/:jobId/cancel`. When the last active job finishes, the recording is re-fetched so the **Stored transcriptions** list updates.
- **Transcriptions list**: Each item shows created time and “· diarized” if `diarization_enabled`; click → `#/r/:id/t/:transcriptionId`.
- **Delete recording**: Confirm dialog → `DELETE /v1/recordings/:id` → navigate home.

## Feature: transcription jobs (`jobs`)

- **Load**: `GET /v1/transcription-jobs?limit=100` — active jobs plus historic rows while still in Redis.
- **Refresh** button re-fetches the list.
- **Table columns**: created time, job type, status badge, recording name (link to `#/r/:id` when present), progress text, actions.
- **Cancel**: For `cancellable` rows → `POST /v1/transcription-jobs/:jobId/cancel`.
- **View transcript**: When `status === completed` and both `recording_id` and `transcription_id` are set → `#/r/:rid/t/:tid`.
- **Polling**: Every ~2s while any job is queued or running; stops when leaving the route or when no active jobs remain.

## Feature: transcript viewer (`transcript`)

- **Load**: `GET /v1/recordings/:rid/transcriptions/:tid`.
- **Formatted body**: If `segments[]` exists, builds text with **one segment per paragraph**: `[resolved_label] body`, where label comes from `transcription.speakers[]` (`display_name` or id). **Strips** a leading duplicate speaker prefix from segment text (e.g. `SPEAKER_00: `) when it matches the segment’s diarization id (case-insensitive). Unknown speaker id still shows body only. If no segments, falls back to `full_text`.
- **Copy**: Copies the full `formattedTranscript()` output to the clipboard via `navigator.clipboard.writeText()`; shows a brief "Copied!" confirmation that auto-hides after 2 seconds.
- **Download**: Client-side **Blob** download as `transcription-<sanitized-id>.txt` from `formattedTranscript()` output.
- **Speakers block**: Expandable accordion rows (sorted by segment count, highest first). Each header shows diarization id, optional `display_name`, **segment count** badge, “matched” badge with **similarity percentage** (`match_similarity`, e.g. `87%`) when matched, **≈ nearest** icon (lazy-loads top 3 stored embedding hits via `POST /v1/speakers/nearest-embeddings` with threshold and matched/below labels), and **Save as…** (when embedding exists; does not toggle the row). Header also has **Re-match transcripts** (`POST /v1/speakers/rematch`); on success reloads the current transcription. Expanding a row lists that speaker’s segments only: time range, stripped text preview, and **play** control (same snippet API as the Segments list below). State resets when loading another transcription.
- **Segments list**: Per segment: time range (`start`–`end`, one decimal), resolved speaker label, stripped text, and optional **play** control if `start`/`end` are valid numbers and `end > start`.
  - **Snippet playback**: Single shared `Audio` element, `preload="none"`. Play uses `GET /v1/audio/snippet?recording_id=&start=&end=`. Tapping same segment while playing pauses; switching segment sets new `src` and plays.

## Feature: speakers (`speakers`)

### List and CRUD

- **Load**: `GET /v1/speakers` → list with `display_name`, `id`, `embedding_count`.
- **Re-match transcripts**: Button in page header → `POST /v1/speakers/rematch`; shows count of updated transcriptions (`updated_transcriptions`).
- **Delete speaker**: Confirm → `DELETE /v1/speakers/:id`; clears expanded row state and cached snippet panels for that speaker.

### Collapsible “Add or update speaker embedding”

Two tabs:

1. **From audio snippet**
   - Optional label field (`extractSnippetName`).
   - Source: **file** input or **in-page mic record** (simple `MediaRecorder` on mic-only stream, WebM blob + preview URL).
   - **Extract**: `POST /v1/speakers/extract-from-audio` multipart (returns `202` + `job_id`), then polls the job endpoint; uses `job.result` when complete. Same form fields: optional `name`, `response_format: verbose_json`, `timestamps: true`, `word_timestamps: false`, `diarize: true`, `include_diarization_in_text: false`, `temperature: 0`.
   - Shows chunk/diarization progress while the worker runs; on completion shows optional `recording_id` note (hidden enrollment recording), formatted transcript preview (`extractFormattedTranscript()` — same bracket/strip rules as main transcript), and **Detected speakers** with **≈ nearest** icon (same nearest-embeddings popover as transcript speakers), **Save as…** per speaker with embedding.
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
- **`rematchAllTranscriptions`**: `POST /v1/speakers/rematch` — returns `{ updated_transcriptions }`.
- **`getNearestEmbeddings(embedding, limit?)`**: `POST /v1/speakers/nearest-embeddings` — returns `{ threshold, data[] }` with per-hit `meets_threshold`.

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
