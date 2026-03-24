/**
 * Thin fetch client for the NoobScribe API (same origin as /ui).
 */
(function (global) {
  "use strict";

  function apiBase() {
    return "";
  }

  async function handle(res) {
    const text = await res.text();
    if (!res.ok) {
      let detail = text;
      try {
        const j = JSON.parse(text);
        detail = j.detail || JSON.stringify(j);
      } catch (e) {
        /* plain text */
      }
      throw new Error(detail || res.statusText);
    }
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch (e) {
      return text;
    }
  }

  global.NoobScribeApi = {
    health() {
      return fetch(apiBase() + "/health").then(handle);
    },

    listRecordings(limit, offset) {
      const q = new URLSearchParams();
      if (limit != null) q.set("limit", String(limit));
      if (offset != null) q.set("offset", String(offset));
      const s = q.toString();
      return fetch(apiBase() + "/v1/recordings" + (s ? "?" + s : "")).then(handle);
    },

    getRecording(id) {
      return fetch(apiBase() + "/v1/recordings/" + encodeURIComponent(id)).then(handle);
    },

    createRecording(file, name, hideInRecordings) {
      const fd = new FormData();
      fd.append("file", file);
      if (name) fd.append("name", name);
      if (hideInRecordings) fd.append("hide_in_recordings", "true");
      return fetch(apiBase() + "/v1/recordings", { method: "POST", body: fd }).then(handle);
    },

    extractSpeakersFromAudio(file, opts) {
      opts = opts || {};
      const fd = new FormData();
      fd.append("file", file);
      if (opts.name) fd.append("name", opts.name);
      if (opts.language) fd.append("language", opts.language);
      fd.append("response_format", opts.response_format || "verbose_json");
      fd.append("timestamps", String(opts.timestamps !== false));
      fd.append("word_timestamps", String(!!opts.word_timestamps));
      fd.append("diarize", String(opts.diarize !== false));
      if (opts.include_diarization_in_text != null) {
        fd.append("include_diarization_in_text", String(!!opts.include_diarization_in_text));
      }
      fd.append("temperature", String(opts.temperature ?? 0));
      return fetch(apiBase() + "/v1/speakers/extract-from-audio", {
        method: "POST",
        body: fd,
      }).then(handle);
    },

    patchRecording(id, name) {
      return fetch(apiBase() + "/v1/recordings/" + encodeURIComponent(id), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      }).then(handle);
    },

    deleteRecording(id) {
      return fetch(apiBase() + "/v1/recordings/" + encodeURIComponent(id), {
        method: "DELETE",
      }).then(handle);
    },

    recordingAudioUrl(id) {
      return apiBase() + "/v1/recordings/" + encodeURIComponent(id) + "/audio";
    },

    audioSnippetUrl(recordingId, start, end) {
      const q = new URLSearchParams();
      q.set("recording_id", String(recordingId));
      q.set("start", String(start));
      q.set("end", String(end));
      return apiBase() + "/v1/audio/snippet?" + q.toString();
    },

    transcribeRecording(id, opts) {
      const fd = new FormData();
      if (opts.language) fd.append("language", opts.language);
      fd.append("response_format", opts.response_format || "verbose_json");
      fd.append("timestamps", String(opts.timestamps !== false));
      fd.append("word_timestamps", String(!!opts.word_timestamps));
      fd.append("diarize", String(opts.diarize !== false));
      if (opts.include_diarization_in_text != null) {
        fd.append("include_diarization_in_text", String(!!opts.include_diarization_in_text));
      }
      fd.append("temperature", String(opts.temperature ?? 0));
      return fetch(apiBase() + "/v1/recordings/" + encodeURIComponent(id) + "/transcribe", {
        method: "POST",
        body: fd,
      }).then(handle);
    },

    listTranscriptions(recordingId) {
      return fetch(
        apiBase() + "/v1/recordings/" + encodeURIComponent(recordingId) + "/transcriptions"
      ).then(handle);
    },

    getTranscription(recordingId, transcriptionId) {
      return fetch(
        apiBase() +
          "/v1/recordings/" +
          encodeURIComponent(recordingId) +
          "/transcriptions/" +
          encodeURIComponent(transcriptionId)
      ).then(handle);
    },

    listSpeakers() {
      return fetch(apiBase() + "/v1/speakers").then(handle);
    },

    createSpeaker(displayName, embedding) {
      return fetch(apiBase() + "/v1/speakers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: displayName, embedding: embedding }),
      }).then(handle);
    },

    addEmbeddingToSpeaker(speakerId, embedding) {
      return fetch(apiBase() + "/v1/speakers/" + encodeURIComponent(speakerId), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ embedding: embedding }),
      }).then(handle);
    },

    deleteSpeaker(id) {
      return fetch(apiBase() + "/v1/speakers/" + encodeURIComponent(id), {
        method: "DELETE",
      }).then(handle);
    },

    getSpeakerEmbeddings(speakerId) {
      return fetch(
        apiBase() +
          "/v1/speakers/" +
          encodeURIComponent(speakerId) +
          "/embeddings"
      ).then(handle);
    },

    deleteSpeakerEmbedding(speakerId, embeddingIndex) {
      return fetch(
        apiBase() +
          "/v1/speakers/" +
          encodeURIComponent(speakerId) +
          "/embeddings/" +
          encodeURIComponent(String(embeddingIndex)),
        { method: "DELETE" }
      ).then(handle);
    },

    getEmbeddingSnippets(speakerId, embeddingIndex) {
      return fetch(
        apiBase() +
          "/v1/speakers/" +
          encodeURIComponent(speakerId) +
          "/embeddings/" +
          encodeURIComponent(String(embeddingIndex)) +
          "/snippets"
      ).then(handle);
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
