/**
 * Alpine.js SPA: hash routing, recordings, transcription viewer, speakers.
 */
function noobScribeApp() {
  return {
    route: { name: "list", recordingId: null, transcriptionId: null },
    health: null,
    healthError: null,
    recordings: [],
    listLoading: false,
    listError: null,
    hasMore: false,
    listOffset: 0,

    uploadFile: null,
    uploadName: "",
    uploadBusy: false,
    uploadError: null,

    recording: null,
    recLoading: false,
    recError: null,
    transcribeBusy: false,
    transcribeError: null,
    transcribeOpts: {
      language: "",
      diarize: true,
      word_timestamps: false,
      include_diarization_in_text: true,
    },

    transcription: null,
    trLoading: false,
    trError: null,

    /** Transcript segment snippet playback (single hidden Audio, no preload until play) */
    _trSegmentAudio: null,
    trSegmentPlayIndex: null,
    trSegmentIsPlaying: false,

    speakers: [],
    spLoading: false,
    spError: null,
    /** Speakers page: accordion + embeddings */
    expandedSpeakers: {},
    speakerEmbeddingsBySpeaker: {},
    embeddingsLoadingFor: null,
    snippetPanelKey: null,
    snippetsByKey: {},
    snippetsLoadingKey: null,
    newSpeakerName: "",
    manualEmbeddingJson: "",
    manualTargetSpeakerId: "",
    manualBusy: false,

    /** Speakers page: "audio" | "manual" tab in add-embedding card */
    speakerEmbeddingTab: "audio",
    /** Collapsible add/update embedding card */
    speakerEmbeddingSectionExpanded: true,

    /** From-audio enrollment (hidden recording + diarization) */
    extractResult: null,
    extractBusy: false,
    extractError: null,
    extractSnippetName: "",
    spExtractFile: null,
    spExtractBlob: null,
    spExtractUrl: null,
    spExtractRecording: false,
    _spExtractMediaRecorder: null,
    _spExtractStream: null,

    saveSpeakerLabel: "",
    saveSpeakerTargetId: "",
    saveSpeakerEmbedding: null,
    saveSpeakerBusy: false,
    saveSpeakerError: null,

    /** In-browser recording (#/record) */
    recorder: null,
    _levelRaf: null,
    _recordTimerId: null,
    _recordStartedAt: null,
    recorderState: "idle",
    windowCaptureOn: false,
    micCaptureOn: false,
    windowLevel: 0,
    micLevel: 0,
    recordedBlob: null,
    recordedUrl: null,
    recordName: "",
    recordUploadBusy: false,
    recordError: null,
    recordElapsedSec: 0,

    /** Mobile slide-out navigation */
    navOpen: false,

    init() {
      window.addEventListener("hashchange", () => this.navigate());
      this.navigate();
      NoobScribeApi.health()
        .then((h) => {
          this.health = h;
          this.healthError = null;
        })
        .catch((e) => {
          this.healthError = String(e.message || e);
        });
    },

    navigate() {
      const prevName = this.route && this.route.name;
      const raw = (location.hash || "#/").replace(/^#/, "") || "/";
      const parts = raw.split("/").filter(Boolean);
      if (prevName === "speakers" && parts[0] !== "speakers") {
        this.teardownSpExtractCapture();
        this.clearExtractResult();
      }
      if (prevName === "record" && parts[0] !== "record") {
        this.teardownRecordPage();
      }
      if (parts.length === 0 || parts[0] === "") {
        this.route = { name: "list", recordingId: null, transcriptionId: null };
        this.loadList(true);
      } else if (parts[0] === "upload") {
        this.route = { name: "upload", recordingId: null, transcriptionId: null };
      } else if (parts[0] === "record") {
        this.route = { name: "record", recordingId: null, transcriptionId: null };
        this.ensureRecorder();
        this.startLevelPolling();
      } else if (parts[0] === "speakers") {
        this.route = { name: "speakers", recordingId: null, transcriptionId: null };
        this.loadSpeakers();
      } else if (parts[0] === "r" && parts[1]) {
        const rid = parts[1];
        if (parts[2] === "t" && parts[3]) {
          this.route = { name: "transcript", recordingId: rid, transcriptionId: parts[3] };
          this.loadTranscription(rid, parts[3]);
        } else {
          this.route = { name: "detail", recordingId: rid, transcriptionId: null };
          this.loadRecording(rid);
        }
      } else {
        this.route = { name: "list", recordingId: null, transcriptionId: null };
        this.loadList(true);
      }
      if (this.route.name !== "transcript") {
        this.stopTranscriptSegmentPlayback();
      }
    },

    go(path) {
      this.navOpen = false;
      location.hash = "#" + path;
    },

    /** Sidebar / drawer: highlight Recordings for list, detail, and transcript views */
    navActive(name) {
      if (name === "recordings") {
        return ["list", "detail", "transcript"].includes(this.route.name);
      }
      return this.route.name === name;
    },

    ensureRecorder() {
      if (typeof NoobScribeRecorder === "undefined") {
        this.recordError = "Recorder module not loaded";
        return;
      }
      if (!this.recorder) {
        this.recorder = new NoobScribeRecorder();
      }
    },

    teardownRecordPage() {
      this.stopLevelPolling();
      this.stopRecordTimer();
      if (this.recordedUrl) {
        try {
          URL.revokeObjectURL(this.recordedUrl);
        } catch (e) {
          /* ignore */
        }
        this.recordedUrl = null;
      }
      if (this.recorder) {
        this.recorder.destroy();
        this.recorder = null;
      }
      this.recordedBlob = null;
      this.recordError = null;
      this.recordUploadBusy = false;
      this.recordName = "";
      this.recorderState = "idle";
      this.windowLevel = 0;
      this.micLevel = 0;
      this.windowCaptureOn = false;
      this.micCaptureOn = false;
      this.recordElapsedSec = 0;
    },

    startLevelPolling() {
      this.stopLevelPolling();
      const tick = () => {
        if (this.route.name !== "record" || !this.recorder) return;
        this.windowLevel = this.recorder.readWindowLevel();
        this.micLevel = this.recorder.readMicLevel();
        this.windowCaptureOn = this.recorder.windowActive;
        this.micCaptureOn = this.recorder.micActive;
        this._levelRaf = requestAnimationFrame(tick);
      };
      this._levelRaf = requestAnimationFrame(tick);
    },

    stopLevelPolling() {
      if (this._levelRaf != null) {
        cancelAnimationFrame(this._levelRaf);
        this._levelRaf = null;
      }
    },

    _syncRecorderState() {
      if (!this.recorder) {
        this.recorderState = "idle";
        return;
      }
      if (this.recordedBlob) {
        this.recorderState = "recorded";
        return;
      }
      if (this.recorder.isRecording) {
        this.recorderState = "recording";
        return;
      }
      if (this.recorder.windowActive || this.recorder.micActive) {
        this.recorderState = "capturing";
        return;
      }
      this.recorderState = "idle";
    },

    startRecordTimer() {
      this.stopRecordTimer();
      this._recordStartedAt = Date.now();
      this.recordElapsedSec = 0;
      this._recordTimerId = setInterval(() => {
        if (this._recordStartedAt) {
          this.recordElapsedSec = Math.floor((Date.now() - this._recordStartedAt) / 1000);
        }
      }, 500);
    },

    stopRecordTimer() {
      if (this._recordTimerId != null) {
        clearInterval(this._recordTimerId);
        this._recordTimerId = null;
      }
      this._recordStartedAt = null;
    },

    async toggleWindowCapture() {
      this.recordError = null;
      this.ensureRecorder();
      if (!this.recorder) return;
      try {
        if (this.recorder.windowActive) {
          this.recorder.stopWindowCapture();
        } else {
          await this.recorder.startWindowCapture();
        }
        this._syncRecorderState();
      } catch (e) {
        this.recordError = String(e.message || e);
        this._syncRecorderState();
      }
    },

    async toggleMicCapture() {
      this.recordError = null;
      this.ensureRecorder();
      if (!this.recorder) return;
      try {
        if (this.recorder.micActive) {
          this.recorder.stopMicCapture();
        } else {
          await this.recorder.startMicCapture();
        }
        this._syncRecorderState();
      } catch (e) {
        this.recordError = String(e.message || e);
        this._syncRecorderState();
      }
    },

    canStartRecording() {
      return (
        this.recorder &&
        this.recorder.hasLiveAudioSource &&
        !this.recorder.isRecording &&
        !this.recordedBlob
      );
    },

    startRec() {
      this.recordError = null;
      if (!this.canStartRecording()) return;
      try {
        this.recorder.startRecording();
        this.startRecordTimer();
        this._syncRecorderState();
      } catch (e) {
        this.recordError = String(e.message || e);
        this._syncRecorderState();
      }
    },

    async stopRec() {
      if (!this.recorder || !this.recorder.isRecording) return;
      this.recordError = null;
      this.stopRecordTimer();
      try {
        const blob = await this.recorder.stopRecording();
        if (this.recordedUrl) {
          try {
            URL.revokeObjectURL(this.recordedUrl);
          } catch (e) {
            /* ignore */
          }
        }
        this.recordedBlob = blob;
        this.recordedUrl = URL.createObjectURL(blob);
        this._syncRecorderState();
      } catch (e) {
        this.recordError = String(e.message || e);
        this._syncRecorderState();
      }
    },

    discardRecording() {
      if (this.recordedUrl) {
        try {
          URL.revokeObjectURL(this.recordedUrl);
        } catch (e) {
          /* ignore */
        }
      }
      this.recordedUrl = null;
      this.recordedBlob = null;
      this.recordElapsedSec = 0;
      this._syncRecorderState();
    },

    formatRecordElapsed() {
      const s = this.recordElapsedSec || 0;
      const m = Math.floor(s / 60);
      const r = s % 60;
      return m + ":" + (r < 10 ? "0" : "") + r;
    },

    async uploadRecording() {
      if (!this.recordedBlob) {
        this.recordError = "Nothing to upload";
        return;
      }
      this.recordUploadBusy = true;
      this.recordError = null;
      try {
        const file = new File([this.recordedBlob], "recording.webm", {
          type: this.recordedBlob.type || "audio/webm",
        });
        const rec = await NoobScribeApi.createRecording(file, this.recordName || null);
        this.teardownRecordPage();
        this.go("/r/" + rec.id);
      } catch (e) {
        this.recordError = String(e.message || e);
      } finally {
        this.recordUploadBusy = false;
      }
    },

    async loadList(reset) {
      if (reset) {
        this.listOffset = 0;
        this.recordings = [];
      }
      this.listLoading = true;
      this.listError = null;
      try {
        const res = await NoobScribeApi.listRecordings(50, this.listOffset);
        const chunk = res.data || [];
        this.hasMore = !!res.has_more;
        this.recordings = reset ? chunk : this.recordings.concat(chunk);
      } catch (e) {
        this.listError = String(e.message || e);
      } finally {
        this.listLoading = false;
      }
    },

    loadMore() {
      if (!this.hasMore || this.listLoading) return;
      this.listOffset += 50;
      this.loadList(false);
    },

    onUploadFile(e) {
      const f = e.target.files && e.target.files[0];
      this.uploadFile = f || null;
    },

    async submitUpload() {
      if (!this.uploadFile) {
        this.uploadError = "Choose a file";
        return;
      }
      this.uploadBusy = true;
      this.uploadError = null;
      try {
        const rec = await NoobScribeApi.createRecording(this.uploadFile, this.uploadName || null);
        this.uploadFile = null;
        this.uploadName = "";
        this.go("/r/" + rec.id);
      } catch (e) {
        this.uploadError = String(e.message || e);
      } finally {
        this.uploadBusy = false;
      }
    },

    async loadRecording(id) {
      this.recLoading = true;
      this.recError = null;
      this.recording = null;
      try {
        this.recording = await NoobScribeApi.getRecording(id);
      } catch (e) {
        this.recError = String(e.message || e);
      } finally {
        this.recLoading = false;
      }
    },

    audioSrc(id) {
      return NoobScribeApi.recordingAudioUrl(id);
    },

    async runTranscribe() {
      if (!this.route.recordingId) return;
      this.transcribeBusy = true;
      this.transcribeError = null;
      try {
        await NoobScribeApi.transcribeRecording(this.route.recordingId, {
          language: this.transcribeOpts.language || null,
          diarize: this.transcribeOpts.diarize,
          word_timestamps: this.transcribeOpts.word_timestamps,
          include_diarization_in_text: this.transcribeOpts.include_diarization_in_text,
          response_format: "verbose_json",
          timestamps: true,
        });
        await this.loadRecording(this.route.recordingId);
      } catch (e) {
        this.transcribeError = String(e.message || e);
      } finally {
        this.transcribeBusy = false;
      }
    },

    async renameRecording() {
      if (!this.recording || !this.recording.name) return;
      try {
        await NoobScribeApi.patchRecording(this.recording.id, this.recording.name);
        await this.loadRecording(this.recording.id);
      } catch (e) {
        this.recError = String(e.message || e);
      }
    },

    async deleteRecording() {
      if (!this.recording || !confirm("Delete this recording and all transcriptions?")) return;
      try {
        await NoobScribeApi.deleteRecording(this.recording.id);
        this.go("/");
      } catch (e) {
        this.recError = String(e.message || e);
      }
    },

    async loadTranscription(rid, tid) {
      this.stopTranscriptSegmentPlayback();
      this.trLoading = true;
      this.trError = null;
      this.transcription = null;
      try {
        this.transcription = await NoobScribeApi.getTranscription(rid, tid);
      } catch (e) {
        this.trError = String(e.message || e);
      } finally {
        this.trLoading = false;
      }
    },

    stopTranscriptSegmentPlayback() {
      this.trSegmentPlayIndex = null;
      this.trSegmentIsPlaying = false;
      const a = this._trSegmentAudio;
      if (!a) return;
      a.pause();
      a.removeAttribute("src");
      try {
        a.load();
      } catch (e) {
        /* ignore */
      }
    },

    _ensureTrSegmentAudio() {
      if (this._trSegmentAudio) return this._trSegmentAudio;
      const a = new Audio();
      a.preload = "none";
      a.addEventListener("play", () => {
        this.trSegmentIsPlaying = true;
      });
      a.addEventListener("pause", () => {
        this.trSegmentIsPlaying = false;
      });
      a.addEventListener("ended", () => {
        this.trSegmentIsPlaying = false;
      });
      this._trSegmentAudio = a;
      return a;
    },

    transcriptSegmentSnippetRid() {
      return (
        (this.route && this.route.recordingId) ||
        (this.transcription && this.transcription.recording_id) ||
        null
      );
    },

    transcriptSegmentCanPlay(seg) {
      if (!seg) return false;
      const s = seg.start;
      const e = seg.end;
      if (s == null || e == null) return false;
      const sn = Number(s);
      const en = Number(e);
      if (Number.isNaN(sn) || Number.isNaN(en)) return false;
      return en > sn;
    },

    transcriptSegmentPlayButtonState(idx) {
      if (this.trSegmentPlayIndex === idx && this.trSegmentIsPlaying) return "pause";
      return "play";
    },

    toggleTranscriptSegmentPlay(idx) {
      const rid = this.transcriptSegmentSnippetRid();
      const segs = this.transcription && this.transcription.segments;
      if (!rid || !segs || idx < 0 || idx >= segs.length) return;
      const seg = segs[idx];
      if (!this.transcriptSegmentCanPlay(seg)) return;
      const audio = this._ensureTrSegmentAudio();
      const url = NoobScribeApi.audioSnippetUrl(rid, seg.start, seg.end);

      if (this.trSegmentPlayIndex === idx && !audio.paused) {
        audio.pause();
        return;
      }
      if (this.trSegmentPlayIndex === idx && audio.paused && audio.src) {
        audio.play().catch((e) => {
          console.warn(e);
          this.trSegmentIsPlaying = false;
        });
        return;
      }

      audio.pause();
      audio.src = url;
      this.trSegmentPlayIndex = idx;
      audio.play().catch((e) => {
        console.warn(e);
        this.trSegmentIsPlaying = false;
      });
    },

    async openSaveSpeaker(speakerRow) {
      this.saveSpeakerLabel = "";
      this.saveSpeakerTargetId = "";
      this.saveSpeakerEmbedding = speakerRow.embedding ? speakerRow.embedding.slice() : null;
      this.saveSpeakerError = null;
      if (this.route && this.route.name === "speakers") {
        this.speakerEmbeddingTab = "audio";
        this.speakerEmbeddingSectionExpanded = true;
      }
      await this.loadSpeakers();
    },

    cancelSaveSpeaker() {
      this.saveSpeakerEmbedding = null;
      this.saveSpeakerTargetId = "";
      this.saveSpeakerLabel = "";
      this.saveSpeakerError = null;
    },

    async confirmSaveSpeaker() {
      if (!this.saveSpeakerEmbedding) {
        this.saveSpeakerError = "Embedding required";
        return;
      }
      if (!this.saveSpeakerTargetId && !this.saveSpeakerLabel.trim()) {
        this.saveSpeakerError = "Enter a name for a new speaker or select an existing one";
        return;
      }
      this.saveSpeakerBusy = true;
      this.saveSpeakerError = null;
      try {
        if (this.saveSpeakerTargetId) {
          this._purgeSnippetCacheForSpeaker(this.saveSpeakerTargetId);
          await NoobScribeApi.addEmbeddingToSpeaker(this.saveSpeakerTargetId, this.saveSpeakerEmbedding);
        } else {
          await NoobScribeApi.createSpeaker(this.saveSpeakerLabel.trim(), this.saveSpeakerEmbedding);
        }
        this.saveSpeakerEmbedding = null;
        this.saveSpeakerTargetId = "";
        this.saveSpeakerLabel = "";
        await this.loadSpeakers();
        await this.refreshExpandedSpeakerEmbeddings();
      } catch (e) {
        this.saveSpeakerError = String(e.message || e);
      } finally {
        this.saveSpeakerBusy = false;
      }
    },

    async refreshExpandedSpeakerEmbeddings() {
      const ids = Object.keys(this.expandedSpeakers);
      for (const sid of ids) {
        if (this.expandedSpeakers[sid]) {
          await this.fetchSpeakerEmbeddings(sid);
        }
      }
    },

    teardownSpExtractCapture() {
      if (this._spExtractMediaRecorder && this._spExtractMediaRecorder.state !== "inactive") {
        try {
          this._spExtractMediaRecorder.stop();
        } catch (e) {
          /* ignore */
        }
      }
      this._spExtractMediaRecorder = null;
      if (this._spExtractStream) {
        for (const t of this._spExtractStream.getTracks()) {
          try {
            t.stop();
          } catch (e) {
            /* ignore */
          }
        }
        this._spExtractStream = null;
      }
      this.spExtractRecording = false;
      if (this.spExtractUrl) {
        try {
          URL.revokeObjectURL(this.spExtractUrl);
        } catch (e) {
          /* ignore */
        }
        this.spExtractUrl = null;
      }
      this.spExtractBlob = null;
      this.spExtractFile = null;
    },

    onSpExtractFileInput(e) {
      const f = e.target.files && e.target.files[0];
      this.spExtractFile = f || null;
      this.extractError = null;
    },

    clearExtractResult() {
      this.extractResult = null;
      this.extractError = null;
    },

    discardSpExtractRecording() {
      this.teardownSpExtractCapture();
      this.extractError = null;
    },

    async toggleSpExtractMicRecord() {
      this.extractError = null;
      if (this.spExtractRecording) {
        const mr = this._spExtractMediaRecorder;
        if (mr && mr.state !== "inactive") {
          mr.stop();
        }
        return;
      }
      if (this.spExtractUrl) {
        try {
          URL.revokeObjectURL(this.spExtractUrl);
        } catch (e) {
          /* ignore */
        }
        this.spExtractUrl = null;
      }
      this.spExtractBlob = null;
      this.spExtractFile = null;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this._spExtractStream = stream;
        const mr = new MediaRecorder(stream);
        const chunks = [];
        mr.ondataavailable = (ev) => {
          if (ev.data && ev.data.size) chunks.push(ev.data);
        };
        mr.onstop = () => {
          const blob = new Blob(chunks, { type: mr.mimeType || "audio/webm" });
          if (this.spExtractUrl) {
            try {
              URL.revokeObjectURL(this.spExtractUrl);
            } catch (e) {
              /* ignore */
            }
          }
          this.spExtractBlob = blob;
          this.spExtractUrl = URL.createObjectURL(blob);
          this.spExtractFile = null;
          if (this._spExtractStream) {
            for (const t of this._spExtractStream.getTracks()) {
              try {
                t.stop();
              } catch (e) {
                /* ignore */
              }
            }
            this._spExtractStream = null;
          }
          this._spExtractMediaRecorder = null;
          this.spExtractRecording = false;
        };
        this._spExtractMediaRecorder = mr;
        mr.start();
        this.spExtractRecording = true;
      } catch (e) {
        this.extractError = String(e.message || e);
        this.spExtractRecording = false;
        if (this._spExtractStream) {
          for (const t of this._spExtractStream.getTracks()) {
            try {
              t.stop();
            } catch (e2) {
              /* ignore */
            }
          }
          this._spExtractStream = null;
        }
      }
    },

    async runExtractFromAudio() {
      let file;
      if (this.spExtractBlob) {
        file = new File([this.spExtractBlob], "snippet.webm", {
          type: this.spExtractBlob.type || "audio/webm",
        });
      } else if (this.spExtractFile) {
        file = this.spExtractFile;
      } else {
        this.extractError = "Choose a file or record audio first";
        return;
      }
      this.extractBusy = true;
      this.extractError = null;
      try {
        const res = await NoobScribeApi.extractSpeakersFromAudio(file, {
          name: this.extractSnippetName.trim() || undefined,
          diarize: true,
          word_timestamps: false,
          timestamps: true,
          include_diarization_in_text: false,
        });
        this.extractResult = res;
        const speakers = res && res.speakers;
        const hasSpeakerEmbeddings =
          Array.isArray(speakers) &&
          speakers.some((s) => s && Array.isArray(s.embedding) && s.embedding.length > 0);
        if (!hasSpeakerEmbeddings) {
          this.extractError =
            "No speaker embeddings were returned. The clip may need clearer speech, diarization may be off or unavailable (set a Hugging Face token or DIARIZATION_MODEL_PATH), or try a longer snippet.";
        }
        await this.loadSpeakers();
      } catch (e) {
        this.extractError = String(e.message || e);
      } finally {
        this.extractBusy = false;
      }
    },

    extractSpeakerMap() {
      const map = Object.create(null);
      const t = this.extractResult;
      if (!t || !Array.isArray(t.speakers)) return map;
      for (const s of t.speakers) {
        if (!s || !s.id) continue;
        const name = s.display_name && String(s.display_name).trim();
        map[s.id] = name || s.id;
      }
      return map;
    },

    resolveExtractSpeaker(raw) {
      const id = this.normalizeSpeakerId(raw);
      if (!id || id === "unknown") return id || "";
      const map = this.extractSpeakerMap();
      return map[id] != null ? map[id] : id;
    },

    _stripDiarizationPrefixExtract(text, rawSpeaker) {
      let body = (text || "").trim();
      const sid = this.normalizeSpeakerId(rawSpeaker);
      if (!sid || sid === "unknown" || !body) return body;
      const re = new RegExp("^" + this._escapeRegExp(sid) + ":\\s*", "i");
      return body.replace(re, "").trim();
    },

    extractFormattedTranscript() {
      const t = this.extractResult;
      if (!t) return "";
      const segs = t.segments;
      if (Array.isArray(segs) && segs.length > 0) {
        const lines = segs.map((seg) => {
          const label = seg.speaker ? this.resolveExtractSpeaker(seg.speaker) : "";
          const body = this._stripDiarizationPrefixExtract(seg.text, seg.speaker);
          if (label && label !== "unknown") {
            return "[" + label + "] " + body;
          }
          return body;
        });
        return lines.join("\n\n");
      }
      const ft = t.text;
      return ft != null ? String(ft) : "";
    },

    async loadSpeakers() {
      this.spLoading = true;
      this.spError = null;
      try {
        const res = await NoobScribeApi.listSpeakers();
        this.speakers = res.data || [];
      } catch (e) {
        this.spError = String(e.message || e);
        this.speakers = [];
      } finally {
        this.spLoading = false;
      }
    },

    async deleteSpeaker(id) {
      if (!confirm("Delete this speaker?")) return;
      try {
        await NoobScribeApi.deleteSpeaker(id);
        const ex = { ...this.expandedSpeakers };
        delete ex[id];
        this.expandedSpeakers = ex;
        const emb = { ...this.speakerEmbeddingsBySpeaker };
        delete emb[id];
        this.speakerEmbeddingsBySpeaker = emb;
        this._purgeSnippetCacheForSpeaker(id);
        await this.loadSpeakers();
      } catch (e) {
        this.spError = String(e.message || e);
      }
    },

    speakerExpanded(sid) {
      return !!this.expandedSpeakers[sid];
    },

    async toggleSpeakerRow(sid) {
      const next = !this.expandedSpeakers[sid];
      this.expandedSpeakers = { ...this.expandedSpeakers, [sid]: next };
      if (next) {
        await this.fetchSpeakerEmbeddings(sid);
      }
    },

    async fetchSpeakerEmbeddings(sid) {
      this.embeddingsLoadingFor = sid;
      this.spError = null;
      try {
        const res = await NoobScribeApi.getSpeakerEmbeddings(sid);
        this.speakerEmbeddingsBySpeaker = {
          ...this.speakerEmbeddingsBySpeaker,
          [sid]: res.data || [],
        };
      } catch (e) {
        this.spError = String(e.message || e);
      } finally {
        this.embeddingsLoadingFor = null;
      }
    },

    embeddingsForSpeaker(sid) {
      return this.speakerEmbeddingsBySpeaker[sid] || [];
    },

    snippetKey(sid, embeddingIndex) {
      return sid + ":" + String(embeddingIndex);
    },

    embeddingSnippetsOpen(sid, embeddingIndex) {
      return this.snippetPanelKey === this.snippetKey(sid, embeddingIndex);
    },

    async toggleEmbeddingSnippets(sid, embeddingIndex) {
      const k = this.snippetKey(sid, embeddingIndex);
      if (this.snippetPanelKey === k) {
        this.snippetPanelKey = null;
        return;
      }
      this.snippetPanelKey = k;
      if (!(k in this.snippetsByKey)) {
        await this.fetchSnippets(sid, embeddingIndex);
      }
    },

    async fetchSnippets(sid, embeddingIndex) {
      const k = this.snippetKey(sid, embeddingIndex);
      this.snippetsLoadingKey = k;
      this.spError = null;
      try {
        const res = await NoobScribeApi.getEmbeddingSnippets(sid, embeddingIndex);
        this.snippetsByKey = { ...this.snippetsByKey, [k]: res.data || [] };
      } catch (e) {
        this.spError = String(e.message || e);
      } finally {
        this.snippetsLoadingKey = null;
      }
    },

    snippetsForEmbedding(sid, embeddingIndex) {
      const k = this.snippetKey(sid, embeddingIndex);
      return this.snippetsByKey[k] || [];
    },

    loadSnippetAudio(snippet) {
      if (!snippet || !snippet.preview_url) return;
      snippet._audioSrc = snippet.preview_url;
    },

    _purgeSnippetCacheForSpeaker(sid) {
      const prefix = sid + ":";
      const next = {};
      for (const key of Object.keys(this.snippetsByKey)) {
        if (!key.startsWith(prefix)) {
          next[key] = this.snippetsByKey[key];
        }
      }
      this.snippetsByKey = next;
      if (this.snippetPanelKey && this.snippetPanelKey.startsWith(prefix)) {
        this.snippetPanelKey = null;
      }
    },

    async removeSpeakerEmbedding(sid, embeddingIndex) {
      if (!confirm("Delete this embedding?")) return;
      this.spError = null;
      try {
        await NoobScribeApi.deleteSpeakerEmbedding(sid, embeddingIndex);
        this._purgeSnippetCacheForSpeaker(sid);
        await this.fetchSpeakerEmbeddings(sid);
        await this.loadSpeakers();
      } catch (e) {
        this.spError = String(e.message || e);
      }
    },

    async submitManualSpeaker() {
      let emb;
      try {
        emb = JSON.parse(this.manualEmbeddingJson);
        if (!Array.isArray(emb)) throw new Error("Embedding must be a JSON array of numbers");
      } catch (e) {
        this.spError = "Invalid JSON array: " + (e.message || e);
        return;
      }
      if (!this.manualTargetSpeakerId && !this.newSpeakerName.trim()) {
        this.spError = "Enter a display name or select an existing speaker";
        return;
      }
      this.manualBusy = true;
      this.spError = null;
      try {
        if (this.manualTargetSpeakerId) {
          this._purgeSnippetCacheForSpeaker(this.manualTargetSpeakerId);
          await NoobScribeApi.addEmbeddingToSpeaker(this.manualTargetSpeakerId, emb);
        } else {
          await NoobScribeApi.createSpeaker(this.newSpeakerName.trim(), emb);
        }
        this.newSpeakerName = "";
        this.manualEmbeddingJson = "";
        this.manualTargetSpeakerId = "";
        await this.loadSpeakers();
        await this.refreshExpandedSpeakerEmbeddings();
      } catch (e) {
        this.spError = String(e.message || e);
      } finally {
        this.manualBusy = false;
      }
    },

    formatDate(iso) {
      if (!iso) return "";
      try {
        return new Date(iso).toLocaleString();
      } catch (e) {
        return iso;
      }
    },

    /** Map diarization id (e.g. SPEAKER_00) -> display label (matched name or id). */
    speakerMapFromTranscription() {
      const map = Object.create(null);
      const t = this.transcription;
      if (!t || !Array.isArray(t.speakers)) return map;
      for (const s of t.speakers) {
        if (!s || !s.id) continue;
        const name = s.display_name && String(s.display_name).trim();
        map[s.id] = name || s.id;
      }
      return map;
    },

    normalizeSpeakerId(raw) {
      if (raw == null || raw === "") return "";
      let id = String(raw);
      if (id === "unknown") return "unknown";
      if (id.startsWith("speaker_")) id = id.replace(/^speaker_/, "");
      return id;
    },

    /** Resolved label for transcript UI (uses matched display_name from speakers[]). */
    resolveSpeaker(raw) {
      const id = this.normalizeSpeakerId(raw);
      if (!id || id === "unknown") return id || "";
      const map = this.speakerMapFromTranscription();
      return map[id] != null ? map[id] : id;
    },

    _escapeRegExp(s) {
      return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    },

    /** Strip leading "SPEAKER_XX: " from segment text when it duplicates the diarization label. */
    _stripDiarizationPrefix(text, rawSpeaker) {
      let body = (text || "").trim();
      const sid = this.normalizeSpeakerId(rawSpeaker);
      if (!sid || sid === "unknown" || !body) return body;
      const re = new RegExp("^" + this._escapeRegExp(sid) + ":\\s*", "i");
      return body.replace(re, "").trim();
    },

    /** Full transcript with line breaks between segments and resolved speaker labels. */
    formattedTranscript() {
      const t = this.transcription;
      if (!t) return "";
      const segs = t.segments;
      if (Array.isArray(segs) && segs.length > 0) {
        const lines = segs.map((seg) => {
          const label = seg.speaker ? this.resolveSpeaker(seg.speaker) : "";
          const body = this._stripDiarizationPrefix(seg.text, seg.speaker);
          if (label && label !== "unknown") {
            return "[" + label + "] " + body;
          }
          return body;
        });
        return lines.join("\n\n");
      }
      const ft = t.full_text;
      return ft != null ? String(ft) : "";
    },

    downloadTranscription() {
      if (!this.transcription) return;
      const text = this.formattedTranscript();
      const id = this.transcription.id || "transcript";
      const short = String(id).replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 36) || "transcript";
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "transcription-" + short + ".txt";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    },
  };
}

document.addEventListener("alpine:init", () => {
  Alpine.data("noobScribeApp", noobScribeApp);
});
