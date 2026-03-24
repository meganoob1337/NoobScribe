/**
 * Browser capture: window/tab audio (getDisplayMedia) + microphone (getUserMedia),
 * mixed via Web Audio API, recorded with MediaRecorder (WebM/Opus).
 */
(function (global) {
  "use strict";

  function pickMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "video/webm;codecs=opus",
      "video/webm",
    ];
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported) {
      for (const t of candidates) {
        if (MediaRecorder.isTypeSupported(t)) return t;
      }
    }
    return "";
  }

  function rmsFromAnalyser(analyser, scratch) {
    if (!analyser) return 0;
    const len = analyser.frequencyBinCount;
    if (!scratch || scratch.length !== len) {
      scratch = new Uint8Array(len);
    }
    analyser.getByteTimeDomainData(scratch);
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const v = (scratch[i] - 128) / 128;
      sum += v * v;
    }
    return Math.min(1, Math.sqrt(sum / len) * 2.5);
  }

  class NoobScribeRecorder {
    constructor() {
      this._ctx = null;
      this._destination = null;
      this._windowStream = null;
      this._micStream = null;
      this._windowSource = null;
      this._micSource = null;
      this._windowGain = null;
      this._micGain = null;
      this._windowAnalyser = null;
      this._micAnalyser = null;
      this._windowScratch = null;
      this._micScratch = null;
      this._mediaRecorder = null;
      this._chunks = [];
      this._mimeType = "";
    }

    _ensureContext() {
      if (this._ctx) return;
      const Ctx = global.AudioContext || global.webkitAudioContext;
      if (!Ctx) throw new Error("Web Audio API not supported");
      this._ctx = new Ctx();
      this._destination = this._ctx.createMediaStreamDestination();
    }

    get windowActive() {
      return !!(this._windowStream && this._windowStream.getAudioTracks().some((t) => t.readyState === "live"));
    }

    get micActive() {
      return !!(this._micStream && this._micStream.getAudioTracks().some((t) => t.readyState === "live"));
    }

    get hasLiveAudioSource() {
      return this.windowActive || this.micActive;
    }

    async startWindowCapture() {
      this._ensureContext();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
        throw new Error("Screen/window capture not supported in this browser");
      }
      this.stopWindowCapture();
      const stream = await navigator.mediaDevices.getDisplayMedia({
        audio: true,
        video: true,
      });
      const audioTracks = stream.getAudioTracks();
      if (!audioTracks.length) {
        stream.getTracks().forEach((t) => t.stop());
        throw new Error("No audio track from selection. Try a browser tab or full screen.");
      }
      this._windowStream = stream;
      const audioOnly = new MediaStream(audioTracks);
      this._windowSource = this._ctx.createMediaStreamSource(audioOnly);
      this._windowGain = this._ctx.createGain();
      this._windowGain.gain.value = 1;
      this._windowAnalyser = this._ctx.createAnalyser();
      this._windowAnalyser.fftSize = 256;
      this._windowAnalyser.smoothingTimeConstant = 0.8;
      this._windowSource.connect(this._windowGain);
      this._windowGain.connect(this._windowAnalyser);
      this._windowGain.connect(this._destination);
      stream.getVideoTracks().forEach((t) => t.stop());
      const onEnded = () => {
        this.stopWindowCapture();
      };
      audioTracks.forEach((t) => t.addEventListener("ended", onEnded));
      return true;
    }

    stopWindowCapture() {
      if (this._windowStream) {
        this._windowStream.getTracks().forEach((t) => t.stop());
        this._windowStream = null;
      }
      if (this._windowSource) {
        try {
          this._windowSource.disconnect();
        } catch (e) {
          /* ignore */
        }
        this._windowSource = null;
      }
      if (this._windowGain) {
        try {
          this._windowGain.disconnect();
        } catch (e) {
          /* ignore */
        }
        this._windowGain = null;
      }
      if (this._windowAnalyser) {
        try {
          this._windowAnalyser.disconnect();
        } catch (e) {
          /* ignore */
        }
        this._windowAnalyser = null;
      }
    }

    async startMicCapture() {
      this._ensureContext();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Microphone access not supported");
      }
      this.stopMicCapture();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      this._micStream = stream;
      this._micSource = this._ctx.createMediaStreamSource(stream);
      this._micGain = this._ctx.createGain();
      this._micGain.gain.value = 1;
      this._micAnalyser = this._ctx.createAnalyser();
      this._micAnalyser.fftSize = 256;
      this._micAnalyser.smoothingTimeConstant = 0.8;
      this._micSource.connect(this._micGain);
      this._micGain.connect(this._micAnalyser);
      this._micGain.connect(this._destination);
      return true;
    }

    stopMicCapture() {
      if (this._micStream) {
        this._micStream.getTracks().forEach((t) => t.stop());
        this._micStream = null;
      }
      if (this._micSource) {
        try {
          this._micSource.disconnect();
        } catch (e) {
          /* ignore */
        }
        this._micSource = null;
      }
      if (this._micGain) {
        try {
          this._micGain.disconnect();
        } catch (e) {
          /* ignore */
        }
        this._micGain = null;
      }
      if (this._micAnalyser) {
        try {
          this._micAnalyser.disconnect();
        } catch (e) {
          /* ignore */
        }
        this._micAnalyser = null;
      }
    }

    /** Returns 0–1 for window analyser (uses internal scratch). */
    readWindowLevel() {
      if (!this._windowAnalyser) return 0;
      const len = this._windowAnalyser.frequencyBinCount;
      if (!this._windowScratch || this._windowScratch.length !== len) {
        this._windowScratch = new Uint8Array(len);
      }
      return rmsFromAnalyser(this._windowAnalyser, this._windowScratch);
    }

    /** Returns 0–1 for mic analyser. */
    readMicLevel() {
      if (!this._micAnalyser) return 0;
      const len = this._micAnalyser.frequencyBinCount;
      if (!this._micScratch || this._micScratch.length !== len) {
        this._micScratch = new Uint8Array(len);
      }
      return rmsFromAnalyser(this._micAnalyser, this._micScratch);
    }

    startRecording() {
      this._ensureContext();
      if (!this.hasLiveAudioSource) {
        throw new Error("Enable at least one audio source before recording");
      }
      if (this._ctx.state === "suspended") {
        this._ctx.resume();
      }
      const out = this._destination.stream;
      if (!out.getAudioTracks().length) {
        throw new Error("No mixed audio output");
      }
      this._mimeType = pickMimeType();
      const options = this._mimeType ? { mimeType: this._mimeType } : {};
      if (typeof MediaRecorder === "undefined") {
        throw new Error("MediaRecorder not supported");
      }
      this._chunks = [];
      this._mediaRecorder = new MediaRecorder(out, options);
      this._mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) this._chunks.push(e.data);
      };
      this._mediaRecorder.start(250);
    }

    stopRecording() {
      return new Promise((resolve, reject) => {
        const mr = this._mediaRecorder;
        if (!mr || mr.state === "inactive") {
          reject(new Error("Not recording"));
          return;
        }
        mr.onstop = () => {
          const type = mr.mimeType || this._mimeType || "audio/webm";
          const blob = new Blob(this._chunks, { type: type });
          this._chunks = [];
          this._mediaRecorder = null;
          resolve(blob);
        };
        mr.onerror = (e) => {
          reject(e.error || new Error("Recording failed"));
        };
        try {
          mr.stop();
        } catch (e) {
          reject(e);
        }
      });
    }

    get isRecording() {
      return !!(this._mediaRecorder && this._mediaRecorder.state === "recording");
    }

    destroy() {
      if (this._mediaRecorder && this._mediaRecorder.state !== "inactive") {
        try {
          this._mediaRecorder.stop();
        } catch (e) {
          /* ignore */
        }
      }
      this._mediaRecorder = null;
      this._chunks = [];
      this.stopWindowCapture();
      this.stopMicCapture();
      if (this._destination) {
        try {
          this._destination.disconnect();
        } catch (e) {
          /* ignore */
        }
        this._destination = null;
      }
      if (this._ctx) {
        this._ctx.close().catch(() => {});
        this._ctx = null;
      }
    }
  }

  global.NoobScribeRecorder = NoobScribeRecorder;
})(typeof window !== "undefined" ? window : globalThis);
