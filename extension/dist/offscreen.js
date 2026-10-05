"use strict";
(() => {
  // extension/src/core.ts
  function validCue(cue) {
    return !!cue && typeof cue.text === "string" && cue.text.length <= 1e4 && Number.isFinite(cue.start) && Number.isFinite(cue.end) && cue.start >= 0 && cue.end > cue.start;
  }
  function validClock(clock2) {
    return !!clock2 && typeof clock2.videoId === "string" && typeof clock2.mediaId === "string" && typeof clock2.reason === "string" && typeof clock2.paused === "boolean" && Number.isSafeInteger(clock2.serial) && Number.isFinite(clock2.time) && clock2.time >= 0 && Number.isFinite(clock2.wall_time) && Number.isFinite(clock2.rate) && clock2.rate >= 0.25 && clock2.rate <= 4;
  }
  function mediaNow(clock2, now = Date.now()) {
    return clock2.time + (clock2.paused ? 0 : Math.max(0, now - clock2.wall_time) / 1e3 * clock2.rate);
  }
  var Chunker = class {
    samples = new Int16Array(128e3);
    size = 0;
    fresh = 0;
    start = 0;
    rate = 1;
    reset() {
      this.size = this.fresh = 0;
    }
    push(input, mediaTime, rate, emit2) {
      let offset = 0;
      while (offset < input.length) {
        if (!this.size) {
          this.start = mediaTime + offset / 16e3 * rate;
          this.rate = rate;
        }
        const count = Math.min(input.length - offset, this.samples.length - this.size);
        this.samples.set(input.subarray(offset, offset + count), this.size);
        this.size += count;
        this.fresh += count;
        offset += count;
        if (this.size === this.samples.length) {
          emit2({ pcm: this.samples.slice(), media_time: this.start, rate: this.rate });
          this.samples.copyWithin(0, this.samples.length - 16e3);
          this.size = 16e3;
          this.fresh = 0;
          this.start += 7 * this.rate;
        }
      }
    }
    flush(emit2) {
      if (this.fresh) emit2({ pcm: this.samples.slice(0, this.size), media_time: this.start, rate: this.rate });
      this.reset();
    }
  };
  var PendingPackets = class {
    pending = /* @__PURE__ */ new Set();
    epoch = 0;
    seq = 0;
    send() {
      if (this.pending.size >= 3) throw new Error("Live inference fell behind (3 audio chunks waiting). Capture stopped; use Generate ahead in the companion or a faster model/device.");
      const seq = this.seq++;
      this.pending.add(seq);
      return seq;
    }
    ack(seq) {
      this.pending.delete(seq);
    }
    reset() {
      this.pending.clear();
      return ++this.epoch;
    }
    accepts(epoch) {
      return epoch === this.epoch;
    }
  };

  // extension/src/offscreen.ts
  var state = { active: false, detail: "Capture is off." };
  var stream;
  var context;
  var source;
  var processor;
  var socket;
  var watchdog;
  var finishTimer;
  var clock;
  var chunks = new Chunker();
  var pending = new PendingPackets();
  var starting = false;
  var generation = 0;
  function publish() {
    void chrome.runtime.sendMessage({ target: "worker", type: "live-status", state }).catch(() => {
    });
  }
  function send(message) {
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  }
  async function stop(detail) {
    ++generation;
    state = { ...state, active: false, stopping: false, detail };
    if (watchdog) clearInterval(watchdog);
    watchdog = void 0;
    if (finishTimer) clearTimeout(finishTimer);
    finishTimer = void 0;
    processor?.port.postMessage({ running: false, epoch: pending.epoch });
    processor?.disconnect();
    source?.disconnect();
    stream?.getTracks().forEach((track) => track.stop());
    stream = void 0;
    const oldContext = context;
    context = void 0;
    processor = void 0;
    source = void 0;
    send({ type: "stop" });
    if (socket) {
      socket.onclose = socket.onerror = socket.onmessage = null;
      socket.close();
      socket = void 0;
    }
    chunks.reset();
    pending.reset();
    clock = void 0;
    if (oldContext) await oldContext.close().catch(() => {
    });
    publish();
  }
  async function finishSending() {
    try {
      chunks.flush(emit);
    } catch (error) {
      await stop(error.message);
      return;
    }
    if (finishTimer) clearTimeout(finishTimer);
    processor?.disconnect();
    source?.disconnect();
    const oldContext = context;
    context = void 0;
    processor = void 0;
    source = void 0;
    if (oldContext) await oldContext.close().catch(() => {
    });
    if (!state.stopping) return;
    send({ type: "stop" });
    finishTimer = setTimeout(() => {
      void stop("Capture stopped, but the companion did not finish queued captions. The library job is incomplete; check the companion.");
    }, 18e4);
  }
  async function gracefulStop() {
    if (!state.active || state.stopping) return state;
    if (starting) {
      await stop("Capture cancelled before listening began.");
      return state;
    }
    state = { ...state, stopping: true, detail: "Capture stopped. Finishing the remaining captions in the companion\u2026" };
    publish();
    if (watchdog) clearInterval(watchdog);
    watchdog = void 0;
    source?.disconnect();
    stream?.getTracks().forEach((track) => track.stop());
    stream = void 0;
    if (processor) {
      processor.port.postMessage({ running: false, epoch: pending.epoch, flush: true, final: true });
      finishTimer = setTimeout(() => {
        void stop("Capture stopped, but the final audio buffer could not be flushed. The library job is incomplete.");
      }, 2e3);
    } else await finishSending();
    return state;
  }
  function emit(packet) {
    if (!state.active || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (socket.bufferedAmount > 1024 * 1024) throw new Error("The local connection is too slow. Capture stopped to avoid dropping dialogue.");
    const seq = pending.send();
    send({ type: "audio", epoch: pending.epoch, seq, media_time: packet.media_time, sample_rate: 16e3, rate: packet.rate });
    const bytes = new ArrayBuffer(packet.pcm.length * 2);
    const view = new DataView(bytes);
    packet.pcm.forEach((sample, i) => view.setInt16(i * 2, sample, true));
    socket.send(bytes);
  }
  async function updateClock(next) {
    if (!state.active || state.stopping || !clock || !validClock(next) || next.videoId !== clock.videoId || next.serial <= clock.serial) return;
    if (next.mediaId !== clock.mediaId) {
      await stop("The media changed. Select the new video to continue.");
      return;
    }
    const wasPaused = clock.paused;
    const reset = next.reason === "seeking" || next.rate !== clock.rate;
    clock = next;
    if (next.reason === "ended") {
      await gracefulStop();
      return;
    }
    if (reset) {
      chunks.reset();
      const epoch = pending.reset();
      processor?.port.postMessage({ running: false, epoch });
      send({ type: "reset", epoch });
      await chrome.runtime.sendMessage({ target: "worker", type: "live-epoch", epoch });
    }
    processor?.port.postMessage({ running: !next.paused, epoch: pending.epoch, flush: !reset && !wasPaused && next.paused });
    if (next.paused !== wasPaused) {
      state.detail = next.paused ? "Paused. Resume the selected video to continue listening." : "Listening to tab audio. Captions arrive after speech.";
      publish();
    }
  }
  async function start(message) {
    if (starting || state.active) throw new Error("A capture is already running. Stop it first.");
    if (!validClock(message.clock)) throw new Error("The selected video has an unsupported playback rate or timing state.");
    starting = true;
    const run = ++generation;
    try {
      clock = message.clock;
      chunks = new Chunker();
      pending = new PendingPackets();
      state = { active: true, detail: "Connecting to local inference\u2026", target: message.selection, language: message.language ?? "ja" };
      publish();
      socket = new WebSocket("ws://127.0.0.1:8765/api/live");
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("The local companion did not accept capture within 10 seconds.")), 1e4);
        socket.onopen = () => send({ type: "hello", token: message.token, session_id: crypto.randomUUID(), media_id: message.clock.mediaId, epoch: 0, language: state.language, title: message.title });
        socket.onerror = () => {
          clearTimeout(timeout);
          reject(new Error("Cannot connect to the local companion. Start it and pair again."));
        };
        socket.onclose = () => {
          clearTimeout(timeout);
          reject(new Error("The local companion closed the connection."));
        };
        socket.onmessage = ({ data }) => {
          try {
            const event = JSON.parse(data);
            if (event.type === "error") {
              clearTimeout(timeout);
              reject(new Error(String(event.detail)));
            }
            if (event.type === "ready") {
              clearTimeout(timeout);
              state.job_id = event.job_id;
              resolve();
            }
          } catch {
            clearTimeout(timeout);
            reject(new Error("Invalid response from the local companion."));
          }
        };
      });
      if (generation !== run) throw new Error("Capture was stopped.");
      socket.onclose = () => {
        if (generation === run) void stop("The local companion disconnected. Capture stopped; saved cues remain in your library.");
      };
      socket.onerror = () => {
        if (generation === run) void stop("Local audio transport failed. Capture stopped.");
      };
      socket.onmessage = ({ data }) => {
        if (generation !== run) return;
        try {
          const event = JSON.parse(data);
          if (event.type === "error") {
            void stop(String(event.detail));
            return;
          }
          if (event.type === "stopped") {
            void stop("Capture finished. Saved timed cues are available for replay and in your library.");
            return;
          }
          if (event.type === "ack") pending.ack(event.seq);
          if (event.type === "cue" && pending.accepts(event.epoch) && validCue(event.cue)) {
            state.delay = Math.max(0, (clock ? mediaNow(clock) : event.cue.end) - event.cue.end);
            if (!state.stopping) state.detail = `Live captions \u2022 ${state.delay.toFixed(1)} seconds of media delay`;
            publish();
            void chrome.runtime.sendMessage({ target: "worker", type: "live-cue", epoch: event.epoch, cue: event.cue }).catch(() => {
            });
          }
        } catch {
          void stop("The companion returned an invalid live response.");
        }
      };
      stream = await navigator.mediaDevices.getUserMedia({ audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: message.streamId } }, video: false });
      if (generation !== run) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("Capture was stopped.");
      }
      context = new AudioContext();
      source = context.createMediaStreamSource(stream);
      source.connect(context.destination);
      await context.audioWorklet.addModule(chrome.runtime.getURL("dist/worklet.js"));
      if (generation !== run) throw new Error("Capture was stopped.");
      processor = new AudioWorkletNode(context, "jread-pcm");
      const silence = context.createGain();
      silence.gain.value = 0;
      source.connect(processor);
      processor.connect(silence);
      silence.connect(context.destination);
      processor.port.onmessage = ({ data }) => {
        if (generation !== run || !state.active || !clock || data.epoch !== pending.epoch) return;
        try {
          if (data.flushed) {
            if (state.stopping && data.final) void finishSending();
            else chunks.flush(emit);
            return;
          }
          const pcm = new Int16Array(data.pcm);
          chunks.push(pcm, Math.max(0, mediaNow(clock) - pcm.length / 16e3 * clock.rate), clock.rate, emit);
        } catch (error) {
          void stop(error.message);
        }
      };
      stream.getTracks().forEach((track) => track.addEventListener("ended", () => {
        if (generation === run && state.active && !state.stopping) void stop("The browser ended tab capture. The library may contain an incomplete track.");
      }));
      await context.resume();
      processor.port.postMessage({ running: !clock.paused, epoch: pending.epoch });
      state.detail = clock.paused ? "Capture ready. Play the selected video; all tab audio is included." : "Listening to tab audio. Captions arrive after speech.";
      publish();
      watchdog = setInterval(() => {
        if (clock && Date.now() - clock.wall_time > 5e3) void stop("The video stopped reporting its playback time. Capture stopped to prevent mistimed captions.");
      }, 1e3);
      return state;
    } catch (error) {
      await stop(error.message);
      throw error;
    } finally {
      starting = false;
    }
  }
  chrome.runtime.onMessage.addListener((message, sender, respond) => {
    if (message?.target !== "offscreen" || sender.id !== chrome.runtime.id || sender.tab) return;
    const execute = async () => {
      switch (message.type) {
        case "start":
          return start(message);
        case "clock":
          return updateClock(message.clock);
        case "stop":
          return message.detail ? stop(message.detail) : gracefulStop();
        case "state":
          return state;
        default:
          throw new Error("Unknown capture command.");
      }
    };
    void execute().then((data) => respond({ ok: true, data }), (error) => respond({ ok: false, error: error.message }));
    return true;
  });
})();
//# sourceMappingURL=offscreen.js.map
