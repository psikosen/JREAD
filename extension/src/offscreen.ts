import {Chunker, Clock, Packet, PendingPackets, SourceLanguage, Target, mediaNow, validClock, validCue} from "./core";

interface LiveState { active: boolean; stopping?: boolean; detail: string; target?: Target; job_id?: string; delay?: number; language?: SourceLanguage }
let state: LiveState = {active: false, detail: "Capture is off."};
let stream: MediaStream | undefined;
let context: AudioContext | undefined;
let source: MediaStreamAudioSourceNode | undefined;
let processor: AudioWorkletNode | undefined;
let socket: WebSocket | undefined;
let watchdog: ReturnType<typeof setInterval> | undefined;
let finishTimer: ReturnType<typeof setTimeout> | undefined;
let clock: Clock | undefined;
let chunks = new Chunker();
let pending = new PendingPackets();
let starting = false;
let generation = 0;
function publish() { void chrome.runtime.sendMessage({target: "worker", type: "live-status", state}).catch(() => {}); }
function send(message: unknown) { if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message)); }
async function stop(detail: string) {
  ++generation;
  state = {...state, active: false, stopping: false, detail};
  if (watchdog) clearInterval(watchdog); watchdog = undefined;
  if (finishTimer) clearTimeout(finishTimer); finishTimer = undefined;
  processor?.port.postMessage({running: false, epoch: pending.epoch});
  processor?.disconnect(); source?.disconnect();
  stream?.getTracks().forEach((track) => track.stop()); stream = undefined;
  const oldContext = context; context = undefined; processor = undefined; source = undefined;
  send({type: "stop"});
  if (socket) { socket.onclose = socket.onerror = socket.onmessage = null; socket.close(); socket = undefined; }
  chunks.reset(); pending.reset(); clock = undefined;
  if (oldContext) await oldContext.close().catch(() => {});
  publish();
}
async function finishSending() {
  try { chunks.flush(emit); } catch (error) { await stop((error as Error).message); return; }
  if (finishTimer) clearTimeout(finishTimer);
  processor?.disconnect(); source?.disconnect();
  const oldContext = context; context = undefined; processor = undefined; source = undefined;
  if (oldContext) await oldContext.close().catch(() => {});
  if (!state.stopping) return;
  send({type: "stop"});
  finishTimer = setTimeout(() => { void stop("Capture stopped, but the companion did not finish queued captions. The library job is incomplete; check the companion."); }, 180000);
}
async function gracefulStop() {
  if (!state.active || state.stopping) return state;
  if (starting) { await stop("Capture cancelled before listening began."); return state; }
  state = {...state, stopping: true, detail: "Capture stopped. Finishing the remaining captions in the companion…"}; publish();
  if (watchdog) clearInterval(watchdog); watchdog = undefined;
  source?.disconnect(); stream?.getTracks().forEach((track) => track.stop()); stream = undefined;
  if (processor) {
    processor.port.postMessage({running: false, epoch: pending.epoch, flush: true, final: true});
    finishTimer = setTimeout(() => { void stop("Capture stopped, but the final audio buffer could not be flushed. The library job is incomplete."); }, 2000);
  } else await finishSending();
  return state;
}
function emit(packet: Packet) {
  if (!state.active || !socket || socket.readyState !== WebSocket.OPEN) return;
  if (socket.bufferedAmount > 1024 * 1024) throw new Error("The local connection is too slow. Capture stopped to avoid dropping dialogue.");
  const seq = pending.send();
  send({type: "audio", epoch: pending.epoch, seq, media_time: packet.media_time, sample_rate: 16000, rate: packet.rate});
  // Explicit little-endian wire format; never rely on the platform's typed-array byte order.
  const bytes = new ArrayBuffer(packet.pcm.length * 2); const view = new DataView(bytes);
  packet.pcm.forEach((sample, i) => view.setInt16(i * 2, sample, true));
  socket.send(bytes);
}
async function updateClock(next: Clock) {
  if (!state.active || state.stopping || !clock || !validClock(next) || next.videoId !== clock.videoId || next.serial <= clock.serial) return;
  if (next.mediaId !== clock.mediaId) { await stop("The media changed. Select the new video to continue."); return; }
  const wasPaused = clock.paused;
  const reset = next.reason === "seeking" || next.rate !== clock.rate;
  clock = next;
  if (next.reason === "ended") { await gracefulStop(); return; }
  if (reset) {
    chunks.reset(); const epoch = pending.reset();
    processor?.port.postMessage({running: false, epoch});
    send({type: "reset", epoch});
    await chrome.runtime.sendMessage({target: "worker", type: "live-epoch", epoch});
  }
  processor?.port.postMessage({running: !next.paused, epoch: pending.epoch, flush: !reset && !wasPaused && next.paused});
  if (next.paused !== wasPaused) {
    state.detail = next.paused ? "Paused. Resume the selected video to continue listening." : "Listening to tab audio. Captions arrive after speech.";
    publish();
  }
}
async function start(message: {streamId: string; token: string; selection: Target; clock: Clock; title: string; language?: SourceLanguage}) {
  if (starting || state.active) throw new Error("A capture is already running. Stop it first.");
  if (!validClock(message.clock)) throw new Error("The selected video has an unsupported playback rate or timing state.");
  starting = true;
  const run = ++generation;
  try {
    clock = message.clock; chunks = new Chunker(); pending = new PendingPackets();
    state = {active: true, detail: "Connecting to local inference…", target: message.selection, language: message.language ?? "ja"}; publish();
    socket = new WebSocket("ws://127.0.0.1:8765/api/live");
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("The local companion did not accept capture within 10 seconds.")), 10000);
      socket!.onopen = () => send({type: "hello", token: message.token, session_id: crypto.randomUUID(), media_id: message.clock.mediaId, epoch: 0, language: state.language, title: message.title});
      socket!.onerror = () => { clearTimeout(timeout); reject(new Error("Cannot connect to the local companion. Start it and pair again.")); };
      socket!.onclose = () => { clearTimeout(timeout); reject(new Error("The local companion closed the connection.")); };
      socket!.onmessage = ({data}) => {
        try {
          const event = JSON.parse(data);
          if (event.type === "error") { clearTimeout(timeout); reject(new Error(String(event.detail))); }
          if (event.type === "ready") { clearTimeout(timeout); state.job_id = event.job_id; resolve(); }
        } catch { clearTimeout(timeout); reject(new Error("Invalid response from the local companion.")); }
      };
    });
    if (generation !== run) throw new Error("Capture was stopped.");
    socket.onclose = () => { if (generation === run) void stop("The local companion disconnected. Capture stopped; saved cues remain in your library."); };
    socket.onerror = () => { if (generation === run) void stop("Local audio transport failed. Capture stopped."); };
    socket.onmessage = ({data}) => {
      if (generation !== run) return;
      try {
        const event = JSON.parse(data);
        if (event.type === "error") { void stop(String(event.detail)); return; }
        if (event.type === "stopped") { void stop("Capture finished. Saved timed cues are available for replay and in your library."); return; }
        if (event.type === "ack") pending.ack(event.seq);
        if (event.type === "cue" && pending.accepts(event.epoch) && validCue(event.cue)) {
          state.delay = Math.max(0, (clock ? mediaNow(clock) : event.cue.end) - event.cue.end);
          if (!state.stopping) state.detail = `Live captions • ${state.delay.toFixed(1)} seconds of media delay`; publish();
          void chrome.runtime.sendMessage({target: "worker", type: "live-cue", epoch: event.epoch, cue: event.cue}).catch(() => {});
        }
      } catch { void stop("The companion returned an invalid live response."); }
    };
    stream = await navigator.mediaDevices.getUserMedia({audio: {mandatory: {chromeMediaSource: "tab", chromeMediaSourceId: message.streamId}} as MediaTrackConstraints, video: false});
    if (generation !== run) { stream.getTracks().forEach((track) => track.stop()); throw new Error("Capture was stopped."); }
    context = new AudioContext();
    source = context.createMediaStreamSource(stream);
    source.connect(context.destination);
    await context.audioWorklet.addModule(chrome.runtime.getURL("dist/worklet.js"));
    if (generation !== run) throw new Error("Capture was stopped.");
    processor = new AudioWorkletNode(context, "jread-pcm");
    const silence = context.createGain(); silence.gain.value = 0;
    source.connect(processor); processor.connect(silence); silence.connect(context.destination);
    processor.port.onmessage = ({data}) => {
      if (generation !== run || !state.active || !clock || data.epoch !== pending.epoch) return;
      try {
        if (data.flushed) { if (state.stopping && data.final) void finishSending(); else chunks.flush(emit); return; }
        const pcm = new Int16Array(data.pcm);
        chunks.push(pcm, Math.max(0, mediaNow(clock) - pcm.length / 16000 * clock.rate), clock.rate, emit);
      } catch (error) { void stop((error as Error).message); }
    };
    stream.getTracks().forEach((track) => track.addEventListener("ended", () => { if (generation === run && state.active && !state.stopping) void stop("The browser ended tab capture. The library may contain an incomplete track."); }));
    await context.resume();
    processor.port.postMessage({running: !clock!.paused, epoch: pending.epoch});
    state.detail = clock!.paused ? "Capture ready. Play the selected video; all tab audio is included." : "Listening to tab audio. Captions arrive after speech."; publish();
    watchdog = setInterval(() => {
      if (clock && Date.now() - clock.wall_time > 5000) void stop("The video stopped reporting its playback time. Capture stopped to prevent mistimed captions.");
    }, 1000);
    return state;
  } catch (error) { await stop((error as Error).message); throw error; }
  finally { starting = false; }
}
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message?.target !== "offscreen" || sender.id !== chrome.runtime.id || sender.tab) return;
  const execute = async () => {
    switch (message.type) {
      case "start": return start(message);
      case "clock": return updateClock(message.clock);
      case "stop": return message.detail ? stop(message.detail) : gracefulStop();
      case "state": return state;
      default: throw new Error("Unknown capture command.");
    }
  };
  void execute().then((data) => respond({ok: true, data}), (error) => respond({ok: false, error: (error as Error).message}));
  return true;
});
