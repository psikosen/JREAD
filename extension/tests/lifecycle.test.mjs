import assert from "node:assert/strict";
import {build} from "esbuild";
import vm from "node:vm";
import {webcrypto} from "node:crypto";

// Exercise the actual offscreen message handler with only browser IO replaced.
const result = await build({entryPoints: [new URL("../src/offscreen.ts", import.meta.url).pathname], bundle: true, write: false, platform: "browser", format: "iife"});
const events = [];
const sockets = [];
const tracks = [];
let handler;
let processor;
class Socket {
  static OPEN = 1;
  readyState = 0;
  bufferedAmount = 0;
  sent = [];
  constructor() { sockets.push(this); queueMicrotask(() => { this.readyState = 1; this.onopen?.(); }); }
  send(data) {
    this.sent.push(typeof data === "string" ? JSON.parse(data) : data);
    if (typeof data === "string" && JSON.parse(data).type === "hello") queueMicrotask(() => this.receive({type: "ready", job_id: "test-job"}));
  }
  receive(data) { this.onmessage?.({data: JSON.stringify(data)}); }
  close() { this.readyState = 3; this.onclose?.(); }
}
class Node {
  connect() {}
  disconnect() {}
}
class Context {
  destination = new Node();
  audioWorklet = {addModule: async () => {}};
  createMediaStreamSource() { return new Node(); }
  createGain() { return Object.assign(new Node(), {gain: {value: 0}}); }
  async close() {}
  async resume() {}
}
class Processor extends Node {
  port = {onmessage: null, postMessage: (data) => {
    if (data.flush) queueMicrotask(() => this.port.onmessage?.({data: {epoch: data.epoch, flushed: true, final: data.final === true}}));
  }};
  constructor() { super(); processor = this; }
  audio(epoch, samples) { this.port.onmessage({data: {epoch, pcm: new Int16Array(samples).buffer}}); }
}
vm.runInNewContext(result.outputFiles[0].text, {
  chrome: {runtime: {id: "extension", getURL: (path) => `chrome-extension://extension/${path}`, onMessage: {addListener: (value) => { handler = value; }}, sendMessage: async (event) => { events.push(structuredClone(event)); return {ok: true}; }}},
  WebSocket: Socket, AudioContext: Context, AudioWorkletNode: Processor,
  navigator: {mediaDevices: {getUserMedia: async () => {
    const track = {stopped: false, stop() { this.stopped = true; }, addEventListener() {}};
    tracks.push(track); return {getTracks: () => [track]};
  }}},
  crypto: webcrypto, setTimeout, clearTimeout, setInterval, clearInterval, Date, Int16Array, ArrayBuffer, DataView, console,
});
function command(message) {
  return new Promise((resolve, reject) => {
    const accepted = handler({target: "offscreen", ...message}, {id: "extension", url: "chrome-extension://extension/dist/worker.js"}, (response) => response.ok ? resolve(response.data) : reject(new Error(response.error)));
    assert.equal(accepted, true);
  });
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
const clock = {videoId: "video", mediaId: "media", time: 10, paused: false, wall_time: Date.now(), reason: "start", rate: 1, serial: 1};
const start = (language) => command({type: "start", streamId: "opaque", token: "private-test-token", selection: {tabId: 7, frameId: 2, videoId: "video"}, clock: {...clock, wall_time: Date.now()}, title: "Test", language});
let replied = false;
assert.equal(handler({target: "offscreen", type: "start"}, {id: "extension", tab: {id: 7}}, () => { replied = true; }), undefined);
assert.equal(replied, false, "content scripts cannot start privileged audio capture");

await start('es');
assert.equal(sockets.length, 1);
assert.equal(sockets[0].sent.find(item => item?.type === 'hello').language, 'es', 'selected language reaches the companion handshake');
assert.equal((await command({type: 'state'})).language, 'es', 'reopened popup can identify the active language');
const oldProcessor = processor;
processor.audio(0, 128000);
assert.equal(sockets[0].sent.filter((item) => item?.type === "audio").length, 1);
await command({type: "clock", clock: {...clock, time: 40, reason: "seeking", serial: 2, wall_time: Date.now()}});
assert(sockets[0].sent.some((item) => item?.type === "reset" && item.epoch === 1));
const cue = {id: 1, start: 40, end: 41, text: "English"};
sockets[0].receive({type: "cue", epoch: 0, cue});
assert.equal(events.filter((event) => event.type === "live-cue").length, 0);
sockets[0].receive({type: "cue", epoch: 1, cue});
assert.equal(events.filter((event) => event.type === "live-cue").length, 1);
processor.audio(1, 1600);
const finishing = await command({type: "stop"});
assert.equal(finishing.stopping, true);
await tick();
assert.equal(tracks[0].stopped, true, "stop releases captured tab audio immediately");
assert.equal(sockets[0].readyState, 1, "stop keeps the socket alive for accepted inference results");
const finalAudio = sockets[0].sent.findLast((item) => item?.type === "audio");
assert.equal(finalAudio.epoch, 1);
assert(sockets[0].sent.some((item) => item instanceof ArrayBuffer && item.byteLength === 3200), "stop flushes final PCM tail");
assert.equal(sockets[0].sent.at(-1).type, "stop");
sockets[0].receive({type: "stopped", job_id: "test-job"});
await tick();
assert.equal((await command({type: "state"})).active, false);
assert.equal(sockets[0].readyState, 3);

await start();
assert.equal(sockets[1].sent.find(item => item?.type === 'hello').language, 'ja', 'legacy start defaults to Japanese');
oldProcessor.audio(0, 128000);
assert.equal(sockets[1].sent.filter((item) => item?.type === "audio").length, 0, "late audio from a stopped session cannot contaminate a new session");
for (let i = 0; i < 4; i++) processor.audio(0, 128000);
await tick();
const failed = await command({type: "state"});
assert.equal(failed.active, false);
assert.match(failed.detail, /fell behind/);
assert.equal(tracks[1].stopped, true);
assert.equal(sockets[1].sent.filter((item) => item?.type === "audio").length, 3, "audio cannot exceed three outstanding chunks");
await start();
await command({type: "clock", clock: {...clock, paused: true, time: 42, reason: "ended", serial: 2, wall_time: Date.now()}});
await tick();
assert.equal((await command({type: "state"})).stopping, true, "end of video finishes the live job automatically");
sockets[2].receive({type: "stopped", job_id: "test-job"});
await tick();
console.log("Extension lifecycle checks passed: sender isolation, epoch rejection, final-buffer flush, graceful stop and bounded failure.");
