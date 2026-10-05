import assert from 'node:assert/strict';
import {build} from 'esbuild';
import vm from 'node:vm';

// Exercise the real popup click handlers; replace only DOM/browser IO.
const built = await build({entryPoints: [new URL('../src/popup.ts', import.meta.url).pathname], bundle: true, write: false, platform: 'browser', format: 'iife'});
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: '0', disabled: false, checked: false, textContent: '', listeners: {},
    classList: {toggle() {}},
    addEventListener(type, callback) { this.listeners[type] = callback; },
    replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ''; },
  });
  return elements.get(id);
}
let storageChanged;
let nextStart;
let lastStart;
const preferences = {sourceLanguage: 'es'};
vm.runInNewContext(built.outputFiles[0].text, {
  document: {getElementById: element, createElement: () => ({value: '', textContent: ''})},
  chrome: {
    runtime: {sendMessage: async message => {
      if (message.type === 'scan') return {ok: true, data: {frames: [{frameId: 0, url: 'https://example.test', videos: [{id: 'video', title: 'Test player', tracks: [], duration: 20}]}]}};
      if (message.type === 'state') return {ok: true, data: {active: false, detail: 'Capture is off.'}};
      if (message.type === 'start') { lastStart = message; return nextStart(); }
      throw Error('Unexpected command: '+message.type);
    }},
    tabs: {query: async () => [{id: 7}]},
    storage: {session: {get: async () => ({token: 'test-token'})}, local: {get: async () => preferences, set: async values => {Object.assign(preferences, values);}},
      onChanged: {addListener: listener => {storageChanged = listener;}}},
  },
  fetch: async url => ({ok: true, json: async () => url.endsWith('/api/status') ? {engine: {ready: true}} : []}),
  AbortSignal, FormData, URL, console,
});
const tick = () => new Promise(resolve => setImmediate(resolve));
await tick();
assert.equal(element('source-language').value, 'es', 'restore the remembered spoken language');
assert(element('source-language').children.some(item => item.value === 'es' && item.textContent === 'Spanish'));
let complete;
nextStart = () => new Promise(resolve => {complete = resolve;});
const pending = element('start').listeners.click();
assert.match(element('live-state').textContent, /Starting tab capture/);
assert.equal(element('start').disabled, true);
assert.equal(element('source-language').disabled, true, 'language cannot change during startup');
await tick();
assert.equal(lastStart.language, 'es', 'Spanish reaches the worker start request');
storageChanged({live_state: {newValue: {active: false, detail: 'Capture is off.'}}}, 'session');
assert.equal(element('start').disabled, true, 'idle state cannot re-enable a pending Start');
assert.match(element('live-state').textContent, /Starting tab capture/, 'idle state cannot erase startup feedback');
complete({ok: true, data: {active: true, detail: 'Capture ready. Play the selected video.'}});
await pending;
assert.equal(element('start').disabled, true);
assert.equal(element('stop').disabled, false);
assert.equal(element('source-language').disabled, true, 'language remains fixed during capture');
assert.match(element('live-state').textContent, /Capture ready/);
storageChanged({live_state: {newValue: {active: false, detail: 'Capture is off.'}}}, 'session');
nextStart = async () => ({ok: false, error: 'Pair with the companion first.'});
await element('start').listeners.click();
assert.match(element('live-state').textContent, /Could not start captions: Pair with the companion first/);
assert.equal(element('start').disabled, false, 'failure permits another start attempt');
assert.equal(element('source-language').disabled, false);
element('source-language').value = 'fr';
element('source-language').listeners.change();
await tick();
assert.equal(preferences.sourceLanguage, 'fr', 'language changes are remembered');
assert.match(element('message').textContent, /Pair with the companion first/);
element('videos').value = '';
await element('start').listeners.click();
assert.match(element('live-state').textContent, /Choose an accessible video first/);
assert.equal(element('start').disabled, false);
console.log('Popup checks passed: startup feedback, idle-state race, failures, language selection/persistence and capture lock.');
