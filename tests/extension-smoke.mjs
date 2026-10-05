import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdtemp, readFile, writeFile, rm, mkdir} from 'node:fs/promises';
import {tmpdir, homedir} from 'node:os';
import {resolve, join} from 'node:path';

// Real unpacked extension + real backend, always in a disposable browser profile.
// Prerequisites: npm run build:extension; running companion with an existing media job.
// Set JREAD_TEST_JOB_ID for a different local job. No token is printed or retained in evidence.
const base = 'http://127.0.0.1:8765';
const token = (await readFile(join(homedir(), '.local/share/JREAD/token'), 'utf8')).trim();
const mediaJob = process.env.JREAD_TEST_JOB_ID || (await api('/api/jobs')).find((job) => job.has_media && job.status === 'complete' && /japanese.*demo/i.test(job.title))?.id;
assert(mediaJob, 'Create a completed Japanese demo media job or set JREAD_TEST_JOB_ID.');
const extensionId = (await readFile('scripts/extension-id.txt', 'utf8')).trim();
const profile = await mkdtemp(join(tmpdir(), 'jread-extension-browser-'));
const fixtureName = `.extension-smoke-${process.pid}.html`;
const fixturePath = resolve('companion', fixtureName);
const media = `/api/jobs/${mediaJob}/media`;
const frame = `<video aria-label="Frame fixture video" id="frame-video" src="${media}" controls muted preload="metadata" width="440"></video>`;
await writeFile(fixturePath, `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>JREAD extension fixture</title><style>body{font:16px system-ui;background:#1e1928;color:#f9f6ff;padding:24px}video{display:block;width:600px;max-width:100%}iframe{width:640px;height:260px;border:1px solid #685878;margin-top:24px}button{padding:12px;margin:12px 0}</style></head><body><h1>JREAD extension playback check</h1><video id="player" aria-label="Main fixture video" src="${media}" controls muted preload="metadata"></video><button id="fullscreen">Fullscreen video</button><iframe id="frame" title="Same-origin player" srcdoc="${frame.replaceAll('&','&amp;').replaceAll('"','&quot;')}"></iframe></body></html>`);
let context;
const deadline = setTimeout(() => { void context?.close(); }, 90000);
let imported;
let liveJob;
let liveResult = 'not attempted; set JREAD_TEST_LIVE=1 for a real tab-capture attempt';
const audioEvidence = {sent: [], received: [], contexts: []};
const checks = [];
const errors = [];
async function api(path, init = {}) {
  const response = await fetch(base + path, {...init, headers: {Authorization: `Bearer ${token}`, ...init.headers}});
  assert(response.ok, `Backend ${path}: HTTP ${response.status}`);
  return response.json();
}
try {
  const job = await api(`/api/jobs/${mediaJob}`);
  assert(job.has_media && job.duration > 9, 'The fixture needs at least 9 seconds of playable local media.');
  if (process.env.JREAD_TEST_LIVE === '1') assert(job.duration <= 20, 'Live regression capture is limited to a fixture of at most20 seconds.');
  context = await chromium.launchPersistentContext(profile, {executablePath: '/usr/bin/brave-browser', channel: 'chromium', headless: true, ignoreDefaultArgs: ['--mute-audio'], viewport: {width: 1100, height: 900}, reducedMotion: 'reduce', args: [`--disable-extensions-except=${resolve('extension')}`, `--load-extension=${resolve('extension')}`, '--autoplay-policy=no-user-gesture-required', ...(process.env.JREAD_TEST_LIVE === '1' ? ['--enable-unsafe-extension-debugging'] : [])]});
  let worker = context.serviceWorkers().find((item) => item.url().includes(extensionId));
  worker ||= await context.waitForEvent('serviceworker', {predicate: (item) => item.url().includes(extensionId), timeout: 15000});
  assert.equal(new URL(worker.url()).hostname, extensionId);
  checks.push('compiled MV3 worker loaded with stable extension ID');
  const companion = await context.newPage();
  companion.on('pageerror', (error) => errors.push(error.message));
  await companion.goto(`${base}/#token=${token}`);
  await companion.locator('#engine-badge').filter({hasText: 'Ready'}).waitFor();
  assert.equal(new URL(companion.url()).hash, '');
  const player = await context.newPage();
  player.on('pageerror', (error) => errors.push(error.message));
  await player.goto(`${base}/${fixtureName}`);
  await player.waitForFunction(() => document.querySelector('#player').readyState >= 1);
  await player.evaluate(() => {
    const video = document.querySelector('#player');
    const original = video.addTextTrack('subtitles', 'Original fixture track', 'en');
    original.addCue(new VTTCue(0, 9, 'Original track'));
    original.mode = 'showing';
    document.querySelector('#fullscreen').addEventListener('click', () => video.requestFullscreen());
  });
  const popup = await context.newPage();
  popup.on('pageerror', (error) => errors.push(error.message));
  // Simulate the toolbar popup's target: the video tab remains active during initialization.
  await player.bringToFront();
  await popup.goto(`chrome-extension://${extensionId}/popup.html`);
  const tabs = await popup.evaluate(() => chrome.tabs.query({}));
  const tabId = tabs.find((tab) => tab.url?.endsWith(fixtureName))?.id;
  assert(Number.isInteger(tabId));
  await popup.evaluate((tabId) => chrome.tabs.update(tabId, {active: true}), tabId);
  await popup.reload();
  const code = (await api('/api/pair/new', {method: 'POST'})).code;
  await popup.locator('#pair-code').fill(code);
  await popup.locator('#pair-form button').click();
  await popup.waitForFunction(() => document.querySelector('#connection').textContent === 'Paired' && document.querySelector('#pair-code').value === '');
  checks.push('real companion pairing through popup and token kept out of DOM');
  const send = (message) => popup.evaluate((message) => chrome.runtime.sendMessage({target: 'worker', ...message}), message);
  const scanned = await send({type: 'scan', tabId});
  assert(scanned.ok, scanned.error);
  const top = scanned.data.frames.find((item) => item.frameId === 0 && item.videos.length);
  const child = scanned.data.frames.find((item) => item.frameId !== 0 && item.videos.length);
  assert(top, 'Main video discovered');
  assert(child, 'Permitted same-origin iframe video discovered');
  checks.push('main and permitted iframe players discovered via actual injection');
  const selection = {tabId, frameId: 0, videoId: top.videos[0].id};
  const subtitle = '1\n00:00:01,000 --> 00:00:03,000\nLiteral &lt;script&gt;alert(1)&lt;/script&gt; &amp; text\n\n2\n00:00:04,000 --> 00:00:06,000\nSecond cue\n\n3\n00:00:07,000 --> 00:00:09,000\nFinal cue\n';
  const response = popup.waitForResponse((response) => response.url().endsWith('/api/import') && response.request().method() === 'POST');
  await popup.locator('#import').setInputFiles({name: 'extension-regression.srt', mimeType: 'text/plain', buffer: Buffer.from(subtitle)});
  imported = (await (await response).json()).id;
  await popup.waitForFunction((id) => document.querySelector('#jobs').value === id, imported);
  await popup.locator('#scan').click();
  await popup.locator('#videos option').filter({hasText: 'Main fixture video'}).waitFor({state: 'attached'});
  const optionValue = await popup.locator('#videos option').filter({hasText: 'Main fixture video'}).getAttribute('value');
  await popup.locator('#videos').selectOption(optionValue);
  await popup.locator('#jobs').selectOption(imported);
  await popup.locator('#matched').check();
  popup.on('dialog', (dialog) => dialog.accept());
  await popup.locator('#apply').click();
  await popup.locator('#message').filter({hasText: 'Applied 3 timed cues'}).waitFor();
  const trackInfo = await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).map((track) => ({label: track.label, mode: track.mode, cues: track.cues?.length})));
  assert.equal(trackInfo.find((track) => track.label === 'JREAD English')?.cues, 3);
  assert.equal(trackInfo.find((track) => track.label === 'Original fixture track')?.mode, 'disabled');
  checks.push('popup import, media matching and apply create real native cues');
  const activeCue = async (time, expected) => {
    await player.evaluate((time) => { const video = document.querySelector('#player'); video.pause(); video.currentTime = time; }, time);
    await player.waitForFunction((expected) => {
      const track = Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD English');
      return track.activeCues?.[0]?.getCueAsHTML().textContent === expected;
    }, expected);
  };
  await activeCue(1.5, 'Literal <script>alert(1)</script> & text');
  await activeCue(4.5, 'Second cue');
  await player.evaluate(() => { document.querySelector('#player').playbackRate = 1.5; });
  await activeCue(7.5, 'Final cue');
  await activeCue(1.5, 'Literal <script>alert(1)</script> & text');
  checks.push('pause, forward/backward seek and playback rate keep source-timed captions aligned; markup stays literal');
  await player.locator('#fullscreen').click();
  await player.waitForFunction(() => document.fullscreenElement?.id === 'player');
  assert.equal(await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD English').mode), 'showing');
  await mkdir('test-results', {recursive: true});
  await player.screenshot({path: 'test-results/extension-fullscreen.png'});
  await player.evaluate(() => document.exitFullscreen());
  checks.push('native caption track remains enabled in video fullscreen');
  const styleResult = await send({type: 'style', style: {font_size: 36, position: 20, background: false, outline: true, show_japanese: false}});
  assert(styleResult.ok, styleResult.error);
  assert.equal(await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD English').cues[0].line), 20);
  const isolated = await worker.evaluate(async (tabId) => {
    const result = await chrome.scripting.executeScript({target: {tabId, frameIds: [0]}, func: async () => {
      const reply = await chrome.runtime.sendMessage({target: 'worker', type: 'native-start'});
      let readable = false;
      try { const value = await chrome.storage.session.get('token'); readable = !!value.token; } catch {}
      return {reply, readable};
    }});
    return result[0].result;
  }, tabId);
  assert.equal(isolated.reply.ok, false);
  assert.equal(isolated.readable, false);
  checks.push('content scripts cannot launch native commands or read session credentials');
  await player.evaluate(() => { const video = document.querySelector('#player'); video.playbackRate = 1; video.currentTime = 3.5; });
  await player.waitForFunction(() => !document.querySelector('#player').seeking);
  const contentMessage = (message) => popup.evaluate(({tabId, message}) => chrome.tabs.sendMessage(tabId, {target: 'content', ...message}, {frameId: 0}), {tabId, message});
  assert((await contentMessage({type: 'bind', videoId: selection.videoId, style: {font_size: 28, position: 90}})).ok);
  for (const [index, text] of ['First delayed cue', 'Second delayed cue'].entries()) assert((await contentMessage({type: 'cue', epoch: 0, cue: {id: `late-${index}`, start: index, end: index + 1, text}})).ok);
  await player.waitForFunction(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD delayed live')?.activeCues[0]?.text.includes('First delayed cue'));
  assert.equal(await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD delayed live').cues.length), 2);
  assert((await contentMessage({type:'cue',epoch:0,cue:{id:'late-1',start:1,end:2,text:'Corrected second delayed cue'}})).ok);
  assert.equal(await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD delayed live').cues.length), 2);
  assert.equal(await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD English').cues.length), 2);
  await player.evaluate(() => document.querySelector('#player').play());
  await player.waitForFunction(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD delayed live')?.activeCues[0]?.text.includes('Corrected second delayed cue'));
  await player.evaluate(() => { const video = document.querySelector('#player'); video.pause(); video.currentTime = 1.5; });
  await player.waitForFunction(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD delayed live').mode === 'disabled');
  assert.equal(await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'JREAD English').cues.length), 2);
  checks.push('multiple late results play sequentially; same-ID corrections update queued/native captions without duplicates; seeking clears the delayed queue and preserves originals for replay');
  assert((await send({type: 'clear'})).ok);
  assert.equal(await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).find((track) => track.label === 'Original fixture track').mode), 'showing');
  assert((await send({type: 'apply', selection, jobId: imported})).ok);
  assert.equal(await player.evaluate(() => Array.from(document.querySelector('#player').textTracks).filter((track) => track.label === 'JREAD English').length), 1);
  if (process.env.JREAD_TEST_LIVE === '1') {
    await player.evaluate(() => { const video = document.querySelector('#player'); video.currentTime = 0; video.playbackRate = 1; video.muted = false; });
    await player.waitForFunction(() => !document.querySelector('#player').seeking);
    await player.bringToFront();
    // Opt-in debugging applies only to this disposable profile. Trigger the actual
    // toolbar action so Chromium grants normal activeTab capture permission.
    const browserSession = await context.browser().newBrowserCDPSession();
    const {targetInfos} = await browserSession.send('Target.getTargets', {filter: [{type: 'tab'}, {exclude: true}]});
    const tabTarget = targetInfos.find((target) => target.type === 'tab' && target.url === player.url());
    assert(tabTarget, 'The real video tab must be the extension-action target.');
    await browserSession.send('Extensions.triggerAction', {id: extensionId, targetId: tabTarget.targetId});
    await worker.evaluate(async () => {
      const existing = await chrome.runtime.getContexts({contextTypes: ['OFFSCREEN_DOCUMENT']});
      if (!existing.length) await chrome.offscreen.createDocument({url: 'offscreen.html', reasons: ['USER_MEDIA'], justification: 'Isolated capture regression diagnostics.'});
    });
    const offscreenTarget = (await browserSession.send('Target.getTargets')).targetInfos.find((target) => target.url === `chrome-extension://${extensionId}/offscreen.html`);
    assert(offscreenTarget, 'Offscreen capture document exists.');
    const {sessionId} = await browserSession.send('Target.attachToTarget', {targetId: offscreenTarget.targetId, flatten: false});
    browserSession.on('Target.receivedMessageFromTarget', ({message}) => {
      const event = JSON.parse(message);
      if (event.method === 'Network.webSocketFrameSent' && event.params.response.opcode === 2) {
        const pcm = Buffer.from(event.params.response.payloadData, 'base64');
        let square = 0, max = 0;
        for (let i = 0; i + 1 < pcm.length; i += 2) { const sample = pcm.readInt16LE(i) / 32768; square += sample * sample; max = Math.max(max, Math.abs(sample)); }
        audioEvidence.sent.push({bytes: pcm.length, rms: Math.sqrt(square / (pcm.length / 2)), max});
      }
      if (event.method === 'Network.webSocketFrameReceived' && event.params.response.opcode === 1) {
        const data = JSON.parse(event.params.response.payloadData);
        audioEvidence.received.push({type: data.type, ...(data.type === 'error' ? {detail: data.detail} : {})});
      }
      if (event.method === 'WebAudio.contextCreated' || event.method === 'WebAudio.contextChanged') audioEvidence.contexts.push(event.params.context);
    });
    await browserSession.send('Target.sendMessageToTarget', {sessionId, message: JSON.stringify({id: 1, method: 'Network.enable'})});
    await browserSession.send('Target.sendMessageToTarget', {sessionId, message: JSON.stringify({id: 2, method: 'WebAudio.enable'})});
    const liveStarted = performance.now();
    const capture = await send({type: 'start', selection, title: 'Extension real tab-capture regression'});
    if (!capture.ok && /not been invoked|activeTab/.test(capture.error)) liveResult = 'browser rejected automation without toolbar user activation: ' + capture.error;
    else {
      assert(capture.ok, capture.error);
      liveJob = capture.data.job_id;
      assert(liveJob, 'Real live session returned a saved job');
      await player.evaluate(() => document.querySelector('#player').play());
      await player.waitForFunction(() => document.querySelector('#player').ended, null, {timeout: 20000});
      const playbackEnded = performance.now();
      audioEvidence.finalState = await popup.evaluate(() => new Promise((resolve, reject) => {
        const done = (state) => { clearTimeout(timeout); chrome.storage.onChanged.removeListener(changed); resolve(state); };
        const changed = (changes, area) => { if (area === 'session' && changes.live_state?.newValue?.active === false) done(changes.live_state.newValue); };
        const timeout = setTimeout(() => { chrome.storage.onChanged.removeListener(changed); reject(new Error('Live completion did not arrive within 60 seconds.')); }, 60000);
        chrome.storage.onChanged.addListener(changed);
        chrome.runtime.sendMessage({target: 'worker', type: 'state'}).then((reply) => { if (reply.ok && reply.data.active === false) done(reply.data); }, reject);
      }));
      const completed = await api(`/api/jobs/${liveJob}`);
      audioEvidence.deviceConfirmation = {model: completed.model, device: completed.device, translator: completed.translator, status: completed.status, cue_count: completed.cue_count};
      audioEvidence.timing = {fixture_seconds: job.duration, start_to_playback_end_seconds: (playbackEnded - liveStarted) / 1000, finishing_seconds: (performance.now() - playbackEnded) / 1000, total_seconds: (performance.now() - liveStarted) / 1000};
      liveResult = `tab capture started; saved job status=${completed.status}, cue_count=${completed.cue_count}`;
      assert.equal(completed.status, 'complete', completed.error || 'Live job must finish cleanly');
      assert(completed.cue_count > 0, 'Real captured Japanese speech must produce captions');
      const capturedTabs = await worker.evaluate(() => chrome.tabCapture.getCapturedTabs());
      assert(!capturedTabs.some((tab) => tab.status === 'active' || tab.status === 'pending'));
      liveResult = `passed: real tab audio -> offscreen AudioWorklet -> local inference -> ${completed.cue_count} saved cues; automatic end and capture release`;
      checks.push(liveResult);
    }
  }
  await player.evaluate(() => document.querySelector('#player').remove());
  await player.waitForFunction(() => !Array.from(document.querySelectorAll('style')).some((style) => style.textContent.includes('data-jread-video')));
  const frameSelection = {tabId, frameId: child.frameId, videoId: child.videos[0].id};
  const appliedFrame = await send({type: 'apply', selection: frameSelection, jobId: imported});
  assert(appliedFrame.ok, appliedFrame.error);
  const childFrame = player.frames().find((frame) => frame !== player.mainFrame());
  await childFrame.waitForFunction(() => Array.from(document.querySelector('video').textTracks).some((track) => track.label === 'JREAD English' && track.cues.length === 3));
  checks.push('clear restores prior tracks; reapply reuses one track; removed player cleanup; iframe application');
  await popup.screenshot({path: 'test-results/extension-popup.png', fullPage: true});
  assert.deepEqual(errors, []);
  checks.push('no page JavaScript errors');
  const result = {passed: true, browser: await context.browser().version(), checks, live_tab_capture: liveResult, audioEvidence};
  await writeFile('test-results/extension-smoke.json', JSON.stringify(result, null, 2));
  await rm('test-results/extension-smoke-failure.json', {force: true});
  console.log(`PASS: ${checks.join('; ')}`);
} catch (error) {
  await mkdir('test-results', {recursive: true});
  await writeFile('test-results/extension-smoke-failure.json', JSON.stringify({passed: false, checks, live_tab_capture: liveResult, audioEvidence, error: error.message}, null, 2));
  throw error;
} finally {
  clearTimeout(deadline);
  if (imported) await api(`/api/jobs/${imported}`, {method: 'DELETE'}).catch(() => {});
  if (liveJob) await api(`/api/jobs/${liveJob}`, {method: 'DELETE'}).catch(() => {});
  await context?.close();
  await rm(fixturePath, {force: true});
  await rm(profile, {recursive: true, force: true});
}
