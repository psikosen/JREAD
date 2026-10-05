const API = 'http://127.0.0.1:8765';
const ACTIVE = new Set(['uploading', 'queued', 'preparing', 'recognizing', 'translating', 'live']);
const byId = id => document.getElementById(id);
const captionTracks = new WeakMap();
const state = { token: null, status: null, settings: null, jobs: [], file: null, importKind: 'media', selected: null, pairingExpires: 0, refreshBusy: false, confirmAction: null };

// Capture the launcher token before removing it from browser history.
const fragment = new URLSearchParams(location.hash.slice(1));
const launchToken = fragment.get('token');
history.replaceState(null, '', location.pathname + location.search);
if (launchToken) {
  sessionStorage.setItem('jread-token', launchToken);
}
state.token = sessionStorage.getItem('jread-token');

function setConnection(kind, label) {
  byId('connection-dot').className = `dot ${kind || ''}`;
  byId('connection-label').textContent = label;
}
function showError(id, message) {
  const el = byId(id);
  el.textContent = message;
  el.hidden = !message;
}
function toast(message) {
  const el = byId('toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 2800);
}
async function request(path, options = {}) {
  if (!state.token) throw new Error('Open JREAD from its launcher to connect this session.');
  const headers = new Headers(options.headers || {});
  headers.set('Authorization', `Bearer ${state.token}`);
  let body = options.body;
  if (body && !(body instanceof FormData) && typeof body !== 'string') {
    headers.set('Content-Type', 'application/json');
    body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(`${API}${path}`, { ...options, headers, body });
  } catch {
    throw new Error('JREAD’s local service is unavailable. Start it from the desktop launcher and try again.');
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  const type = response.headers.get('content-type') || '';
  return type.includes('application/json') ? response.json() : response.blob();
}
async function establishSession() {
  if (!state.token) return;
  await request('/api/session', { method: 'POST' });
}
const fmtTime = seconds => {
  if (!Number.isFinite(Number(seconds))) return '—';
  const t = Math.max(0, Number(seconds));
  return `${Math.floor(t / 60).toString().padStart(2, '0')}:${Math.floor(t % 60).toString().padStart(2, '0')}.${Math.floor((t % 1) * 10)}`;
};
const niceStatus = value => ({ uploading: 'Uploading media', queued: 'Queued', preparing: 'Preparing audio', recognizing: 'Recognizing Japanese', translating: 'Translating to English', complete: 'Complete', failed: 'Needs attention', cancelled: 'Paused', live: 'Live captions' }[value] || value || 'Unknown');
const deviceMatches = (requested, actual) => requested === 'auto' || actual === requested || (requested === 'cuda' && actual?.startsWith('cuda:'));
const gemmaLabels = { gemma4: 'Gemma 4 E4B', 'gemma4-12b': 'Gemma 4 12B' };
const speechLabels = { small: 'Whisper small', medium: 'Whisper medium', 'large-v3-turbo': 'Large v3 Turbo' };
const activeJobs = () => state.jobs.some(job => ACTIVE.has(job.status) || job.cancellation_pending) || Boolean(state.status?.engine?.downloading);
function renderStatus() {
  const engine = state.status?.engine || {};
  const ready = Boolean(engine.ready);
  const downloading = Boolean(engine.downloading);
  const gemmaLabel = gemmaLabels[state.settings?.translator];
  const useGemma = Boolean(gemmaLabel);
  byId('engine-model').textContent = engine.model ? `${engine.model_label || speechLabels[engine.model] || engine.model}${useGemma ? ` + ${gemmaLabel}` : ' · baseline translation'}` : 'No model selected';
  byId('engine-detail').textContent = engine.error || (ready ? `${engine.device || state.settings?.device || 'Local device'} · ready for Japanese → English` : downloading ? 'Model download in progress.' : 'Choose a model and install it to begin.');
  byId('engine-badge').textContent = engine.error ? 'Attention' : downloading ? 'Installing' : ready ? 'Ready' : 'Not installed';
  byId('engine-badge').className = `badge ${engine.error ? 'failed' : downloading ? 'busy' : ready ? 'ready' : ''}`;
  byId('engine-progress').hidden = !downloading;
  byId('model-select').value = state.settings?.model || 'small';
  byId('device-select').value = state.settings?.device || 'cpu';
  const selectedDevice = byId('device-select').value;
  const selectedDeviceReady = deviceMatches(selectedDevice, engine.device);
  const selectedReady = Boolean(engine.asr_ready ?? ready) && engine.model === byId('model-select').value && selectedDeviceReady;
  byId('model-action').textContent = selectedReady ? 'Speech model ready' : downloading ? 'Downloading…' : 'Install speech model';
  byId('model-action').disabled = downloading || selectedReady;
  byId('remove-model').hidden = !selectedReady;
  byId('translator-select').value = state.settings?.translator || 'whisper';
  byId('model-select').querySelector('[value="large-v3-turbo"]').disabled = !useGemma;
  byId('translator-select').querySelector('[value="whisper"]').disabled = byId('model-select').value === 'large-v3-turbo';
  byId('model-note').textContent = useGemma ? 'Large v3 Turbo requires Gemma for English translation.' : 'Select Gemma translation to enable Large v3 Turbo.';
  byId('glossary-select').value = state.settings?.caption_glossary || 'none';
  byId('caption-context').checked = Boolean(state.settings?.caption_context);
  byId('caption-boundaries').checked = Boolean(state.settings?.caption_boundaries);
  for (const id of ['glossary-select', 'caption-context', 'caption-boundaries']) byId(id).disabled = Boolean(state.status?.live || state.status?.active_job || !useGemma);
  byId('gemma-action').hidden = !useGemma;
  byId('gemma-action').disabled = downloading || Boolean(engine.gemma_ready);
  byId('gemma-action').textContent = engine.gemma_ready ? `${gemmaLabel} ready` : downloading ? 'Downloading…' : `Install ${gemmaLabel}`;
  byId('remove-gemma').hidden = !useGemma || !(engine.gemma_model_ready ?? engine.gemma_ready);
  byId('remove-gemma').textContent = `Remove ${gemmaLabel || 'Gemma'}`;
  const gemmaDownloadGB = state.settings?.translator === 'gemma4-12b' ? '7.0' : '5.2';
  byId('translator-note').textContent = useGemma ? `${gemmaLabel} translates the Japanese transcript locally on CPU. Initial download: about ${gemmaDownloadGB} GB. Translation adds processing time; 12B may increase live delay.` : 'Whisper’s direct translation is the comparison baseline. It can lose meaning; review generated captions.';
  const disk = state.status?.disk_free_bytes;
  byId('disk-note').textContent = Number.isFinite(disk) ? `${(disk / 1024 ** 3).toFixed(1)} GB free on this device` : '';
  if (state.settings) {
    byId('font-size').value = state.settings.font_size || 28;
    byId('font-size-value').textContent = `${state.settings.font_size || 28} px`;
    byId('show-japanese').checked = Boolean(state.settings.show_japanese);
    byId('sample-jp').hidden = !state.settings.show_japanese;
    byId('caption-background').checked = state.settings.caption_background !== false;
    byId('caption-position').value = state.settings.caption_position ?? 10;
    byId('caption-position-value').textContent = `${state.settings.caption_position ?? 10}% from bottom`;
    byId('caption-sample').style.fontSize = `${state.settings.font_size || 28}px`;
    byId('caption-sample').style.setProperty('--caption-bg', state.settings.caption_background === false ? 'transparent' : 'rgba(0,0,0,.76)');
  }
  updateImportButton();
}
function updateImportButton() {
  const button = byId('generate-button');
  const hasFile = Boolean(state.file);
  const engine = state.status?.engine || {};
  const requestedDevice = state.settings?.device;
  const canRun = state.importKind === 'subs' || Boolean(engine.ready && engine.model === state.settings?.model && deviceMatches(requestedDevice, engine.device));
  button.textContent = state.importKind === 'subs' ? 'Add subtitle track' : 'Generate Japanese → English';
  button.disabled = !hasFile || !canRun || !state.token;
  byId('import-note').textContent = !hasFile ? (state.importKind === 'subs' ? 'Choose an SRT or WebVTT file to add it to your library.' : 'Choose a video to create a local subtitle job.') : !canRun ? 'Install a model before generating subtitles.' : state.importKind === 'subs' ? 'The subtitle track will be saved locally and can be exported or edited.' : 'Recognition and translation run on this device. Large videos may take a while.';
}
function renderJobs() {
  const query = byId('search-input').value.trim().toLocaleLowerCase();
  const jobs = state.jobs.filter(job => `${job.title || ''} ${job.status || ''}`.toLocaleLowerCase().includes(query));
  byId('library-count').textContent = `${state.jobs.length} ${state.jobs.length === 1 ? 'episode' : 'episodes'} saved`;
  byId('empty-state').hidden = state.jobs.length > 0;
  const list = byId('job-list');
  list.replaceChildren();
  for (const job of jobs) {
    const card = document.createElement('article'); card.className = 'job-card';
    const title = document.createElement('div'); title.className = 'job-title';
    const name = document.createElement('strong'); name.textContent = job.title || 'Untitled episode';
    const meta = document.createElement('small'); meta.textContent = `${job.cue_count || 0} captions · ${new Date(Number(job.created_at) * 1000).toLocaleString()}`;
    title.append(name, meta);
    const progressWrap = document.createElement('div'); progressWrap.className = 'job-progress';
    const status = document.createElement('span'); status.className = `job-status ${job.status || ''}`; status.textContent = job.cancellation_pending ? 'Cancelling · finishing current step' : niceStatus(job.status);
    progressWrap.append(status);
    if (ACTIVE.has(job.status)) { const progress = document.createElement('progress'); progress.max = 1; progress.value = Number(job.progress) || 0; progress.setAttribute('aria-label', `${niceStatus(job.status)} progress`); progressWrap.append(progress); }
    if (job.error) { const err = document.createElement('span'); err.textContent = job.error; err.className = 'error-inline'; progressWrap.append(err); }
    const actions = document.createElement('div'); actions.className = 'job-actions';
    const open = document.createElement('button'); open.className = 'button secondary'; open.textContent = job.has_media ? 'Open player' : 'Review'; open.addEventListener('click', () => openJob(job.id)); actions.append(open);
    if (job.status === 'live') { const note = document.createElement('span'); note.className = 'live-note'; note.textContent = 'Stop in extension'; note.title = 'Stop live capture from the JREAD browser extension.'; actions.append(note); }
    else if (ACTIVE.has(job.status) && !job.cancellation_pending) { const cancel = document.createElement('button'); cancel.className = 'text-button'; cancel.textContent = 'Cancel'; cancel.addEventListener('click', () => mutateJob(job.id, 'cancel')); actions.append(cancel); }
    if (!job.cancellation_pending && (job.status === 'failed' || job.status === 'cancelled') && job.has_media) { const retry = document.createElement('button'); retry.className = 'text-button'; retry.textContent = 'Retry'; retry.addEventListener('click', () => mutateJob(job.id, 'retry')); actions.append(retry); }
    if (!ACTIVE.has(job.status) && !job.cancellation_pending) { const del = document.createElement('button'); del.className = 'text-button danger-text'; del.textContent = 'Delete'; del.addEventListener('click', () => confirm('Delete episode?', `“${job.title || 'Untitled episode'}” and its saved subtitles will be removed.`, () => deleteJob(job.id))); actions.append(del); }
    card.append(title, progressWrap, actions); list.append(card);
  }
  if (state.jobs.length && !jobs.length) {
    const empty = document.createElement('div'); empty.className = 'empty-state panel'; const msg = document.createElement('p'); msg.textContent = 'No episodes match that search.'; empty.append(msg); list.append(empty);
  }
  if (state.selected) {
    const job = state.jobs.find(item => item.id === state.selected);
    if (job) byId('detail-title').textContent = job.title || 'Untitled episode';
  }
  updateImportButton();
  schedulePolling();
}
async function refresh() {
  if (!state.token || state.refreshBusy || document.hidden) return;
  state.refreshBusy = true;
  try {
    const [status, jobs] = await Promise.all([request('/api/status'), request('/api/jobs')]);
    state.status = status; state.jobs = jobs;
    renderStatus(); renderJobs();
    showError('library-error', '');
    setConnection('ready', 'Local service connected');
    const editingCue = byId('detail').contains(document.activeElement) && document.activeElement.matches('.cue-text, .cue-timing input');
    if (state.selected && !editingCue) await loadSelected(false);
  } catch (error) {
    setConnection('bad', 'Service unavailable');
    showError('library-error', error.message);
  } finally { state.refreshBusy = false; schedulePolling(); }
}
let pollTimer;
function schedulePolling() {
  clearTimeout(pollTimer);
  if (document.hidden || !state.token || !activeJobs()) return;
  pollTimer = setTimeout(refresh, 4000);
}
async function loadSettings() {
  state.settings = await request('/api/settings');
  renderStatus();
}
function setKind(kind) {
  state.importKind = kind;
  byId('media-tab').classList.toggle('active', kind === 'media');
  byId('subs-tab').classList.toggle('active', kind === 'subs');
  byId('media-tab').setAttribute('aria-selected', String(kind === 'media'));
  byId('subs-tab').setAttribute('aria-selected', String(kind === 'subs'));
  byId('file-input').accept = kind === 'media' ? 'video/*,.mkv,.mov,.webm,.mp4,.wav,.mp3,.flac,.ogg,.m4a,.aac' : '.srt,.vtt,text/vtt,application/x-subrip';
  byId('file-input').value = '';
  state.file = null;
  byId('file-label').textContent = kind === 'media' ? 'Choose a video or audio file' : 'Choose an SRT or WebVTT file';
  byId('file-help').textContent = kind === 'media' ? 'Video or WAV, MP3, FLAC, OGG, M4A, AAC · up to 2 GB' : 'SubRip (.srt) or WebVTT (.vtt)';
  updateImportButton();
}
function chooseFile(file) {
  if (!file) return;
  const extension = file.name.split('.').pop().toLowerCase();
  if (state.importKind === 'media' && file.size > 2 * 1024 ** 3) { showError('import-error', 'This media file exceeds the 2 GB upload limit.'); return; }
  const audioExtensions = new Set(['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac']);
  const videoFile = file.type.startsWith('video/') || ['mp4', 'webm', 'mov', 'mkv'].includes(extension);
  const audioFile = audioExtensions.has(extension) && (!file.type || file.type.startsWith('audio/'));
  if (state.importKind === 'media' && !videoFile && !audioFile) { showError('import-error', 'Choose a video or WAV, MP3, FLAC, OGG, M4A, or AAC audio file.'); return; }
  if (state.importKind === 'subs' && !['srt', 'vtt'].includes(extension)) { showError('import-error', 'Choose an .srt or .vtt subtitle file.'); return; }
  state.file = file; showError('import-error', '');
  byId('file-label').textContent = file.name;
  byId('file-help').textContent = `${(file.size / 1024 ** 2).toFixed(1)} MB · ready to add`;
  updateImportButton();
}
async function submitFile() {
  if (!state.file) return;
  const button = byId('generate-button'); button.disabled = true; button.textContent = state.importKind === 'subs' ? 'Adding track…' : 'Preparing upload…';
  showError('import-error', '');
  try {
    const data = new FormData(); data.append('file', state.file);
    const title = byId('title-input').value.trim(); if (title) data.append('title', title);
    if (state.importKind === 'media') data.append('language', 'ja');
    const job = await request(state.importKind === 'subs' ? '/api/import' : '/api/jobs', { method: 'POST', body: data });
    state.file = null; byId('file-input').value = ''; byId('file-label').textContent = state.importKind === 'subs' ? 'Choose an SRT or WebVTT file' : 'Choose a video or audio file';
    byId('title-input').value = '';
    toast(state.importKind === 'subs' ? 'Subtitle track added to your library.' : 'Subtitle job started.');
    await refresh(); await openJob(job.id);
  } catch (error) { showError('import-error', error.message); }
  finally { button.textContent = state.importKind === 'subs' ? 'Add subtitle track' : 'Generate Japanese → English'; updateImportButton(); }
}
async function saveSetting(patch) {
  try {
    await request('/api/settings', { method: 'PATCH', body: patch });
    state.settings = await request('/api/settings');
    if (['model', 'device', 'translator'].some(key => key in patch)) state.status = await request('/api/status');
    renderStatus();
    if (state.selected) await loadSelected();
  }
  catch (error) { toast(error.message); await loadSettings().catch(() => {}); }
}
async function modelAction() {
  showError('engine-error', '');
  try { await request('/api/models/download', { method: 'POST', body: { model: byId('model-select').value } }); toast('Model download started.'); await refresh(); }
  catch (error) { showError('engine-error', error.message); }
}
async function removeModel() {
  const model = state.status?.engine?.model || state.settings?.model;
  if (!model) return;
  confirm('Remove model?', `The downloaded ${speechLabels[model] || model} model will be removed from this device.`, async () => {
    try { await request('/api/models/remove', { method: 'POST', body: { model } }); toast('Model removed.'); await refresh(); }
    catch (error) { showError('engine-error', error.message); }
  });
}
async function mutateJob(id, action) {
  try { await request(`/api/jobs/${encodeURIComponent(id)}/${action}`, { method: 'POST' }); toast(action === 'cancel' ? 'Cancellation requested.' : 'Job resumed.'); await refresh(); }
  catch (error) { showError('library-error', error.message); }
}
async function deleteJob(id) {
  try { await request(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (state.selected === id) closeDetail(); toast('Episode and saved subtitles deleted.'); await refresh();
  } catch (error) { showError('library-error', error.message); }
}
function confirm(title, copy, action) {
  byId('confirm-title').textContent = title; byId('confirm-copy').textContent = copy; state.confirmAction = action; byId('confirm-dialog').showModal();
}
function closeDetail() {
  state.selected = null; byId('detail').hidden = true; const video = byId('video'); video.pause(); video.removeAttribute('src'); video.load();
  delete video.dataset.job;
  byId('cue-list').replaceChildren();
}
async function openJob(id) {
  state.selected = id; byId('detail').hidden = false; byId('detail-title').textContent = 'Loading episode…';
  byId('detail').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  await loadSelected(true);
}
async function loadSelected(showErrorState = true) {
  if (!state.selected) return;
  const selectedId = state.selected;
  try {
    const job = await request(`/api/jobs/${encodeURIComponent(selectedId)}`);
    if (state.selected !== selectedId) return;
    byId('detail-title').textContent = job.title || 'Untitled episode';
    byId('player-meta').textContent = `${niceStatus(job.status)}${job.duration ? ` · ${fmtTime(job.duration)}` : ''}${job.model ? ` · ${speechLabels[job.model] || job.model}` : ''}${gemmaLabels[job.translator] ? ` + ${gemmaLabels[job.translator]}` : ''}`;
    const video = byId('video'); const noVideo = byId('no-video');
    if (job.has_media) {
      noVideo.hidden = true; video.hidden = false;
      const src = `${API}/api/jobs/${encodeURIComponent(job.id)}/media`;
      if (video.dataset.job !== job.id) { video.pause(); video.src = src; video.dataset.job = job.id; }
      setNativeCues(video, job.cues || [], job.id);
    } else { video.pause(); video.hidden = true; noVideo.hidden = false; }
    renderCues(job.cues || [], job);
    showError('detail-error', job.has_media && video.error ? 'This media cannot be played here. You can still edit and export its subtitles for another player.' : '');
  } catch (error) { if (showErrorState) showError('detail-error', error.message); }
}
function setNativeCues(video, cues, jobId) {
  let track = captionTracks.get(video);
  if (!track) { track = video.addTextTrack('captions', 'JREAD English', 'en'); captionTracks.set(video, track); }
  track.mode = 'showing';
  video.style.setProperty('--caption-size', `${state.settings?.font_size || 28}px`);
  video.style.setProperty('--caption-bg', state.settings?.caption_background === false ? 'transparent' : 'rgba(0,0,0,.76)');
  const signature = `${jobId}:${state.settings?.show_japanese}:${state.settings?.caption_position}:${JSON.stringify(cues.map(cue => [cue.id, cue.start, cue.end, cue.text, cue.source_text]))}`;
  if (video.dataset.cueSignature === signature) return;
  video.dataset.cueSignature = signature;
  while (track.cues?.length) track.removeCue(track.cues[0]);
  for (const cue of cues) {
    if (!Number.isFinite(Number(cue.start)) || !Number.isFinite(Number(cue.end)) || Number(cue.end) <= Number(cue.start)) continue;
    const text = state.settings?.show_japanese && cue.source_text ? `${cue.source_text}\n${cue.text || ''}` : (cue.text || '');
    try {
      const nativeCue = new VTTCue(Number(cue.start), Number(cue.end), text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'));
      nativeCue.snapToLines = false;
      nativeCue.line = 100 - Number(state.settings?.caption_position ?? 10);
      track.addCue(nativeCue);
    } catch {}
  }
}
function renderCues(cues, job) {
  const list = byId('cue-list'); list.replaceChildren();
  byId('cue-count').textContent = `${cues.length} ${cues.length === 1 ? 'cue' : 'cues'}`;
  byId('cue-empty').hidden = cues.length > 0 || ACTIVE.has(job.status);
  for (const cue of cues) {
    const row = document.createElement('article'); row.className = 'cue-row';
    const timing = document.createElement('div'); timing.className = 'cue-timing';
    const startLabel = document.createElement('label'); startLabel.textContent = 'Start '; const start = document.createElement('input'); start.type = 'number'; start.min = '0'; start.step = '0.001'; start.value = Number(cue.start).toFixed(3); start.setAttribute('aria-label', 'Caption start time in seconds'); startLabel.append(start);
    const endLabel = document.createElement('label'); endLabel.textContent = 'End '; const end = document.createElement('input'); end.type = 'number'; end.min = '0'; end.step = '0.001'; end.value = Number(cue.end).toFixed(3); end.setAttribute('aria-label', 'Caption end time in seconds'); endLabel.append(end);
    const preview = document.createElement('button'); preview.className = 'text-button'; preview.textContent = `${fmtTime(cue.start)} → ${fmtTime(cue.end)}`; preview.title = 'Play this caption'; preview.addEventListener('click', () => { const v = byId('video'); if (!v.hidden) { v.currentTime = Number(cue.start); v.play().catch(() => {}); } });
    timing.append(preview, startLabel, endLabel);
    const source = document.createElement('div'); source.className = 'cue-source'; source.textContent = cue.source_text || '';
    const text = document.createElement('textarea'); text.className = 'cue-text'; text.value = cue.text || ''; text.setAttribute('aria-label', `English caption ${fmtTime(cue.start)}`); text.maxLength = 4000;
    const actions = document.createElement('div'); actions.className = 'cue-actions';
    const save = document.createElement('button'); save.className = 'text-button'; save.textContent = 'Save edit';
    const undo = document.createElement('button'); undo.className = 'text-button'; undo.textContent = 'Undo'; undo.disabled = cue.status !== 'user_edited';
    save.addEventListener('click', async () => {
      const values = { text: text.value, start: Number(start.value), end: Number(end.value) };
      if (!Number.isFinite(values.start) || !Number.isFinite(values.end) || values.start < 0 || values.end <= values.start) { showError('detail-error', 'Caption timing needs a start before its end.'); return; }
      save.disabled = true;
      try { await request(`/api/jobs/${encodeURIComponent(job.id)}/cues/${encodeURIComponent(cue.id)}`, { method: 'PATCH', body: values }); toast('Caption edit saved.'); await loadSelected(); }
      catch (error) { showError('detail-error', error.message); }
      finally { save.disabled = false; }
    });
    undo.addEventListener('click', async () => {
      undo.disabled = true;
      try { await request(`/api/jobs/${encodeURIComponent(job.id)}/cues/${encodeURIComponent(cue.id)}/undo`, { method: 'POST' }); toast('Last caption edit undone.'); await loadSelected(); }
      catch (error) { showError('detail-error', error.message); undo.disabled = false; }
    });
    actions.append(save, undo);
    if (cue.status === 'needs_review') { const flag = document.createElement('span'); flag.className = 'review-flag'; flag.textContent = 'Review'; actions.append(flag); }
    row.append(timing); if (source.textContent) row.append(source); row.append(text, actions); list.append(row);
  }
}
async function exportJob(format) {
  if (!state.selected) return;
  try {
    const blob = await request(`/api/jobs/${encodeURIComponent(state.selected)}/export?format=${format}`);
    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `${(byId('detail-title').textContent || 'subtitles').replace(/[^\p{L}\p{N}._ -]/gu, '').trim() || 'subtitles'}.${format}`; a.click(); URL.revokeObjectURL(url);
  } catch (error) { showError('detail-error', error.message); }
}
async function createPairing(reuse = false) {
  showError('pair-error', '');
  byId('pair-button').disabled = true;
  try {
    const result = await request(`/api/pair/new${reuse ? '?reuse=true' : ''}`, { method: 'POST' });
    byId('pair-value').textContent = result.code;
    state.pairingExpires = Date.now() + Math.min(300, Number(result.expires_in) || 300) * 1000;
    byId('pair-code').hidden = false; tickPairing();
  } catch (error) { showError('pair-error', error.message); }
  finally { byId('pair-button').disabled = false; }
}
function tickPairing() {
  clearTimeout(tickPairing.timer);
  const remaining = Math.max(0, Math.ceil((state.pairingExpires - Date.now()) / 1000));
  byId('pair-countdown').textContent = remaining ? `Expires in ${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, '0')}` : 'Expired — click Refresh code';
  byId('copy-pair').disabled = !remaining;
  if (remaining) tickPairing.timer = setTimeout(tickPairing, 1000);
  else byId('pair-value').textContent = 'Expired';
}

byId('media-tab').addEventListener('click', () => setKind('media'));
byId('subs-tab').addEventListener('click', () => setKind('subs'));
byId('file-input').addEventListener('change', event => chooseFile(event.target.files[0]));
byId('dropzone').addEventListener('dragover', event => { event.preventDefault(); byId('dropzone').classList.add('drag'); });
byId('dropzone').addEventListener('dragleave', () => byId('dropzone').classList.remove('drag'));
byId('dropzone').addEventListener('drop', event => { event.preventDefault(); byId('dropzone').classList.remove('drag'); chooseFile(event.dataTransfer.files[0]); });
byId('generate-button').addEventListener('click', submitFile);
byId('model-select').addEventListener('change', event => saveSetting({ model: event.target.value }));
byId('device-select').addEventListener('change', event => saveSetting({ device: event.target.value }));
byId('model-action').addEventListener('click', modelAction);
byId('remove-model').addEventListener('click', removeModel);
byId('translator-select').addEventListener('change', event => saveSetting({ translator: event.target.value }));
byId('glossary-select').addEventListener('change', event => saveSetting({ caption_glossary: event.target.value }));
byId('caption-context').addEventListener('change', event => saveSetting({ caption_context: event.target.checked }));
byId('caption-boundaries').addEventListener('change', event => saveSetting({ caption_boundaries: event.target.checked }));
byId('gemma-action').addEventListener('click', async () => {
  try { await request('/api/models/download', { method: 'POST', body: { model: byId('translator-select').value } }); await refresh(); }
  catch (error) { showError('engine-error', error.message); }
});
byId('remove-gemma').addEventListener('click', () => confirm('Remove Gemma 4?', 'The translation model will be removed from this device. Your subtitles and speech model stay saved.', async () => {
  try { await request('/api/models/remove', { method: 'POST', body: { model: byId('translator-select').value } }); await refresh(); }
  catch (error) { showError('engine-error', error.message); }
}));
byId('font-size').addEventListener('input', event => { byId('font-size-value').textContent = `${event.target.value} px`; });
byId('font-size').addEventListener('change', event => saveSetting({ font_size: Number(event.target.value) }));
byId('show-japanese').addEventListener('change', event => { saveSetting({ show_japanese: event.target.checked }); byId('sample-jp').hidden = !event.target.checked; });
byId('caption-background').addEventListener('change', event => saveSetting({ caption_background: event.target.checked }));
byId('caption-position').addEventListener('input', event => { byId('caption-position-value').textContent = `${event.target.value}% from bottom`; });
byId('caption-position').addEventListener('change', event => saveSetting({ caption_position: Number(event.target.value) }));
byId('pair-button').addEventListener('click', () => createPairing());
byId('copy-pair').addEventListener('click', async () => { try { await navigator.clipboard.writeText(byId('pair-value').textContent); toast('Pairing code copied.'); } catch { toast('Select and copy the pairing code.'); } });
byId('search-input').addEventListener('input', renderJobs);
byId('empty-import').addEventListener('click', () => byId('file-input').click());
byId('close-detail').addEventListener('click', closeDetail);
byId('video').addEventListener('error', () => {
  if (state.selected) showError('detail-error', 'This media cannot be played here. You can still edit and export its subtitles for another player.');
});
byId('export-srt').addEventListener('click', () => exportJob('srt'));
byId('export-vtt').addEventListener('click', () => exportJob('vtt'));
byId('confirm-dialog').addEventListener('close', () => { if (byId('confirm-dialog').returnValue === 'confirm' && state.confirmAction) { const action = state.confirmAction; state.confirmAction = null; Promise.resolve(action()); } else state.confirmAction = null; });
byId('delete-all').addEventListener('click', () => {
  const active = state.jobs.some(job => ACTIVE.has(job.status) || job.cancellation_pending);
  confirm('Clear subtitle library?', active ? 'Cancel active jobs first. Live capture must be stopped from the browser extension. Then clear saved episodes and subtitle tracks. Downloaded models stay installed.' : 'Saved episodes, videos, and subtitle tracks will be removed. Downloaded models stay installed.', async () => {
  try {
    if (active) throw new Error('Cancel active jobs first. Live capture must be stopped from the browser extension.');
    const ids = state.jobs.map(job => job.id);
    for (const id of ids) await request(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' });
    closeDetail(); toast('All saved episodes deleted.'); await refresh();
  } catch (error) { showError('library-error', error.message); }
  });
});
document.addEventListener('visibilitychange', () => { if (document.hidden) { clearTimeout(pollTimer); } else refresh(); });

async function start() {
  if (!state.token) { setConnection('bad', 'Open from JREAD launcher'); showError('library-error', 'This page needs the secure session from the JREAD desktop launcher.'); updateImportButton(); return; }
  setConnection('', 'Connecting to local service');
  try {
    await establishSession();
    const [status, settings, jobs] = await Promise.all([request('/api/status'), request('/api/settings'), request('/api/jobs')]);
    state.status = status; state.settings = settings; state.jobs = jobs;
    renderStatus(); renderJobs(); setConnection('ready', 'Local service connected');
    showError('library-error', '');
    await createPairing(true);
  } catch (error) { setConnection('bad', 'Service unavailable'); showError('library-error', error.message); }
  schedulePolling();
}
start();
