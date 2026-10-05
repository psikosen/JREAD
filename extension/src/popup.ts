import {BASE, CaptionStyle, DEFAULT_STYLE, SOURCE_LANGUAGES, Target, VideoInfo, cleanStyle, durationMismatch, validSourceLanguage} from "./core";

const byId = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const input = (id: string) => byId<HTMLInputElement>(id);
const select = (id: string) => byId<HTMLSelectElement>(id);
const button = (id: string) => byId<HTMLButtonElement>(id);
let tabId: number | undefined;
let candidates: (VideoInfo & {frameId: number; url: string})[] = [];
let jobs: any[] = [];
let active = false;
let stopping = false;
let starting = false;
function announce(text: string, error = false) { byId("message").textContent = text; byId("message").classList.toggle("error", error); }
async function command(message: object) {
  const result = await chrome.runtime.sendMessage({target: "worker", ...message});
  if (!result?.ok) throw new Error(result?.error || "The extension did not respond. Reload it and try again.");
  return result.data;
}
async function api(path: string, init: RequestInit = {}, paired = true) {
  const {token} = await chrome.storage.session.get("token");
  if (paired && !token) throw new Error("Open the companion and pair using its code first.");
  const response = await fetch(BASE + path, {...init, headers: {...(init.body instanceof FormData ? {} : {"Content-Type": "application/json"}), ...(paired ? {Authorization: `Bearer ${token}`} : {}), ...init.headers}, signal: AbortSignal.timeout(15000)});
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : `Companion returned ${response.status}.`);
  return value;
}
function option(label: string, value: string) { const item = document.createElement("option"); item.textContent = label; item.value = value; return item; }
function picked() { const item = candidates[Number(select("videos").value)]; if (select("videos").value === "" || !item || tabId === undefined) throw new Error("Choose an accessible video first."); return item; }
function target(): Target { const item = picked(); return {tabId: tabId!, frameId: item.frameId, videoId: item.id}; }
function updateExisting() {
  const video = candidates[Number(select("videos").value)];
  select("existing").replaceChildren(...(video?.tracks || []).map((track) => option(`${track.label} (${track.language || "language unspecified"})`, String(track.index))));
  byId("existing-wrap").hidden = !video?.tracks.length;
  input("matched").checked = false;
}
async function scan() {
  const result = await command({type: "scan", tabId});
  candidates = result.frames.flatMap((frame: any) => frame.videos.map((video: VideoInfo) => ({...video, frameId: frame.frameId, url: frame.url})));
  select("videos").replaceChildren(...(candidates.length ? candidates.map((video, index) => {
    let site = "frame"; try { site = new URL(video.url).hostname; } catch {}
    return option(`${video.title} · ${video.duration ? Math.round(video.duration) + "s" : "unknown length"} · ${site}${video.frameId ? " (frame)" : ""}`, String(index));
  }) : [option("No accessible video found", "")]));
  const blocked = result.frames.filter((frame: any) => frame.blocked);
  const origins = [...new Set<string>(blocked.map((frame: any) => frame.origin).filter(Boolean))];
  select("origins").replaceChildren(...origins.map((origin) => option(origin, origin)));
  byId("frame-permission").hidden = !origins.length;
  byId("capability").textContent = `${candidates.length} video(s) accessible. ${blocked.length ? `${blocked.length} frame(s) blocked or unsupported. ` : ""}${result.truncated ? "Only the first 60 frames were checked. " : ""}${candidates.length ? "Saved tracks and live captions available; live capture needs a ready model." : "Start the site's video, then refresh. Closed shadow players are unsupported."}`;
  updateExisting();
}
async function library() {
  const [status, list] = await Promise.all([api("/api/status"), api("/api/jobs")]);
  byId("connection").textContent = "Paired";
  const translation = status.engine?.translator_label || (status.engine?.translator === "gemma4-12b" ? "Gemma 4 12B" : status.engine?.translator === "gemma4" ? "Gemma 4 E4B" : "Whisper translation");
  const speechModel = status.engine?.model_label || (status.engine?.model === "large-v3-turbo" ? "Large v3 Turbo" : `Whisper ${status.engine?.model || "ASR"}`);
  byId("engine").textContent = `${speechModel} → ${translation} · ${status.engine?.ready ? "Ready" : status.engine?.downloading ? "Downloading…" : "Install models in the companion"}`;
  jobs = list.filter((job: any) => job.cue_count > 0);
  select("jobs").replaceChildren(...(jobs.length ? jobs.map((job) => option(`${job.title} · ${job.cue_count} cues${job.status === "complete" ? "" : ` (${job.status}, partial)`}`, job.id)) : [option("No saved cues — import a track or generate ahead", "")]));
  input("matched").checked = false;
}
function displayStyle(style: CaptionStyle) {
  input("font-size").value = String(style.font_size); input("position").value = String(style.position);
  input("background").checked = style.background; input("outline").checked = style.outline; input("japanese").checked = style.show_japanese;
}
function liveState(state: any) { active = !!state?.active; stopping = !!state?.stopping; if (!starting || active) byId("live-state").textContent = state?.detail || "Capture is off."; button("start").disabled = active || starting; button("stop").disabled = !active || stopping; button("apply").disabled = active || starting; select("source-language").disabled = active || starting; if (active && validSourceLanguage(state.language)) select("source-language").value = state.language; }
function action(id: string, fn: () => Promise<unknown>) {
  button(id).addEventListener("click", async () => {
    if (id === "start") { starting = true; select("source-language").disabled = true; }
    button(id).disabled = true;
    try { await fn(); } catch (error) { const detail = (error as Error).message; announce(detail, true); if (id === "start") byId("live-state").textContent = `Could not start captions: ${detail}`; }
    finally { if (id === "start") starting = false; button(id).disabled = false; if (["start", "stop", "apply"].includes(id)) { button("start").disabled = active || starting; button("stop").disabled = !active || stopping; button("apply").disabled = active || starting; select("source-language").disabled = active || starting; } }
  });
}
action("launch", async () => { await command({type: "native-start"}); announce("The companion is opening. Use its Pair extension button to obtain a code."); });
action("scan", scan);
action("refresh-library", library);
// Permission requests must originate directly from the user's click.
button("grant").addEventListener("click", () => {
  const origin = select("origins").value; if (!origin) return;
  void chrome.permissions.request({origins: [origin]}).then(async (allowed) => { if (allowed) await scan(); else announce("Permission was not granted. That frame remains unavailable."); }).catch((error) => announce(error.message, true));
});
select("videos").addEventListener("change", updateExisting);
select("jobs").addEventListener("change", () => { input("matched").checked = false; });
select("source-language").replaceChildren(...Object.entries(SOURCE_LANGUAGES).map(([code, name]) => option(name, code)));
select("source-language").addEventListener("change", () => {
  const language = select("source-language").value;
  if (validSourceLanguage(language)) void chrome.storage.local.set({sourceLanguage: language}).catch(error => announce(error.message, true));
});
byId<HTMLFormElement>("pair-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const result = await api("/api/pair", {method: "POST", body: JSON.stringify({code: input("pair-code").value.trim()})}, false);
    if (typeof result.token !== "string") throw new Error("The companion returned an invalid pairing response.");
    await chrome.storage.session.set({token: result.token}); input("pair-code").value = "";
    const {style} = await chrome.storage.local.get("style");
    if (!style) {
      const settings = await api("/api/settings");
      const defaults = cleanStyle({font_size: settings.font_size, show_japanese: settings.show_japanese, background: settings.caption_background, position: 100 - (settings.caption_position ?? 10)});
      await chrome.storage.local.set({style: defaults}); displayStyle(defaults);
    }
    await library(); announce("Paired. Choose a player and a subtitle track, or start live captions.");
  } catch (error) { announce((error as Error).message, true); }
});
action("apply", async () => {
  const video = picked(); const job = jobs.find((item) => item.id === select("jobs").value);
  if (!job) throw new Error("Choose a saved track first.");
  if (!input("matched").checked) throw new Error("Confirm that the selected track matches this episode and audio.");
  if (durationMismatch(video.duration, job.duration) && !confirm(`The video is ${Math.round(video.duration!)} seconds; the track is ${Math.round(job.duration)} seconds. Timing may not match. Apply anyway?`)) return;
  const result = await command({type: "apply", selection: target(), jobId: job.id}); announce(`Applied ${result.count} timed cues. Seek, pause and speed changes follow the video automatically.`);
});
action("use-existing", async () => { await command({type: "existing", selection: target(), index: Number(select("existing").value)}); announce("The player's existing track is enabled. Its language is unchanged."); });
input("import").addEventListener("change", async () => {
  const file = input("import").files?.[0]; if (!file) return;
  try {
    if (!/\.(srt|vtt)$/i.test(file.name) || file.size > 5 * 1024 * 1024) throw new Error("Choose an SRT or VTT subtitle file up to 5 MB.");
    const body = new FormData(); body.append("file", file); body.append("title", file.name.replace(/\.(srt|vtt)$/i, ""));
    const job = await api("/api/import", {method: "POST", body}); await library(); select("jobs").value = job.id; announce("Track imported into your companion library. Match it to the video and apply it.");
  } catch (error) { announce((error as Error).message, true); } finally { input("import").value = ""; }
});
action("start", async () => { const video = picked(); const language = select("source-language").value; if (!validSourceLanguage(language)) throw new Error("Choose a supported spoken language."); byId("live-state").textContent = "Starting tab capture…"; announce("Starting tab capture. Waiting for the browser and local companion…"); await chrome.storage.local.set({sourceLanguage: language}); liveState(await command({type: "start", selection: target(), title: video.title, language})); announce(`${SOURCE_LANGUAGES[language]} captions started in English. Play the selected video; captions are delayed and tab audio is mixed.`); });
action("stop", async () => { await command({type: "stop"}); liveState(await command({type: "state"})); announce("Capture stopped. The companion is finishing any remaining audio; watch the capture status for completion."); });
action("clear", async () => { await command({type: "clear"}); liveState(await command({type: "state"})); announce("JREAD captions removed."); });
action("save-style", async () => { const style = cleanStyle({font_size: Number(input("font-size").value), position: Number(input("position").value), background: input("background").checked, outline: input("outline").checked, show_japanese: input("japanese").checked}); displayStyle(await command({type: "style", style})); announce("Caption appearance saved."); });
chrome.storage.onChanged.addListener((changes, area) => { if (area === "session" && changes.live_state) liveState(changes.live_state.newValue); });
void (async () => {
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true}); tabId = tab?.id;
  const {style, sourceLanguage} = await chrome.storage.local.get(["style", "sourceLanguage"]); displayStyle(cleanStyle(style || DEFAULT_STYLE));
  select("source-language").value = validSourceLanguage(sourceLanguage) ? sourceLanguage : "ja";
  const results = await Promise.allSettled([scan(), library(), command({type: "state"}).then(liveState)]);
  for (const result of results) if (result.status === "rejected") announce(result.reason.message, true);
})().catch((error) => announce(error.message, true));
