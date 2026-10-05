import {BASE, CaptionStyle, Target, cleanStyle, validClock, validCue, validSourceLanguage} from "./core";

const POPUP = chrome.runtime.getURL("popup.html");
const OFFSCREEN = chrome.runtime.getURL("offscreen.html");
let creating: Promise<void> | undefined;
let beginning = false;
chrome.storage.session.setAccessLevel({accessLevel: "TRUSTED_CONTEXTS"}).catch(() => {});

async function offscreen(message: object) {
  const result = await chrome.runtime.sendMessage({target: "offscreen", ...message});
  if (!result?.ok) throw new Error(result?.error || "Audio capture is unavailable. Reload the extension and try again.");
  return result.data;
}
async function hasOffscreen() {
  return (await chrome.runtime.getContexts({contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT], documentUrls: [OFFSCREEN]})).length > 0;
}
async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  creating ||= chrome.offscreen.createDocument({url: "offscreen.html", reasons: [chrome.offscreen.Reason.USER_MEDIA], justification: "Capture tab audio, restore audible playback and send local caption audio."}).finally(() => { creating = undefined; });
  await creating;
}
async function content(target: Target, message: object) {
  const result = await chrome.tabs.sendMessage(target.tabId, {target: "content", ...message}, {frameId: target.frameId});
  if (!result?.ok) throw new Error(result?.error || "Player unavailable. Refresh players.");
  return result.data;
}
function parseTarget(value: Target): Target {
  if (!value || !Number.isInteger(value.tabId) || value.tabId < 0 || !Number.isInteger(value.frameId) || value.frameId < 0 || typeof value.videoId !== "string" || value.videoId.length > 100) throw new Error("Invalid player selection.");
  return {tabId: value.tabId, frameId: value.frameId, videoId: value.videoId};
}
async function local(path: string) {
  const {token} = await chrome.storage.session.get("token");
  if (!token) throw new Error("Pair with the companion first.");
  const response = await fetch(BASE + path, {headers: {Authorization: `Bearer ${token}`}, signal: AbortSignal.timeout(10000)});
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(error.detail || `Companion returned ${response.status}. Pair again if your session expired.`); }
  return response.json();
}
async function stopCapture(detail?: string) {
  if (await hasOffscreen()) await offscreen({type: "stop", detail});
}
async function scan(tabId: number) {
  if (!Number.isInteger(tabId)) throw new Error("Open a video tab first.");
  const tab = await chrome.tabs.get(tabId);
  if (!tab.url || !/^https?:/.test(tab.url)) throw new Error("This page cannot be scripted. Open a normal HTTP(S) page with an HTML5 video.");
  const frames = await chrome.webNavigation.getAllFrames({tabId}) || [];
  const outcomes = await Promise.all(frames.slice(0, 60).map(async (frame) => {
    try {
      await chrome.scripting.executeScript({target: {tabId, frameIds: [frame.frameId]}, files: ["dist/content.js"]});
      const videos = await content({tabId, frameId: frame.frameId, videoId: ""}, {type: "scan"});
      return {frameId: frame.frameId, url: frame.url, videos};
    } catch {
      let origin: string | null = null; try { const url = new URL(frame.url); if (/^https?:$/.test(url.protocol)) origin = url.origin + "/*"; } catch {}
      return {frameId: frame.frameId, url: frame.url, videos: [], blocked: true, origin};
    }
  }));
  return {frames: outcomes, truncated: frames.length > 60};
}
async function popup(message: any) {
  switch (message.type) {
    case "scan": return scan(message.tabId);
    case "native-start": {
      const response = await chrome.runtime.sendNativeMessage("com.jread.companion", {action: "start"});
      if (response?.base_url !== BASE || response?.version !== 1) throw new Error(response?.error || "The native companion did not return the expected local address.");
      return response;
    }
    case "state": {
      if (await hasOffscreen()) return offscreen({type: "state"});
      return {active: false, detail: "Capture is off."};
    }
    case "stop": return stopCapture();
    case "apply": {
      const target = parseTarget(message.selection);
      if (await hasOffscreen() && (await offscreen({type: "state"})).active) throw new Error("Stop live capture before applying a library track.");
      if (typeof message.jobId !== "string" || !/^[A-Za-z0-9_-]{1,100}$/.test(message.jobId)) throw new Error("Invalid library job.");
      const job = await local(`/api/jobs/${message.jobId}`);
      if (!Array.isArray(job.cues) || job.cues.length > 50000 || !job.cues.every(validCue)) throw new Error("This job has no valid subtitle track.");
      const {style} = await chrome.storage.local.get<{style?: CaptionStyle}>("style");
      const result = await content(target, {type: "apply", videoId: target.videoId, cues: job.cues, style: cleanStyle(style)});
      await chrome.storage.session.set({selection: target}); return result;
    }
    case "existing": {
      const target = parseTarget(message.selection); await stopCapture();
      return content(target, {type: "existing", videoId: target.videoId, index: message.index});
    }
    case "clear": {
      await stopCapture(); const {selection} = await chrome.storage.session.get<{selection?: Target}>("selection");
      if (selection) await content(selection, {type: "clear"}); return;
    }
    case "style": {
      const style: CaptionStyle = cleanStyle(message.style); await chrome.storage.local.set({style});
      const {selection} = await chrome.storage.session.get<{selection?: Target}>("selection");
      if (selection) await content(selection, {type: "style", style}).catch(() => {}); return style;
    }
    case "start": {
      if (beginning) throw new Error("Capture is starting."); beginning = true;
      let target: Target | undefined;
      try {
        target = parseTarget(message.selection);
        const language = message.language ?? "ja";
        if (!validSourceLanguage(language)) throw new Error("Choose a supported spoken language.");
        if (await hasOffscreen() && (await offscreen({type: "state"})).active) throw new Error("Stop the current capture first.");
        const {token} = await chrome.storage.session.get("token"); if (!token) throw new Error("Pair with the companion first.");
        const status = await local("/api/status"); if (!status.engine?.ready) throw new Error("Download a model in the companion and wait until it is ready.");
        const {style} = await chrome.storage.local.get<{style?: CaptionStyle}>("style");
        await ensureOffscreen();
        const clock = await content(target, {type: "bind", videoId: target.videoId, style: cleanStyle(style)});
        await chrome.storage.session.set({selection: target});
        const streamId = await chrome.tabCapture.getMediaStreamId({targetTabId: target.tabId});
        return await offscreen({type: "start", streamId, token, selection: target, clock, language, title: String(message.title || "Live captions").slice(0, 200)});
      } catch (error) {
        if (target) await content(target, {type: "stopped"}).catch(() => {}); throw error;
      } finally { beginning = false; }
    }
    default: throw new Error("Unknown extension command.");
  }
}
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message?.target !== "worker" || sender.id !== chrome.runtime.id) return;
  const execute = async () => {
    if (sender.url === POPUP) return popup(message);
    if (sender.url === OFFSCREEN && !sender.tab) {
      const {selection} = await chrome.storage.session.get<{selection?: Target}>("selection");
      switch (message.type) {
        case "live-status":
          await chrome.storage.session.set({live_state: message.state});
          await chrome.action.setBadgeText({text: message.state.active ? "LIVE" : ""});
          await chrome.action.setBadgeBackgroundColor({color: "#713ed6"});
          if (!message.state.active && selection) await content(selection, {type: "stopped"}).catch(() => {});
          return;
        case "live-epoch": if (selection) return content(selection, {type: "epoch", epoch: message.epoch}); return;
        case "live-cue": if (selection && validCue(message.cue)) return content(selection, {type: "cue", epoch: message.epoch, cue: message.cue}); return;
        default: throw new Error("Unknown capture event.");
      }
    }
    if (sender.tab?.id !== undefined && sender.frameId !== undefined) {
      const {selection, live_state} = await chrome.storage.session.get<{selection?: Target; live_state?: {active: boolean}}>(["selection", "live_state"]);
      if (!selection || sender.tab.id !== selection.tabId || sender.frameId !== selection.frameId) throw new Error("This player was not selected.");
      if (message.type === "video-gone" && message.videoId === selection.videoId) return stopCapture(String(message.detail).slice(0, 300));
      if (message.type === "clock" && validClock(message.clock) && message.clock.videoId === selection.videoId) {
        if (live_state?.active) return offscreen({type: "clock", clock: message.clock});
        return;
      }
      throw new Error("Content scripts may only report selected-player timing.");
    }
    throw new Error("Message sender is not authorized.");
  };
  void execute().then((data) => respond({ok: true, data}), (error) => respond({ok: false, error: (error as Error).message}));
  return true;
});
chrome.tabs.onRemoved.addListener((tabId) => { void chrome.storage.session.get<{selection?: Target}>("selection").then(({selection}) => { if (selection?.tabId === tabId) return stopCapture("The captured tab was closed."); }).catch(() => {}); });
chrome.tabs.onUpdated.addListener((tabId, change) => { if (change.status === "loading") void chrome.storage.session.get<{selection?: Target}>("selection").then(({selection}) => { if (selection?.tabId === tabId) return stopCapture("The page navigated. Select its new video to continue."); }).catch(() => {}); });
