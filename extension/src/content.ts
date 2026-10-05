import {CaptionStyle, Clock, Cue, DEFAULT_STYLE, VideoInfo, cleanStyle, cueText, lateCueTimes, validCue} from "./core";

const scope = globalThis as typeof globalThis & {__jreadInstalled?: boolean};
if (!scope.__jreadInstalled) {
  scope.__jreadInstalled = true;
  const videos = new Map<string, HTMLVideoElement>();
  const ids = new WeakMap<HTMLVideoElement, string>();
  const ownTracks = new WeakMap<HTMLVideoElement, TextTrack>();
  const liveTracks = new WeakMap<HTMLVideoElement, TextTrack>();
  let selected: HTMLVideoElement | undefined;
  let selectedId = "";
  let source = "";
  let mediaId = "";
  let track: TextTrack | undefined;
  let liveTrack: TextTrack | undefined;
  let style = DEFAULT_STYLE;
  let styleElement: HTMLStyleElement | undefined;
  let capturing = false;
  let stalled = false;
  let serial = 0;
  let epoch = 0;
  let delayed: VTTCue[] = [];
  let abort: AbortController | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;
  let observer: MutationObserver | undefined;
  const cues = new Map<string, {raw: Cue; native: VTTCue}>();
  const originalModes = new Map<TextTrack, TextTrackMode>();

  function tell(message: object) { void chrome.runtime.sendMessage({target: "worker", ...message}).catch(() => {}); }
  function snapshot(reason: string): Clock {
    return {videoId: selectedId, mediaId, time: selected?.currentTime || 0, rate: selected?.playbackRate || 1, paused: !selected || selected.paused || selected.ended || selected.seeking || stalled, wall_time: Date.now(), reason, serial: ++serial};
  }
  function removeDelayed() {
    if (liveTrack) { for (const cue of delayed) liveTrack.removeCue(cue); liveTrack.mode = "disabled"; }
    delayed = []; if (track) track.mode = "showing";
  }
  function pruneDelayed() {
    while (delayed.length && selected && delayed[0].endTime <= selected.currentTime) liveTrack?.removeCue(delayed.shift()!);
    if (!delayed.length) { if (liveTrack) liveTrack.mode = "disabled"; if (track) track.mode = "showing"; }
  }
  function announce(reason: string) {
    if (!capturing || !selected) return;
    if (!selected.isConnected || (selected.currentSrc || selected.src) !== source || ["emptied", "replacement"].includes(reason)) {
      tell({type: "video-gone", videoId: selectedId, detail: "The video changed or was removed. Select the new video to continue."});
      cleanup(); return;
    }
    if (selected.playbackRate < 0.25 || selected.playbackRate > 4) {
      tell({type: "video-gone", videoId: selectedId, detail: "Live capture supports playback rates from 0.25× to 4×."});
      capturing = false; return;
    }
    tell({type: "clock", clock: snapshot(reason)});
  }
  function cleanup() {
    capturing = false; abort?.abort(); observer?.disconnect(); if (timer) clearInterval(timer);
    removeDelayed();
    if (track) { for (const cue of Array.from(track.cues || [])) track.removeCue(cue); track.mode = "disabled"; }
    for (const [other, mode] of originalModes) other.mode = mode;
    originalModes.clear(); cues.clear(); styleElement?.remove();
    if (selected) selected.removeAttribute("data-jread-video");
    selected = undefined; track = undefined; liveTrack = undefined;
  }
  function setStyle(next: CaptionStyle) {
    style = cleanStyle(next);
    if (!selected) return;
    styleElement ||= document.createElement("style");
    styleElement.textContent = `video[data-jread-video="${selectedId}"]::cue {font-size:${style.font_size}px;color:white;background-color:${style.background ? "rgba(0,0,0,.78)" : "transparent"};text-shadow:${style.outline ? "1px 1px 2px black,-1px -1px 2px black,1px -1px 2px black,-1px 1px 2px black" : "none"};}`;
    (document.head || document.documentElement).append(styleElement);
    for (const {raw, native} of cues.values()) { native.text = cueText(raw, style.show_japanese); native.line = style.position; }
    for (const cue of delayed) cue.line = style.position;
  }
  function nativeCue(cue: Cue): VTTCue {
    const item = new VTTCue(cue.start, cue.end, cueText(cue, style.show_japanese));
    item.id = String(cue.id); item.snapToLines = false; item.line = style.position; item.position = 50; item.align = "center"; item.size = 86;
    return item;
  }
  function putCue(raw: Cue, live: boolean) {
    if (!track || !selected || !validCue(raw)) return;
    const old = cues.get(String(raw.id)); if (old) track.removeCue(old.native);
    const native = nativeCue(raw); track.addCue(native); cues.set(String(raw.id), {raw, native});
    if (live && liveTrack && !selected.seeking) {
      pruneDelayed();
      const queued = delayed.find((cue) => cue.id === String(raw.id));
      if (queued) {
        queued.text = cueText({...raw, text: `[Live +${Math.max(0, queued.startTime - raw.end).toFixed(1)}s] ${raw.text}`}, style.show_japanese);
        return;
      }
      if (delayed.length || raw.end < selected.currentTime) {
        try {
          const times = lateCueTimes(raw, selected.currentTime, delayed.at(-1)?.endTime || 0, selected.playbackRate, delayed.length);
          const cue = nativeCue({...raw, ...times, text: `[Live +${Math.max(0, times.start - raw.end).toFixed(1)}s] ${raw.text}`});
          delayed.push(cue); liveTrack.addCue(cue); track.mode = "hidden"; liveTrack.mode = "showing";
        } catch (error) {
          capturing = false; removeDelayed();
          tell({type: "video-gone", videoId: selectedId, detail: (error as Error).message});
        }
      }
    }
  }
  function bind(id: string, live: boolean, nextStyle: CaptionStyle) {
    const video = videos.get(id);
    if (!video?.isConnected) throw new Error("That video is no longer available. Refresh players.");
    cleanup(); selected = video; selectedId = id; mediaId = crypto.randomUUID(); source = video.currentSrc || video.src;
    epoch = 0; stalled = false; capturing = live; selected.setAttribute("data-jread-video", id);
    track = ownTracks.get(video);
    if (!track) { track = video.addTextTrack("subtitles", "JREAD English", "en"); ownTracks.set(video, track); }
    liveTrack = liveTracks.get(video);
    if (live && !liveTrack) { liveTrack = video.addTextTrack("subtitles", "JREAD delayed live", "en"); liveTracks.set(video, liveTrack); }
    if (liveTrack) liveTrack.mode = "disabled";
    for (const other of Array.from(video.textTracks)) if (other !== track && other !== liveTrack) { originalModes.set(other, other.mode); other.mode = "disabled"; }
    track.mode = "showing"; setStyle(nextStyle);
    abort = new AbortController();
    for (const event of ["play", "playing", "pause", "seeking", "seeked", "ratechange", "waiting", "stalled", "ended", "emptied", "loadedmetadata"]) {
      video.addEventListener(event, () => {
        if (["waiting", "stalled"].includes(event)) stalled = true;
        if (["playing", "seeked"].includes(event)) stalled = false;
        if (["seeking", "ratechange"].includes(event)) removeDelayed();
        if (event === "emptied" || (video.currentSrc || video.src) !== source) {
          tell({type: "video-gone", videoId: selectedId, detail: "The media source changed. Refresh players and match captions again."}); cleanup(); return;
        }
        announce(event);
      }, {signal: abort.signal});
    }
    (video as HTMLVideoElement & {audioTracks?: EventTarget}).audioTracks?.addEventListener("change", () => {
      tell({type: "video-gone", videoId: selectedId, detail: "The audio track changed. Match captions and select the video again."}); cleanup();
    }, {signal: abort.signal});
    timer = setInterval(() => {
      pruneDelayed();
      announce("tick");
    }, 500);
    observer = new MutationObserver(() => {
      if (!video.isConnected) { tell({type: "video-gone", videoId: selectedId, detail: "The selected player was removed."}); cleanup(); }
    });
    observer.observe(document.documentElement, {childList: true, subtree: true});
    return snapshot("start");
  }
  function scan(): VideoInfo[] {
    for (const [id, video] of videos) if (!video.isConnected) videos.delete(id);
    return Array.from(document.querySelectorAll("video")).map((video, index) => {
      let id = ids.get(video); if (!id) { id = crypto.randomUUID(); ids.set(video, id); } videos.set(id, video);
      return {id, title: video.getAttribute("aria-label") || video.title || `Video ${index + 1}`, duration: Number.isFinite(video.duration) ? video.duration : null, paused: video.paused, time: video.currentTime, tracks: Array.from(video.textTracks).map((item, index) => ({index, label: item.label || item.language || `Track ${index + 1}`, language: item.language, cues: item.cues?.length || 0})).filter((item) => video.textTracks[item.index] !== ownTracks.get(video) && video.textTracks[item.index] !== liveTracks.get(video))};
    });
  }
  chrome.runtime.onMessage.addListener((message, sender, respond) => {
    if (sender.id !== chrome.runtime.id || message?.target !== "content") return;
    try {
      let data: unknown;
      switch (message.type) {
        case "scan": data = scan(); break;
        case "bind": data = bind(message.videoId, true, message.style); break;
        case "apply":
          if (!Array.isArray(message.cues) || message.cues.length > 50000 || !message.cues.every(validCue)) throw new Error("Invalid or oversized subtitle track.");
          bind(message.videoId, false, message.style); for (const cue of message.cues) putCue(cue, false); data = {count: cues.size}; break;
        case "style": setStyle(message.style); break;
        case "epoch": epoch = message.epoch; removeDelayed(); break;
        case "cue": if (capturing && message.epoch === epoch) putCue(message.cue, true); break;
        case "stopped": capturing = false; removeDelayed(); break;
        case "clear": cleanup(); break;
        case "existing": {
          const video = videos.get(message.videoId); if (!video) throw new Error("Refresh players first.");
          cleanup(); const other = video.textTracks[message.index]; if (!other) throw new Error("Track unavailable.");
          for (const item of Array.from(video.textTracks)) item.mode = item === other ? "showing" : "disabled";
          break;
        }
        default: return;
      }
      respond({ok: true, data});
    } catch (error) { respond({ok: false, error: String((error as Error).message)}); }
  });
}
