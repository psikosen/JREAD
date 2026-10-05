"use strict";
(() => {
  // extension/src/core.ts
  var DEFAULT_STYLE = { font_size: 28, position: 90, background: true, outline: true, show_japanese: false };
  function validCue(cue) {
    return !!cue && typeof cue.text === "string" && cue.text.length <= 1e4 && Number.isFinite(cue.start) && Number.isFinite(cue.end) && cue.start >= 0 && cue.end > cue.start;
  }
  function cleanStyle(input = {}) {
    return { font_size: Math.max(14, Math.min(60, Number(input.font_size) || 28)), position: Math.max(10, Math.min(95, Number(input.position) || 90)), background: input.background !== false, outline: input.outline !== false, show_japanese: input.show_japanese === true };
  }
  function cueText(cue, japanese) {
    const plain = cue.text + (japanese && cue.source_text ? "\n" + cue.source_text : "");
    return plain.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function lateCueTimes(cue, currentTime, previousEnd, rate, queued) {
    const start = Math.max(currentTime, previousEnd);
    const end = start + Math.max(2, Math.min(6, cue.end - cue.start)) * rate;
    if (queued >= 12 || end - currentTime > 30 * rate) throw new Error("Delayed captions cannot keep up (over 30 seconds or 12 cues waiting). Capture stopped; replay the saved timed track or use Generate ahead.");
    return { start, end };
  }

  // extension/src/content.ts
  var scope = globalThis;
  if (!scope.__jreadInstalled) {
    let tell2 = function(message) {
      void chrome.runtime.sendMessage({ target: "worker", ...message }).catch(() => {
      });
    }, snapshot2 = function(reason) {
      return { videoId: selectedId, mediaId, time: selected?.currentTime || 0, rate: selected?.playbackRate || 1, paused: !selected || selected.paused || selected.ended || selected.seeking || stalled, wall_time: Date.now(), reason, serial: ++serial };
    }, removeDelayed2 = function() {
      if (liveTrack) {
        for (const cue of delayed) liveTrack.removeCue(cue);
        liveTrack.mode = "disabled";
      }
      delayed = [];
      if (track) track.mode = "showing";
    }, pruneDelayed2 = function() {
      while (delayed.length && selected && delayed[0].endTime <= selected.currentTime) liveTrack?.removeCue(delayed.shift());
      if (!delayed.length) {
        if (liveTrack) liveTrack.mode = "disabled";
        if (track) track.mode = "showing";
      }
    }, announce2 = function(reason) {
      if (!capturing || !selected) return;
      if (!selected.isConnected || (selected.currentSrc || selected.src) !== source || ["emptied", "replacement"].includes(reason)) {
        tell2({ type: "video-gone", videoId: selectedId, detail: "The video changed or was removed. Select the new video to continue." });
        cleanup2();
        return;
      }
      if (selected.playbackRate < 0.25 || selected.playbackRate > 4) {
        tell2({ type: "video-gone", videoId: selectedId, detail: "Live capture supports playback rates from 0.25\xD7 to 4\xD7." });
        capturing = false;
        return;
      }
      tell2({ type: "clock", clock: snapshot2(reason) });
    }, cleanup2 = function() {
      capturing = false;
      abort?.abort();
      observer?.disconnect();
      if (timer) clearInterval(timer);
      removeDelayed2();
      if (track) {
        for (const cue of Array.from(track.cues || [])) track.removeCue(cue);
        track.mode = "disabled";
      }
      for (const [other, mode] of originalModes) other.mode = mode;
      originalModes.clear();
      cues.clear();
      styleElement?.remove();
      if (selected) selected.removeAttribute("data-jread-video");
      selected = void 0;
      track = void 0;
      liveTrack = void 0;
    }, setStyle2 = function(next) {
      style = cleanStyle(next);
      if (!selected) return;
      styleElement ||= document.createElement("style");
      styleElement.textContent = `video[data-jread-video="${selectedId}"]::cue {font-size:${style.font_size}px;color:white;background-color:${style.background ? "rgba(0,0,0,.78)" : "transparent"};text-shadow:${style.outline ? "1px 1px 2px black,-1px -1px 2px black,1px -1px 2px black,-1px 1px 2px black" : "none"};}`;
      (document.head || document.documentElement).append(styleElement);
      for (const { raw, native } of cues.values()) {
        native.text = cueText(raw, style.show_japanese);
        native.line = style.position;
      }
      for (const cue of delayed) cue.line = style.position;
    }, nativeCue2 = function(cue) {
      const item = new VTTCue(cue.start, cue.end, cueText(cue, style.show_japanese));
      item.id = String(cue.id);
      item.snapToLines = false;
      item.line = style.position;
      item.position = 50;
      item.align = "center";
      item.size = 86;
      return item;
    }, putCue2 = function(raw, live) {
      if (!track || !selected || !validCue(raw)) return;
      const old = cues.get(String(raw.id));
      if (old) track.removeCue(old.native);
      const native = nativeCue2(raw);
      track.addCue(native);
      cues.set(String(raw.id), { raw, native });
      if (live && liveTrack && !selected.seeking) {
        pruneDelayed2();
        const queued = delayed.find((cue) => cue.id === String(raw.id));
        if (queued) {
          queued.text = cueText({ ...raw, text: `[Live +${Math.max(0, queued.startTime - raw.end).toFixed(1)}s] ${raw.text}` }, style.show_japanese);
          return;
        }
        if (delayed.length || raw.end < selected.currentTime) {
          try {
            const times = lateCueTimes(raw, selected.currentTime, delayed.at(-1)?.endTime || 0, selected.playbackRate, delayed.length);
            const cue = nativeCue2({ ...raw, ...times, text: `[Live +${Math.max(0, times.start - raw.end).toFixed(1)}s] ${raw.text}` });
            delayed.push(cue);
            liveTrack.addCue(cue);
            track.mode = "hidden";
            liveTrack.mode = "showing";
          } catch (error) {
            capturing = false;
            removeDelayed2();
            tell2({ type: "video-gone", videoId: selectedId, detail: error.message });
          }
        }
      }
    }, bind2 = function(id, live, nextStyle) {
      const video = videos.get(id);
      if (!video?.isConnected) throw new Error("That video is no longer available. Refresh players.");
      cleanup2();
      selected = video;
      selectedId = id;
      mediaId = crypto.randomUUID();
      source = video.currentSrc || video.src;
      epoch = 0;
      stalled = false;
      capturing = live;
      selected.setAttribute("data-jread-video", id);
      track = ownTracks.get(video);
      if (!track) {
        track = video.addTextTrack("subtitles", "JREAD English", "en");
        ownTracks.set(video, track);
      }
      liveTrack = liveTracks.get(video);
      if (live && !liveTrack) {
        liveTrack = video.addTextTrack("subtitles", "JREAD delayed live", "en");
        liveTracks.set(video, liveTrack);
      }
      if (liveTrack) liveTrack.mode = "disabled";
      for (const other of Array.from(video.textTracks)) if (other !== track && other !== liveTrack) {
        originalModes.set(other, other.mode);
        other.mode = "disabled";
      }
      track.mode = "showing";
      setStyle2(nextStyle);
      abort = new AbortController();
      for (const event of ["play", "playing", "pause", "seeking", "seeked", "ratechange", "waiting", "stalled", "ended", "emptied", "loadedmetadata"]) {
        video.addEventListener(event, () => {
          if (["waiting", "stalled"].includes(event)) stalled = true;
          if (["playing", "seeked"].includes(event)) stalled = false;
          if (["seeking", "ratechange"].includes(event)) removeDelayed2();
          if (event === "emptied" || (video.currentSrc || video.src) !== source) {
            tell2({ type: "video-gone", videoId: selectedId, detail: "The media source changed. Refresh players and match captions again." });
            cleanup2();
            return;
          }
          announce2(event);
        }, { signal: abort.signal });
      }
      video.audioTracks?.addEventListener("change", () => {
        tell2({ type: "video-gone", videoId: selectedId, detail: "The audio track changed. Match captions and select the video again." });
        cleanup2();
      }, { signal: abort.signal });
      timer = setInterval(() => {
        pruneDelayed2();
        announce2("tick");
      }, 500);
      observer = new MutationObserver(() => {
        if (!video.isConnected) {
          tell2({ type: "video-gone", videoId: selectedId, detail: "The selected player was removed." });
          cleanup2();
        }
      });
      observer.observe(document.documentElement, { childList: true, subtree: true });
      return snapshot2("start");
    }, scan2 = function() {
      for (const [id, video] of videos) if (!video.isConnected) videos.delete(id);
      return Array.from(document.querySelectorAll("video")).map((video, index) => {
        let id = ids.get(video);
        if (!id) {
          id = crypto.randomUUID();
          ids.set(video, id);
        }
        videos.set(id, video);
        return { id, title: video.getAttribute("aria-label") || video.title || `Video ${index + 1}`, duration: Number.isFinite(video.duration) ? video.duration : null, paused: video.paused, time: video.currentTime, tracks: Array.from(video.textTracks).map((item, index2) => ({ index: index2, label: item.label || item.language || `Track ${index2 + 1}`, language: item.language, cues: item.cues?.length || 0 })).filter((item) => video.textTracks[item.index] !== ownTracks.get(video) && video.textTracks[item.index] !== liveTracks.get(video)) };
      });
    };
    tell = tell2, snapshot = snapshot2, removeDelayed = removeDelayed2, pruneDelayed = pruneDelayed2, announce = announce2, cleanup = cleanup2, setStyle = setStyle2, nativeCue = nativeCue2, putCue = putCue2, bind = bind2, scan = scan2;
    scope.__jreadInstalled = true;
    const videos = /* @__PURE__ */ new Map();
    const ids = /* @__PURE__ */ new WeakMap();
    const ownTracks = /* @__PURE__ */ new WeakMap();
    const liveTracks = /* @__PURE__ */ new WeakMap();
    let selected;
    let selectedId = "";
    let source = "";
    let mediaId = "";
    let track;
    let liveTrack;
    let style = DEFAULT_STYLE;
    let styleElement;
    let capturing = false;
    let stalled = false;
    let serial = 0;
    let epoch = 0;
    let delayed = [];
    let abort;
    let timer;
    let observer;
    const cues = /* @__PURE__ */ new Map();
    const originalModes = /* @__PURE__ */ new Map();
    chrome.runtime.onMessage.addListener((message, sender, respond) => {
      if (sender.id !== chrome.runtime.id || message?.target !== "content") return;
      try {
        let data;
        switch (message.type) {
          case "scan":
            data = scan2();
            break;
          case "bind":
            data = bind2(message.videoId, true, message.style);
            break;
          case "apply":
            if (!Array.isArray(message.cues) || message.cues.length > 5e4 || !message.cues.every(validCue)) throw new Error("Invalid or oversized subtitle track.");
            bind2(message.videoId, false, message.style);
            for (const cue of message.cues) putCue2(cue, false);
            data = { count: cues.size };
            break;
          case "style":
            setStyle2(message.style);
            break;
          case "epoch":
            epoch = message.epoch;
            removeDelayed2();
            break;
          case "cue":
            if (capturing && message.epoch === epoch) putCue2(message.cue, true);
            break;
          case "stopped":
            capturing = false;
            removeDelayed2();
            break;
          case "clear":
            cleanup2();
            break;
          case "existing": {
            const video = videos.get(message.videoId);
            if (!video) throw new Error("Refresh players first.");
            cleanup2();
            const other = video.textTracks[message.index];
            if (!other) throw new Error("Track unavailable.");
            for (const item of Array.from(video.textTracks)) item.mode = item === other ? "showing" : "disabled";
            break;
          }
          default:
            return;
        }
        respond({ ok: true, data });
      } catch (error) {
        respond({ ok: false, error: String(error.message) });
      }
    });
  }
  var tell;
  var snapshot;
  var removeDelayed;
  var pruneDelayed;
  var announce;
  var cleanup;
  var setStyle;
  var nativeCue;
  var putCue;
  var bind;
  var scan;
})();
//# sourceMappingURL=content.js.map
