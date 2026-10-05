export const BASE = "http://127.0.0.1:8765";
export const SOURCE_LANGUAGES = {ja: "Japanese", es: "Spanish", en: "English", fr: "French", de: "German", it: "Italian", pt: "Portuguese", ko: "Korean", zh: "Chinese", ru: "Russian", ar: "Arabic", hi: "Hindi"} as const;
export type SourceLanguage = keyof typeof SOURCE_LANGUAGES;
export function validSourceLanguage(value: unknown): value is SourceLanguage { return typeof value === "string" && Object.hasOwn(SOURCE_LANGUAGES, value); }
export interface Cue { id: number | string; start: number; end: number; text: string; source_text?: string }
export interface CaptionStyle { font_size: number; position: number; background: boolean; outline: boolean; show_japanese: boolean }
export const DEFAULT_STYLE: CaptionStyle = {font_size: 28, position: 90, background: true, outline: true, show_japanese: false};
export interface VideoInfo { id: string; title: string; duration: number | null; paused: boolean; time: number; tracks: {index: number; label: string; language: string; cues: number}[] }
export interface Target { tabId: number; frameId: number; videoId: string }
export interface Clock { videoId: string; mediaId: string; time: number; rate: number; paused: boolean; wall_time: number; reason: string; serial: number }
export function validCue(cue: Cue): boolean {
  return !!cue && typeof cue.text === "string" && cue.text.length <= 10000 && Number.isFinite(cue.start) && Number.isFinite(cue.end) && cue.start >= 0 && cue.end > cue.start;
}
export function cleanStyle(input: Partial<CaptionStyle> = {}): CaptionStyle {
  return {font_size: Math.max(14, Math.min(60, Number(input.font_size) || 28)), position: Math.max(10, Math.min(95, Number(input.position) || 90)), background: input.background !== false, outline: input.outline !== false, show_japanese: input.show_japanese === true};
}
export function cueText(cue: Cue, japanese: boolean): string {
  const plain = cue.text + (japanese && cue.source_text ? "\n" + cue.source_text : "");
  return plain.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
export function durationMismatch(duration: number | null, jobDuration: number | null): boolean {
  return !!duration && !!jobDuration && Math.abs(duration - jobDuration) > Math.max(3, duration * 0.02);
}
export function validClock(clock: Clock): boolean {
  return !!clock && typeof clock.videoId === "string" && typeof clock.mediaId === "string" && typeof clock.reason === "string" && typeof clock.paused === "boolean" && Number.isSafeInteger(clock.serial) && Number.isFinite(clock.time) && clock.time >= 0 && Number.isFinite(clock.wall_time) && Number.isFinite(clock.rate) && clock.rate >= 0.25 && clock.rate <= 4;
}
export function mediaNow(clock: Clock, now = Date.now()): number {
  return clock.time + (clock.paused ? 0 : Math.max(0, now - clock.wall_time) / 1000 * clock.rate);
}
export function lateCueTimes(cue: Cue, currentTime: number, previousEnd: number, rate: number, queued: number) {
  const start = Math.max(currentTime, previousEnd);
  const end = start + Math.max(2, Math.min(6, cue.end - cue.start)) * rate;
  if (queued >= 12 || end - currentTime > 30 * rate) throw new Error("Delayed captions cannot keep up (over 30 seconds or 12 cues waiting). Capture stopped; replay the saved timed track or use Generate ahead.");
  return {start, end};
}

// ponytail: box filtering targets voice; use FIR resampling if measured aliasing harms ASR.
export class PCMResampler {
  private weight = 0;
  private sum = 0;
  private output = new Int16Array(2048);
  private length = 0;
  private ratio: number;
  constructor(rate: number) { if (rate < 16000) throw new Error("Audio sample rate below 16 kHz is unsupported."); this.ratio = rate / 16000; }
  reset() { this.weight = this.sum = this.length = 0; }
  flush(emit: (block: Int16Array) => void) { if (this.length) emit(this.output.slice(0, this.length)); this.reset(); }
  push(channels: Float32Array[], emit: (block: Int16Array) => void) {
    if (!channels.length) return;
    for (let i = 0; i < channels[0].length; i++) {
      let sample = 0;
      for (const channel of channels) sample += channel[i] || 0;
      sample /= channels.length;
      let left = 1;
      while (left > 1e-9) {
        const take = Math.min(left, this.ratio - this.weight);
        this.sum += sample * take; this.weight += take; left -= take;
        if (this.weight >= this.ratio - 1e-9) {
          const value = Math.max(-1, Math.min(1, this.sum / this.ratio));
          this.output[this.length++] = Math.round(value * (value < 0 ? 32768 : 32767));
          this.weight = this.sum = 0;
          if (this.length === this.output.length) {
            emit(this.output); this.output = new Int16Array(2048); this.length = 0;
          }
        }
      }
    }
  }
}
export interface Packet { pcm: Int16Array; media_time: number; rate: number }
export class Chunker {
  private samples = new Int16Array(128000);
  private size = 0;
  private fresh = 0;
  private start = 0;
  private rate = 1;
  reset() { this.size = this.fresh = 0; }
  push(input: Int16Array, mediaTime: number, rate: number, emit: (packet: Packet) => void) {
    let offset = 0;
    while (offset < input.length) {
      if (!this.size) { this.start = mediaTime + offset / 16000 * rate; this.rate = rate; }
      const count = Math.min(input.length - offset, this.samples.length - this.size);
      this.samples.set(input.subarray(offset, offset + count), this.size); this.size += count; this.fresh += count; offset += count;
      if (this.size === this.samples.length) {
        emit({pcm: this.samples.slice(), media_time: this.start, rate: this.rate});
        this.samples.copyWithin(0, this.samples.length - 16000); this.size = 16000; this.fresh = 0; this.start += 7 * this.rate;
      }
    }
  }
  flush(emit: (packet: Packet) => void) {
    if (this.fresh) emit({pcm: this.samples.slice(0, this.size), media_time: this.start, rate: this.rate});
    this.reset();
  }
}
export class PendingPackets {
  private pending = new Set<number>();
  epoch = 0;
  private seq = 0;
  send() {
    if (this.pending.size >= 3) throw new Error("Live inference fell behind (3 audio chunks waiting). Capture stopped; use Generate ahead in the companion or a faster model/device.");
    const seq = this.seq++; this.pending.add(seq); return seq;
  }
  ack(seq: number) { this.pending.delete(seq); }
  reset() { this.pending.clear(); return ++this.epoch; }
  accepts(epoch: number) { return epoch === this.epoch; }
}
