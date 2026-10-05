"use strict";
(() => {
  // extension/src/core.ts
  var PCMResampler = class {
    weight = 0;
    sum = 0;
    output = new Int16Array(2048);
    length = 0;
    ratio;
    constructor(rate) {
      if (rate < 16e3) throw new Error("Audio sample rate below 16 kHz is unsupported.");
      this.ratio = rate / 16e3;
    }
    reset() {
      this.weight = this.sum = this.length = 0;
    }
    flush(emit) {
      if (this.length) emit(this.output.slice(0, this.length));
      this.reset();
    }
    push(channels, emit) {
      if (!channels.length) return;
      for (let i = 0; i < channels[0].length; i++) {
        let sample = 0;
        for (const channel of channels) sample += channel[i] || 0;
        sample /= channels.length;
        let left = 1;
        while (left > 1e-9) {
          const take = Math.min(left, this.ratio - this.weight);
          this.sum += sample * take;
          this.weight += take;
          left -= take;
          if (this.weight >= this.ratio - 1e-9) {
            const value = Math.max(-1, Math.min(1, this.sum / this.ratio));
            this.output[this.length++] = Math.round(value * (value < 0 ? 32768 : 32767));
            this.weight = this.sum = 0;
            if (this.length === this.output.length) {
              emit(this.output);
              this.output = new Int16Array(2048);
              this.length = 0;
            }
          }
        }
      }
    }
  };

  // extension/src/worklet.ts
  var PCMProcessor = class extends AudioWorkletProcessor {
    converter = new PCMResampler(sampleRate);
    running = false;
    epoch = 0;
    constructor() {
      super();
      this.port.onmessage = ({ data }) => {
        if (data.flush && data.epoch === this.epoch) {
          this.converter.flush((pcm) => this.port.postMessage({ epoch: this.epoch, pcm: pcm.buffer }, [pcm.buffer]));
          this.port.postMessage({ epoch: this.epoch, flushed: true, final: data.final === true });
        }
        if (data.epoch !== this.epoch || data.running !== this.running) this.converter.reset();
        this.epoch = data.epoch;
        this.running = data.running;
      };
    }
    process(inputs) {
      if (this.running) this.converter.push(inputs[0] || [], (pcm) => this.port.postMessage({ epoch: this.epoch, pcm: pcm.buffer }, [pcm.buffer]));
      return true;
    }
  };
  registerProcessor("jread-pcm", PCMProcessor);
})();
//# sourceMappingURL=worklet.js.map
