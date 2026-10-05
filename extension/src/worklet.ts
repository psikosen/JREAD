import {PCMResampler} from "./core";
declare const sampleRate: number;
declare class AudioWorkletProcessor { port: MessagePort; }
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void;
class PCMProcessor extends AudioWorkletProcessor {
  private converter = new PCMResampler(sampleRate);
  private running = false;
  private epoch = 0;
  constructor() {
    super();
    this.port.onmessage = ({data}) => {
      if (data.flush && data.epoch === this.epoch) {
        this.converter.flush((pcm) => this.port.postMessage({epoch: this.epoch, pcm: pcm.buffer}, [pcm.buffer]));
        this.port.postMessage({epoch: this.epoch, flushed: true, final: data.final === true});
      }
      if (data.epoch !== this.epoch || data.running !== this.running) this.converter.reset();
      this.epoch = data.epoch; this.running = data.running;
    };
  }
  process(inputs: Float32Array[][]) {
    if (this.running) this.converter.push(inputs[0] || [], (pcm) => this.port.postMessage({epoch: this.epoch, pcm: pcm.buffer}, [pcm.buffer]));
    return true;
  }
}
registerProcessor("jread-pcm", PCMProcessor);
