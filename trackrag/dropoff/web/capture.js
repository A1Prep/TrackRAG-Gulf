// Use the same mono PCM samples as the existing passenger browser capture.
class TrackragPcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.stopped = false;
    this.port.onmessage = event => {
      if (event.data === "stop") {
        this.stopped = true;
        this.port.postMessage({type: "stopped"});
      }
    };
  }

  process(inputs) {
    const samples = inputs[0]?.[0];
    if (!this.stopped && samples?.length) {
      const block = new Float32Array(samples);
      this.port.postMessage(block, [block.buffer]);
    }
    return true;
  }
}
registerProcessor("trackrag-pcm-capture", TrackragPcmCapture);
