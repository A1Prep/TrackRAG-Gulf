// Keep the browser's mono PCM samples. The page closes the input stream after each turn.
class PassengerCapture extends AudioWorkletProcessor {
  process(inputs) {
    const samples = inputs[0]?.[0];
    if (samples?.length) this.port.postMessage(new Float32Array(samples));
    return true;
  }
}
registerProcessor("passenger-capture", PassengerCapture);
