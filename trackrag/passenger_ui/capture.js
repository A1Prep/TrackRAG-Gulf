// The page and the audio thread load this module. Only the audio thread registers the processor.
if (typeof registerProcessor === "function") {
  class PassengerCapture extends AudioWorkletProcessor {
    constructor() {
      super();
      this.done = false;
      this.port.onmessage = () => { this.done = true; this.port.postMessage("stopped"); };
    }
    process(inputs) {
      const samples = inputs[0]?.[0];
      if (!this.done && samples?.length) this.port.postMessage(samples.slice());
      return !this.done;
    }
  }
  registerProcessor("passenger-capture", PassengerCapture);
}

export const noSound = "No sound reached the app. Check that your microphone is unmuted, or choose a different microphone and record again.";
const browserAccess = "Open this page in Edge or Chrome and allow microphone access for this site.";

function within(promise, milliseconds, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), milliseconds);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function micError(error) {
  if (["NotAllowedError", "SecurityError"].includes(error.name))
    return new Error("Microphone access is blocked. Allow the microphone in the browser’s site settings. If this preview has no permission control, open the same address in Edge or Chrome.");
  if (["NotFoundError", "OverconstrainedError"].includes(error.name))
    return new Error("That microphone is unavailable. Connect a microphone or choose another input and record again.");
  if (error.name === "NotReadableError")
    return new Error("The browser could not open this microphone. Choose another input or close the app using it and try again.");
  return error;
}

export function encodeSpeechWav(chunks, rate) {
  const count = chunks.reduce((total, chunk) => total + chunk.length, 0);
  if (!count) throw new Error(noSound);
  const data = new DataView(new ArrayBuffer(44 + 2 * count));
  function writeText(offset, value) {
    for (let i = 0; i < value.length; i++) data.setUint8(offset + i, value.charCodeAt(i));
  }
  writeText(0, "RIFF"); data.setUint32(4, 36 + 2 * count, true); writeText(8, "WAVE");
  writeText(12, "fmt "); data.setUint32(16, 16, true); data.setUint16(20, 1, true);
  data.setUint16(22, 1, true); data.setUint32(24, rate, true); data.setUint32(28, 2 * rate, true);
  data.setUint16(32, 2, true); data.setUint16(34, 16, true); writeText(36, "data"); data.setUint32(40, 2 * count, true);
  let offset = 44, peak = 0;
  for (const chunk of chunks) for (const sample of chunk) {
    const value = Math.max(-1, Math.min(1, sample));
    const pcm = Math.round(value * (value < 0 ? 32768 : 32767));
    peak = Math.max(peak, Math.abs(pcm));
    data.setInt16(offset, pcm, true); offset += 2;
  }
  return {wav: new Blob([data.buffer], {type: "audio/wav"}), silent: peak === 0};
}

async function workletRecorder(context, source) {
  await within(context.audioWorklet.addModule("/capture.js"), 3000, "The browser audio module did not load.");
  const node = new AudioWorkletNode(context, "passenger-capture", {channelCount: 1, channelCountMode: "explicit"});
  const chunks = [];
  let acknowledge;
  const drained = new Promise(resolve => { acknowledge = resolve; });
  node.port.onmessage = ({data}) => {
    if (data === "stopped") acknowledge();
    else chunks.push(data);
  };
  source.connect(node);
  // The processor only reads its input. Its output is always silence.
  node.connect(context.destination);
  return {
    async wav() {
      node.port.postMessage("stop");
      await within(drained, 1500, "The microphone stopped responding. Record again.");
      return encodeSpeechWav(chunks, context.sampleRate);
    },
    close() { node.port.close(); node.disconnect(); }
  };
}

function browserRecorder(stream) {
  if (typeof MediaRecorder !== "function" || typeof OfflineAudioContext !== "function")
    throw new Error("This browser could not start an audio recorder. " + browserAccess);
  const recorder = new MediaRecorder(stream);
  const parts = [];
  recorder.ondataavailable = ({data}) => { if (data.size) parts.push(data); };
  let recordingError;
  recorder.onerror = () => { recordingError = new Error("The microphone stopped recording. Please try again."); };
  const stopped = new Promise(resolve => { recorder.onstop = resolve; });
  recorder.start();
  function stop() { if (recorder.state !== "inactive") recorder.stop(); }
  return {
    async wav() {
      stop();
      await within(stopped, 2500, "The browser could not finish the recording. Please try again.");
      if (recordingError) throw recordingError;
      if (!parts.length) throw new Error(noSound);
      // Use the browser's own codec, then produce the same PCM WAV accepted by the speech service.
      const decoder = new OfflineAudioContext(1, 1, 16000);
      const bytes = await new Blob(parts, {type: recorder.mimeType}).arrayBuffer();
      const audio = await decoder.decodeAudioData(bytes).catch(() => {
        throw new Error("The browser could not read this recording. Record again in Edge or Chrome.");
      });
      const mono = audio.getChannelData(0).slice();
      for (let channel = 1; channel < audio.numberOfChannels; channel++) {
        const other = audio.getChannelData(channel);
        for (let i = 0; i < mono.length; i++) mono[i] += other[i];
      }
      if (audio.numberOfChannels > 1) for (let i = 0; i < mono.length; i++) mono[i] /= audio.numberOfChannels;
      return encodeSpeechWav([mono], audio.sampleRate);
    },
    close: stop
  };
}

export async function openMicrophone(deviceId, onLevel) {
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioContext !== "function")
    throw new Error("This page cannot access a microphone in this browser. " + browserAccess);
  let stream, source, analyser, recorder, meterTimer, closed = false;
  const context = new AudioContext();
  function close() {
    if (closed) return;
    closed = true;
    clearInterval(meterTimer);
    recorder?.close();
    stream?.getTracks().forEach(track => track.stop());
    source?.disconnect(); analyser?.disconnect();
    if (context.state !== "closed") context.close().catch(() => {});
    window.removeEventListener("pagehide", close);
  }
  window.addEventListener("pagehide", close, {once: true});
  try {
    // Resume while the Record click is still a user gesture, then use the device's native rate.
    const resumed = context.resume();
    resumed.catch(() => {});
    const audio = {channelCount: 1, echoCancellation: true};
    if (deviceId) audio.deviceId = {exact: deviceId};
    const permission = navigator.mediaDevices.getUserMedia({audio}).then(input => {
      if (closed) { input.getTracks().forEach(track => track.stop()); throw new Error("Recording ended."); }
      stream = input;
    });
    await within(permission, 20000, "The browser is waiting for microphone access. " + browserAccess);
    await within(resumed, 4000, "The browser could not start audio. " + browserAccess);
    if (closed) throw new Error("Recording ended.");
    source = context.createMediaStreamSource(stream);
    if (context.audioWorklet && typeof AudioWorkletNode === "function" && context.sampleRate >= 16000 && context.sampleRate <= 48000) {
      try { recorder = await workletRecorder(context, source); }
      catch { /* An embedded browser may block worklet modules while allowing its native recorder. */ }
    }
    if (closed) { recorder?.close(); throw new Error("Recording ended."); }
    if (!recorder) recorder = browserRecorder(stream);
    analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    meterTimer = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      let energy = 0;
      for (const sample of samples) energy += sample * sample;
      const rms = Math.sqrt(energy / samples.length);
      onLevel(Math.max(0, Math.min(1, (20 * Math.log10(rms) + 60) / 60)));
    }, 100);
    return {wav: () => recorder.wav(), close};
  } catch (error) { close(); throw micError(error); }
}
