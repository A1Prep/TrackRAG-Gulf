import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const source = await readFile(new URL("../trackrag/passenger_ui/capture.js", import.meta.url));
const {encodeSpeechWav, openMicrophone} = await import("data:text/javascript;base64," + source.toString("base64"));

const {wav, silent} = encodeSpeechWav([Float32Array.of(-2, 0), Float32Array.of(0.5, 2)], 48000);
const pcm = new DataView(await wav.arrayBuffer());
assert.equal(wav.type, "audio/wav");
assert.equal(wav.size, 52);
assert.equal(pcm.getUint16(22, true), 1);
assert.equal(pcm.getUint32(24, true), 48000);
assert.deepEqual([44, 46, 48, 50].map(offset => pcm.getInt16(offset, true)), [-32768, 0, 16384, 32767]);
assert.equal(silent, false);
assert.equal(encodeSpeechWav([new Float32Array(1600)], 16000).silent, true);
assert.throws(() => encodeSpeechWav([], 16000), /No sound/);

// Stub browser devices and the codec here. The full browser trial uses a file-fed Edge microphone.
function browser({mode, getStream}) {
  let released = 0, nativeRecorders = 0, constraints, workletAttempts = 0;
  const stream = {getTracks: () => [{stop() { released++; }}]};
  const window = new EventTarget();
  const devices = {getUserMedia(audio) { constraints = audio; return getStream ? getStream(stream) : Promise.resolve(stream); }};
  const edge = {connect() {}, disconnect() {}};
  class Context {
    sampleRate = mode === "high-rate" ? 96000 : 48000;
    audioWorklet = {addModule: async () => { workletAttempts++; throw new Error("unavailable"); }};
    resume() { return Promise.resolve(); }
    close() { this.state = "closed"; return Promise.resolve(); }
    createMediaStreamSource() { return edge; }
    createAnalyser() { return {...edge, getFloatTimeDomainData() {}}; }
  }
  class Recorder {
    state = "inactive";
    constructor(input) { assert.equal(input, stream); nativeRecorders++; }
    start() { this.state = "recording"; }
    stop() {
      this.state = "inactive";
      queueMicrotask(() => { this.ondataavailable({data:new Blob(["encoded audio"])}); this.onstop(); });
    }
  }
  class Decoder {
    async decodeAudioData() {
      return {numberOfChannels:2, sampleRate:16000, getChannelData: channel => Float32Array.of(channel === 0 ? 0.25 : 0.75)};
    }
  }
  for (const [name, value] of Object.entries({window, navigator:{mediaDevices:devices}, AudioContext:Context,
    MediaRecorder:Recorder, OfflineAudioContext:Decoder, AudioWorkletNode:mode === "no-worklet" ? undefined : class {}})) {
    Object.defineProperty(globalThis, name, {configurable:true, value});
  }
  return {window, counts: () => ({released, nativeRecorders, workletAttempts}), constraints: () => constraints};
}

for (const mode of ["no-worklet", "blocked-module", "high-rate"]) {
  const device = browser({mode});
  const mic = await openMicrophone("passenger-input", () => {});
  try {
    assert.deepEqual(device.constraints().audio.deviceId, {exact:"passenger-input"});
    const recording = await mic.wav();
    const data = new DataView(await recording.wav.arrayBuffer());
    assert.equal(recording.silent, false);
    assert.equal(data.getUint32(24, true), 16000);
    assert.equal(data.getInt16(44, true), 16384);
  } finally { mic.close(); mic.close(); }
  assert.deepEqual(device.counts(), {released:1, nativeRecorders:1, workletAttempts:mode === "blocked-module" ? 1 : 0});
}

const denied = browser({getStream() { throw new DOMException("denied", "NotAllowedError"); }});
await assert.rejects(openMicrophone("", () => {}), /site settings/);
assert.equal(denied.counts().nativeRecorders, 0);
let grant;
const pending = browser({getStream: stream => new Promise(resolve => { grant = () => resolve(stream); })});
const opening = openMicrophone("", () => {});
pending.window.dispatchEvent(new Event("pagehide"));
grant();
await assert.rejects(opening, /Recording ended/);
assert.equal(pending.counts().released, 1);
