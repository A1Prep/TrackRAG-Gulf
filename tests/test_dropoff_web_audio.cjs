// Software checks for the real Task 4 browser encoder and capture lifecycle.
// Run with: node --test tests/test_dropoff_web_audio.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const web = path.join(__dirname, "..", "trackrag", "dropoff", "web");
const source = fs.readFileSync(path.join(web, "app.js"), "utf8");
const tick = () => new Promise(resolve => setImmediate(resolve));

function harness(permission) {
  const nodes = [], contexts = [], uploads = [], requests = [], elements = new Map();
  const track = {stopped: false, stop() { this.stopped = true; }};
  const stream = {getTracks: () => [track]};
  const edge = () => ({connect() {}, disconnect() {}});
  class AudioContext {
    constructor() {
      contexts.push(this);
      this.sampleRate = 16000;
      this.audioWorklet = {addModule: async module => { this.module = module; }};
    }
    createMediaStreamSource() { return edge(); }
    createGain() { return {...edge(), gain: {value: 1}}; }
    async resume() {}
    async close() { this.closed = true; }
  }
  class AudioWorkletNode {
    constructor(context, name, options) {
      nodes.push(this);
      this.name = name; this.options = options;
      this.port = {onmessage: null, postMessage: message => {
        assert.equal(message, "stop");
        queueMicrotask(() => this.port.onmessage?.({data: {type: "stopped"}}));
      }};
    }
    connect() {}
    disconnect() {}
  }
  const context = vm.createContext({
    Blob, Float32Array, URL, setTimeout, clearTimeout,
    setInterval() {},
    document: {getElementById(id) {
      if (!elements.has(id)) elements.set(id, {textContent: "", hidden: false, value: "",
        addEventListener() {}, replaceChildren() {}, classList: {add() {}, remove() {}}});
      return elements.get(id);
    }},
    window: {addEventListener() {}, AudioContext, AudioWorkletNode},
    AudioContext, AudioWorkletNode,
    navigator: {mediaDevices: {getUserMedia: permission || (async () => stream)}},
    fetch: async (url, options) => {
      if (url === "/api/state") return new Promise(() => {});
      requests.push({url, options});
      return {ok: true, json: async () => ({})};
    },
    recordUpload: (...args) => uploads.push(args),
  });
  vm.runInContext(source, context);
  vm.runInContext("sceneToken = 'original-scene'; const originalUploadAudio = uploadAudio; uploadAudio = async (...args) => recordUpload(...args);", context);
  return {context, stream, track, nodes, contexts, uploads, requests, elements,
    run: expression => vm.runInContext(expression, context)};
}

test("encoder writes mono signed PCM WAV with exact original sample rate and clipped amplitudes", async () => {
  const app = harness();
  const blob = app.run("encodeWav([new Float32Array([-2, -1, -0.5]), new Float32Array([0, 0.5, 1, 2])], 48000)");
  assert.equal(blob.type, "audio/wav");
  const bytes = Buffer.from(await blob.arrayBuffer());
  assert.equal(bytes.toString("ascii", 0, 4), "RIFF");
  assert.equal(bytes.readUInt32LE(4), bytes.length - 8);
  assert.equal(bytes.toString("ascii", 8, 16), "WAVEfmt ");
  assert.equal(bytes.readUInt16LE(20), 1);
  assert.equal(bytes.readUInt16LE(22), 1);
  assert.equal(bytes.readUInt32LE(24), 48000);
  assert.equal(bytes.readUInt32LE(28), 96000);
  assert.equal(bytes.readUInt16LE(32), 2);
  assert.equal(bytes.readUInt16LE(34), 16);
  assert.equal(bytes.toString("ascii", 36, 40), "data");
  assert.equal(bytes.readUInt32LE(40), 14);
  assert.deepEqual(Array.from({length: 7}, (_, i) => bytes.readInt16LE(44 + i * 2)),
    [-32768, -32768, -16384, 0, 16384, 32767, 32767]);
});

test("encoder rejects empty, unsupported, nonfinite and over-duration input", () => {
  const app = harness();
  for (const expression of ["encodeWav([], 16000)", "encodeWav([new Float32Array(1)], 8000)",
    "encodeWav([new Float32Array([NaN])], 16000)", "encodeWav([new Float32Array(960001)], 16000)"]) {
    assert.throws(() => app.run(expression));
  }
});

test("capture drains PCM, sends a WAV for its captured scene, and releases microphone resources", async () => {
  const app = harness();
  await app.run("startMicrophone()");
  assert.equal(app.contexts[0].module, "/capture.js");
  assert.equal(app.nodes[0].name, "trackrag-pcm-capture");
  assert.equal(app.nodes[0].options.channelCount, 1);
  app.nodes[0].port.onmessage({data: new Float32Array([0.25, -0.25])});
  await app.run("stopMicrophone()");
  assert.equal(app.uploads.length, 1);
  assert.equal(app.uploads[0][0].type, "audio/wav");
  assert.equal(app.uploads[0][1], "microphone.wav");
  assert.equal(app.uploads[0][2], "original-scene");
  assert.equal(app.track.stopped, true);
  assert.equal(app.contexts[0].closed, true);
  assert.equal(app.run("capture"), null);
});

test("changing scene or replacing input discards a completed recording", async () => {
  for (const change of ["sceneToken = 'new-scene'", "stopMicrophone(true)"]) {
    const app = harness();
    await app.run("startMicrophone()");
    app.nodes[0].port.onmessage({data: new Float32Array([0.25])});
    await app.run(change);
    await app.run("stopMicrophone()");
    assert.equal(app.uploads.length, 0);
    assert.equal(app.track.stopped, true);
    assert.equal(app.contexts[0].closed, true);
  }
});

test("permission arriving after cancellation is closed without creating a recorder or uploading", async () => {
  let grant;
  const app = harness(() => new Promise(resolve => { grant = resolve; }));
  const starting = app.run("startMicrophone()");
  await tick();
  await app.run("stopMicrophone(true); sceneToken = 'new-scene'");
  grant(app.stream);
  await starting;
  assert.equal(app.nodes.length, 0);
  assert.equal(app.track.stopped, true);
  assert.equal(app.uploads.length, 0);
  assert.equal(app.elements.get("record-audio").disabled, false);
});

test("permission denial leaves the prior proposal cancelled and reports the failure", async () => {
  const app = harness(async () => { throw new Error("Permission denied"); });
  await app.run("startMicrophone()");
  assert.equal(app.requests[0].url, "/api/cancel");
  assert.equal(JSON.parse(app.requests[0].options.body).discard_audio, true);
  assert.match(app.elements.get("notice").textContent, /Microphone unavailable: Permission denied/);
  assert.equal(app.uploads.length, 0);
});

test("a scene change or newer typed input during the upload cancellation wait prevents late audio submission", async () => {
  for (const change of ["sceneToken = 'new-scene'", "stopMicrophone(true)"]) {
    const app = harness();
    const posts = [];
    let release;
    app.context.controlPost = (...args) => {
      posts.push(args);
      return new Promise(resolve => { release = resolve; });
    };
    app.run("uploadAudio = originalUploadAudio; post = (...args) => controlPost(...args)");
    const uploading = app.run("uploadAudio(new Blob(['source audio'], {type: 'audio/wav'}), 'source.wav')");
    assert.equal(posts[0][0], "/api/cancel");
    await app.run(change);
    release({});
    await uploading;
    assert.equal(posts.length, 1);
  }
});

test("sample budget caps capture to sixty seconds even when a timer is delayed", async () => {
  const app = harness();
  await app.run("startMicrophone()");
  app.nodes[0].port.onmessage({data: new Float32Array(16000 * 60 + 100)});
  await tick();
  assert.equal(app.uploads.length, 1);
  assert.equal(app.uploads[0][0].size, 44 + 16000 * 60 * 2);
  assert.equal(app.track.stopped, true);
});

test("worklet copies input samples and acknowledges stop after preceding blocks", () => {
  const messages = [];
  let Processor;
  const context = vm.createContext({Float32Array,
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage: value => messages.push(value)}; } },
    registerProcessor(name, constructor) { assert.equal(name, "trackrag-pcm-capture"); Processor = constructor; },
  });
  vm.runInContext(fs.readFileSync(path.join(web, "capture.js"), "utf8"), context);
  const processor = new Processor(), original = new Float32Array([0.5, -0.5]);
  assert.equal(processor.process([[original]]), true);
  original[0] = 0;
  assert.deepEqual(Array.from(messages[0]), [0.5, -0.5]);
  processor.port.onmessage({data: "stop"});
  processor.process([[original]]);
  assert.equal(messages.length, 2);
  assert.equal(messages[1].type, "stopped");
});
