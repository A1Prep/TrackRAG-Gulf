// Normal local ASR plus passenger grounding. Use generated speech fixtures, with the second WAV at 48 kHz.
const assert = require("node:assert/strict");
const {createHash} = require("node:crypto");
const {mkdir, readFile, writeFile} = require("node:fs/promises");
const path = require("node:path");
const {chromium} = require("playwright");

const [base, upload, simulatedMic, out] = process.argv.slice(2);
assert.ok(base && upload && simulatedMic && out, "Pass local PassengerSite URL, synthetic upload WAV, synthetic device WAV and output folder.");
const corrected = "Can you drop me next to the parking spot on the right?";
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const knownPin = [2058.119, 1327.806];

async function exercise(browser, scene, mode, uploadSha) {
  const context = await browser.newContext({permissions: ["microphone"], viewport: {width: 1392, height: 960}});
  const page = await context.newPage();
  page.setDefaultTimeout(40000);
  const posts = [], errors = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("request", req => {
    const parts = new URL(req.url()).pathname.split("/");
    if (req.method() === "POST" && parts[2] === "sessions" && parts[4]) {
      posts.push({action: parts[4], body: parts[4] === "submit" ? req.postDataJSON() : undefined});
    }
  });
  const changed = name => page.waitForResponse(r => r.request().method() === "POST"
    && new URL(r.url()).pathname.endsWith("/" + name));
  try {
    await page.goto(new URL("/?scene=" + scene.key, base).href);
    await page.waitForFunction(() => document.querySelector("#frame")?.naturalWidth > 0);
    assert.ok(await page.locator("#mic").isVisible());
    const sid = (await page.locator("#frame").getAttribute("src")).split("/")[3];
    const snapshot = async () => (await page.request.get(new URL("/api/sessions/" + sid, base).href)).json();
    const captured = changed("audio"), start = performance.now();
    if (mode === "wav_upload") {
      await page.locator("#file").setInputFiles(upload);
    } else {
      await page.getByRole("button", {name: "Record voice", exact: true}).click();
      await page.getByRole("button", {name: "Stop recording", exact: true}).waitFor();
      await page.waitForFunction(() => Number(document.querySelector("#mic-level").value) > .015);
      // The 6.63 s synthetic file includes leading and trailing silence.
      await page.waitForTimeout(4750);
      await page.getByRole("button", {name: "Stop recording", exact: true}).click();
    }
    assert.equal((await captured).status(), 200);
    await page.waitForFunction(() => document.querySelector("#submit").textContent === "Send reviewed transcript");
    const speech = await snapshot(), reviewAt = performance.now();
    assert.equal(speech.voice_stage, "review");
    assert.equal(speech.view.state, "idle");
    assert.equal(speech.view.proposal_id, null);
    assert.equal(await page.locator("#confirm").isVisible(), false);
    const input = page.locator("#instruction");
    assert.ok(speech.transcript && await input.isEditable());
    assert.equal(await input.inputValue(), speech.transcript);
    assert.match(speech.transcript, /parking|park/i);
    assert.match(speech.transcript, /right/i);
    assert.match(speech.audio_sha256, /^[a-f0-9]{64}$/);
    if (mode === "wav_upload") assert.equal(speech.audio_sha256, uploadSha);
    assert.notEqual(speech.transcript, corrected);
    await input.fill(corrected);
    assert.equal(await page.locator("#confirm").isVisible(), false);
    const sending = changed("submit");
    const attempted = performance.now();
    await page.locator("#submit").click();
    assert.equal((await sending).status(), 200);
    await page.locator("#confirm").waitFor({timeout: 90000});
    const offered = await snapshot(), rendered = performance.now();
    const outline = offered.view.bay_outline, pin = offered.view.result?.destination_point_xy;
    assert.equal(offered.view.state, "proposed");
    assert.equal(offered.view.selected_by, "model");
    assert.deepEqual(pin, outline.center_xy);
    assert.equal(outline.passenger_tap_xy, null);
    assert.ok(Math.hypot(pin[0] - knownPin[0], pin[1] - knownPin[1]) < 4);
    assert.deepEqual(posts.filter(r => r.action === "submit").map(r => r.body),
      [{text: corrected, mode: "voice", background: true}]);
    assert.equal(posts.filter(r => r.action === "confirm").length, 0);
    await page.screenshot({path: path.join(out, mode + "-proposal.png")});
    const saving = changed("confirm");
    await page.locator("#confirm").click();
    assert.equal((await saving).status(), 200);
    await page.waitForFunction(() => document.querySelector("#destination-panel").dataset.state === "confirmed");
    const saved = await snapshot();
    assert.equal(saved.view.state, "confirmed");
    assert.deepEqual(saved.view.result.destination_point_xy, pin);
    assert.deepEqual(saved.view.bay_outline.polygon_xy, outline.polygon_xy);
    assert.equal(saved.view.proposal_id, offered.view.proposal_id);
    assert.equal(posts.filter(r => r.action === "confirm").length, 1);
    assert.deepEqual(errors, []);
    await page.screenshot({path: path.join(out, mode + "-saved.png")});
    return {mode, session: sid, raw_reviewed_transcript: speech.transcript, reviewed_text_sent: corrected,
            verified_wav_sha256: speech.audio_sha256, reviewed_before_proposal: true,
            spent_capture_and_asr_ms: Math.round(reviewAt - start),
            submitted_to_render_ms: Math.round(rendered - attempted), destination: pin,
            native_outline: outline.polygon_xy, explicit_confirmation: saved.view.state, posts, errors};
  } finally {
    await context.close();
  }
}

async function main() {
  await mkdir(out, {recursive: true});
  const catalog = await (await fetch(new URL("/api/scenes", base))).json();
  assert.equal(catalog.speech_available, true);
  const scene = catalog.scenes.find(s => s.recording_id === "kartaview_66245" && s.frame_id === 146);
  assert.ok(scene);
  const browser = await chromium.launch({channel: process.env.PASSENGER_CHROMIUM ? undefined : "msedge",
    executablePath: process.env.PASSENGER_CHROMIUM || undefined, headless: true,
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
           "--autoplay-policy=no-user-gesture-required", "--use-file-for-fake-audio-capture=" + path.resolve(simulatedMic)]});
  try {
    const report = {checked_utc: new Date().toISOString(), browser: browser.version(), base, scene: scene.key,
                    source_wav_sha256: hash(await readFile(upload)), simulated_device_wav_sha256: hash(await readFile(simulatedMic)),
                    runs: []};
    for (const mode of ["wav_upload", "record_button_with_simulated_device"]) {
      const r = await exercise(browser, scene, mode, report.source_wav_sha256);
      report.runs.push(r);
      await writeFile(path.join(out, "result.json"), JSON.stringify(report, null, 2) + "\n");
      console.log(mode, r.raw_reviewed_transcript, r.destination, r.explicit_confirmation, r.submitted_to_render_ms);
    }
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
