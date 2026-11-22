// Exercise only a visibly marked offline software fixture with a simulated microphone.
// node tools/passenger_voice_form_browser.cjs FIXTURE.html FAKE-MIC.wav SECOND.wav NEW-PRIVATE-OUTPUT
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");

async function main() {
  const args = process.argv.slice(2);
  assert.equal(args.length, 4);
  const [form, mic, sample, out] = args.map(x => path.resolve(x));
  const source = fs.readFileSync(form, "utf8");
  const plan = JSON.parse(source.match(/<script type="application\/json" id="formData">([^<]+)<\/script>/)[1]);
  assert.equal(plan.software_fixture, true);
  fs.mkdirSync(out);
  const external = [], errors = [];
  const browser = await chromium.launch({headless: true, executablePath: process.env.PASSENGER_CHROMIUM || undefined,
    args: ["--disable-gpu", "--disable-background-networking", "--use-fake-device-for-media-stream",
           "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required",
           "--use-file-for-fake-audio-capture=" + mic]});
  try {
    const ctx = await browser.newContext({acceptDownloads: true, permissions: ["microphone"]});
    ctx.on("request", req => { if (/^(https?|wss?):/i.test(req.url())) external.push(req.method()); });
    async function open() {
      const p = await ctx.newPage();
      p.setDefaultTimeout(12000);
      p.on("pageerror", e => errors.push(e.message));
      p.on("websocket", socket => external.push(socket.url().split(":")[0]));
      p.on("dialog", async dialog => { await dialog.accept(); });
      await p.goto(pathToFileURL(form).href);
      assert(await p.locator("#fixture").isVisible());
      await p.locator("#speaker").selectOption("S01");
      await p.locator("#begin").click();
      assert(await p.locator("#workspace").isHidden());
      await p.locator("#consent").check();
      await p.locator("#begin").click();
      return p;
    }
    async function playAndSave(p, exact) {
      await p.locator("#review").waitFor({state:"visible"});
      assert(await p.locator("#attest").isDisabled() && await p.locator("#keep").isDisabled());
      await p.locator("#player").evaluate(el => el.play());
      await p.waitForFunction(() => !document.querySelector("#attest").disabled);
      if (exact) await p.locator("#same").check();
      else await p.locator("#reference").fill("software fixture verified wording");
      await p.locator("#attest").check();
      await p.locator("#keep").click();
      await p.locator("#review").waitFor({state:"hidden"});
    }
    async function download(p, filename) {
      const waiting = p.waitForEvent("download");
      await p.locator("#export").click();
      const d = await waiting;
      assert(d.suggestedFilename().startsWith("software-fixture-"));
      const output = path.join(out, filename);
      await d.saveAs(output);
      return output;
    }
    const page = await open();
    await page.locator("#record").click();
    await page.waitForFunction(() => !document.querySelector("#stop").disabled ||
      document.querySelector("#status").classList.contains("alert"));
    assert(!(await page.locator("#stop").isDisabled()), await page.locator("#status").innerText());
    await page.waitForTimeout(1850);
    await page.locator("#stop").click();
    await playAndSave(page, true);
    const partial = await download(page, "private-partial.json");
    await page.close();
    const resumed = await open();
    await resumed.locator("#returnFile").setInputFiles(partial);
    await resumed.waitForFunction(() => document.querySelector("#count").textContent.startsWith("1 of"));
    await resumed.locator("#audioFile").setInputFiles(sample);
    await playAndSave(resumed, false);
    const complete = await download(resumed, "private-two.json");
    const a = JSON.parse(fs.readFileSync(partial)), b = JSON.parse(fs.readFileSync(complete));
    assert(a.software_fixture && b.software_fixture && a.entries.length === 1 && b.entries.length === 2);
    assert.deepEqual(a.entries[0],b.entries[0]);
    assert.equal(b.entries[0].original_acquisition, "browser_microphone");
    assert.equal(b.entries[1].original_acquisition, "participant_selected_original_wav");
    assert.notEqual(b.entries[1].audio_sha256, b.entries[0].audio_sha256);
    assert.deepEqual(external, []); assert.deepEqual(errors, []);
    console.log(JSON.stringify({kind:"marked_fixture_offline_browser_exercise", browser:browser.version(),
      tested_form_id:plan.form_id, external_page_requests:external.length, page_errors:errors.length,
      restored_unchanged_takes:1, verified_software_takes:b.entries.length,
      microphone_and_original_wav:true, original_words_verified_before_asr:true,
      formal_human_speakers:0, formal_human_recordings:0}, null, 2));
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e.message); process.exitCode=1; });
