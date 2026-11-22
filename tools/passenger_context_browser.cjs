// Offline integrity check. Production packets are opened and exported with zero decisions.
// node tools/passenger_context_browser.cjs PRIVATE_PACKET PRIVATE_NEW_CHECK_DIR [--software-fixture]
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");

async function main() {
  const [, , packet, output, mode] = process.argv;
  assert(packet && output && (!mode || mode === "--software-fixture"));
  const fixture = mode === "--software-fixture";
  fs.mkdirSync(output);
  const browser = await chromium.launch({executablePath: process.env.PASSENGER_CHROMIUM || undefined, headless: true,
    args: ["--disable-gpu", "--disable-background-networking"]});
  try {
    const ctx = await browser.newContext({viewport: {width: 1440, height: 980}, deviceScaleFactor: 1});
    const page = await ctx.newPage();
    const errors = [], external = [];
    page.on("pageerror", e => errors.push(e.message));
    page.on("request", req => { if (!req.url().startsWith("file:")) external.push(new URL(req.url()).protocol); });
    await page.goto(pathToFileURL(path.resolve(packet, "review.html")).href);
    const count = await page.locator("#case option").count();
    assert.equal(count, fixture ? 3 : 44);
    assert.equal(await page.locator("#save").isDisabled(), true);
    let tracks = 0;
    for (let i = 0; i < count; i++) {
      await page.selectOption("#case", String(i));
      const measurements = await page.evaluate(async () => {
        const data = JSON.parse(document.querySelector("#packetData").textContent);
        const c = data.scenes[Number(document.querySelector("#case").value)], f = c.request.scene.frame;
        const photo = document.querySelector("#photo");
        await photo.decode();
        const tracks = c.request.scene.tracks;
        const rects = [...document.querySelectorAll('[data-layer="tracks"] rect')];
        const sameBoxes = rects.length === tracks.length && rects.every((r, i) => {
          const [x, y, right, bottom] = tracks[i].bbox_xyxy;
          return ["x", "y", "width", "height"].every((k, j) =>
            Number(r.getAttribute(k)) === [x, y, right - x, bottom - y][j]);
        });
        const dot = document.querySelector('[data-layer="candidate"] circle');
        const point = c.target.destination_point_xy;
        const candidate = point ? !!dot && point.every((v,i)=>Number(dot.getAttribute(i ? "cy" : "cx"))===v) : !dot;
        return {fixtureID: c.request.request_id.startsWith("software-fixture-"), tracks: rects.length,
          ok: sameBoxes && candidate && photo.naturalWidth === f.width && photo.naturalHeight === f.height
            && document.querySelector("#instruction").textContent === c.request.text
            && c.request.dialogue.every(t => document.querySelector("#dialogue").textContent.includes(t.text))
            && document.querySelector("#priorQuestion").textContent === (c.target.clarification_question || "")};
      });
      assert.equal(measurements.fixtureID, fixture);
      assert(measurements.ok);
      tracks += measurements.tracks;
    }
    assert.equal(tracks, fixture ? 3 : 494);
    await page.selectOption("#case", "0");
    await page.locator("#photo").evaluate(i => i.decode());
    await page.locator("#showTracks").uncheck();
    assert.equal(await page.locator('[data-layer="tracks"] rect').count(), 0);
    await page.locator("#showTracks").check();
    await page.locator("#showLabels").check();
    assert.equal(await page.locator('[data-layer="tracks"] text').count(), await page.locator('[data-layer="tracks"] rect').count());
    await page.locator("#showCandidate").uncheck();
    assert.equal(await page.locator('[data-layer="candidate"] *').count(), 0);
    await page.locator("#showCandidate").check();
    await page.screenshot({path: path.join(output, "private-desktop.png")});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    await page.screenshot({path: path.join(output, "private-mobile.png"), fullPage: true});
    await page.selectOption("#role", "1");
    async function download(name) {
      const ready = page.waitForEvent("download");
      await page.click("#download");
      await (await ready).saveAs(path.join(output, name));
    }
    await download("blank-browser-export.csv");
    if (fixture) {
      await page.locator("#reviewer").fill("SOFTWARE-FIXTURE-A");
      await page.selectOption("#decision", "correct");
      await page.selectOption("#status", "needs_clarification");
      await page.locator("#question").fill('Software fixture only: "which marked object", exactly?');
      await page.click("#save");
      assert.match(await page.locator("#progress").textContent(), /^1 \/ 3 saved$/);
      await page.reload();
      await page.selectOption("#role", "1");
      assert.equal(await page.locator("#decision").inputValue(), "correct");
      await download("fixture-corrected.csv");
      await page.selectOption("#role", "2");
      assert.equal(await page.locator("#decision").inputValue(), "");
      assert.equal(await page.locator("#reviewer").inputValue(), "");
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    const receipt = {loaded_original_pictures: count, verified_rendered_current_rectangles: tracks,
      all_instruction_dialogue_and_candidate_fields_matched: true, overlay_toggles_checked: true,
      desktop_and_mobile_checked: true, blank_form_exported: true, external_browser_requests: 0,
      javascript_errors: 0, human_decisions_made: 0, software_fixture: fixture};
    fs.writeFileSync(path.join(output, "browser-readiness.json"), JSON.stringify(receipt, null, 2) + "\n");
    console.log(JSON.stringify(receipt));
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
