// Browser-level fault injection against a running normal PassengerSite and an exposed public photo.
const assert = require("node:assert/strict");
const {mkdir, writeFile} = require("node:fs/promises");
const path = require("node:path");
const {chromium} = require("playwright");

const base = process.argv[2] || "http://127.0.0.1:8775";
const output = path.resolve(process.argv[3] || "runs/passenger-lost-reply-browser");
const source = {recording: "kartaview_66245", frame: 146, click: [1953.256, 1275.420],
                priorOutlineCentre: [2058.119, 1327.806]};
const norm = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const uuid = /^[a-f0-9-]{36}$/;
const words = "Can you drop me next to the parking spot on the right?";

async function markedRequest(page, words) {
  await page.locator("#instruction").fill(words);
  await page.locator("#submit-area").click();
  await page.waitForFunction(() => document.querySelector("#destination-panel").dataset.state === "mark");
}

async function traceOne(browser, meta, dropReplies) {
  const page = await browser.newPage({viewport: {width: 1392, height: 960}});
  page.setDefaultTimeout(20000);
  const posts = [], applied = [], recovery = [], failures = [], pageErrors = [];
  const snapshotPath = /^\/api\/sessions\/[^/]+$/;
  page.on("pageerror", error => pageErrors.push(String(error)));
  page.on("request", request => {
    const parts = new URL(request.url()).pathname.split("/");
    if (request.method() === "POST" && parts[2] === "sessions" && parts.length === 5) {
      posts.push({name: parts[4], id: request.headers()["x-action-id"], body: request.postDataJSON()});
    }
  });
  page.on("requestfailed", request => failures.push(new URL(request.url()).pathname.split("/").at(-1)));
  if (dropReplies) {
    await page.route(/\/api\/sessions\/[^/]+\/(area|confirm)$/, async route => {
      assert.equal(route.request().method(), "POST");
      const reply = await route.fetch({maxRedirects: 0});
      assert.equal(reply.status(), 200);
      const body = await reply.json();
      const name = new URL(route.request().url()).pathname.split("/").at(-1);
      assert.equal(body.last_action_id, route.request().headers()["x-action-id"]);
      applied.push({name, body});
      // The actual server has committed the action. Only its reply to Edge gets lost.
      await route.abort("connectionreset");
    });
  }
  try {
    await page.goto(new URL("/?scene=" + meta.key, base).href);
    await page.waitForFunction(() => document.querySelector("#frame")?.naturalWidth > 0);
    const imageUrl = new URL(await page.locator("#frame").getAttribute("src"), base);
    const sid = imageUrl.pathname.split("/")[3];
    const read = () => page.waitForResponse(r => r.request().method() === "GET"
      && snapshotPath.test(new URL(r.url()).pathname) && new URL(r.url()).pathname.endsWith("/" + sid));
    const write = name => page.waitForResponse(r => r.request().method() === "POST"
      && new URL(r.url()).pathname === "/api/sessions/" + sid + "/" + name);
    async function complete(name, button) {
      const start = performance.now(), waiting = dropReplies ? read() : write(name);
      await page.locator(button).click();
      const body = await (await waiting).json();
      assert.match(body.last_action_id, uuid);
      const post = posts.find(x => x.name === name);
      assert.equal(body.last_action_id, post.id);
      if (dropReplies) {
        assert.deepEqual(body, applied.find(x => x.name === name).body);
      }
      recovery.push({name, action: post.id, revision: body.revision, state: body.view.state,
                     proposal_id: body.view.proposal_id, pin: body.view.result?.destination_point_xy,
                     time_ms: Math.round(performance.now() - start)});
      return body;
    }
    await markedRequest(page, words);
    const canvas = page.locator("#overlay");
    await canvas.scrollIntoViewIfNeeded();
    const displayed = await canvas.boundingBox();
    await canvas.click({position: {x: displayed.width * source.click[0] / meta.width,
                                   y: displayed.height * source.click[1] / meta.height}});
    const offer = await complete("area", "#use-box");
    await page.locator("#confirm").waitFor();
    const pin = offer.view.result.destination_point_xy, bay = offer.view.bay_outline;
    assert.equal(offer.view.state, "proposed");
    assert.equal(offer.view.selected_by, "passenger");
    assert.ok(norm(bay.passenger_tap_xy, source.click) < 6, "Recovered tap must be on the actual indicated part of the original photo.");
    assert.ok(norm(pin, source.priorOutlineCentre) < 4, "Recovered pin must be at the independently inspected bay.");
    assert.deepEqual(pin, bay.center_xy);
    assert.ok(await canvas.evaluate((el, p) => el.getContext("2d")
      .getImageData(Math.round(p[0]), Math.round(p[1]), 1, 1).data[3] > 0, pin));
    const label = dropReplies ? "lost" : "normal";
    await page.screenshot({path: path.join(output, label + "-offer.png")});
    const confirmed = await complete("confirm", "#confirm");
    await page.waitForFunction(() => document.querySelector("#destination-panel").dataset.state === "confirmed");
    assert.equal(confirmed.view.state, "confirmed");
    assert.deepEqual(confirmed.view.result.destination_point_xy, pin);
    assert.deepEqual(confirmed.view.bay_outline.polygon_xy, bay.polygon_xy);
    assert.equal(confirmed.revision, offer.revision + 1);
    assert.equal(await page.locator("#confirm").isHidden(), true);
    await page.screenshot({path: path.join(output, label + "-confirmed.png")});
    // A subsequent, deliberately cancelled request must still work after recovering a confirmation.
    await page.locator("#new-request").click();
    await markedRequest(page, "I want to choose another place.");
    await page.locator("#discard-box").click();
    const cancelling = write("cancel");
    await page.locator("#cancel").click();
    const cancelled = await (await cancelling).json();
    await page.waitForFunction(() => document.querySelector("#destination-title").textContent.includes("cancelled"));
    assert.equal(cancelled.view.state, "cancelled");
    assert.equal(cancelled.last_action_id, posts.find(p => p.name === "cancel").id);
    assert.equal(cancelled.revision, confirmed.revision + 2);
    assert.deepEqual(posts.map(p => p.name), ["submit", "area", "confirm", "submit", "cancel"]);
    assert.equal(new Set(posts.map(p => p.id)).size, posts.length);
    assert.deepEqual(applied.map(r => r.name), dropReplies ? ["area", "confirm"] : []);
    assert.deepEqual(failures.filter(r => ["area", "confirm"].includes(r)), dropReplies ? ["area", "confirm"] : []);
    assert.deepEqual(pageErrors, []);
    return {mode: label, session: sid, original_scene: meta.key, recovered: recovery,
            correctly_rendered_tap: bay.passenger_tap_xy, correctly_rendered_pin: pin,
            saved_polygon: bay.polygon_xy, cancellation_revision: cancelled.revision,
            cancellation_action: cancelled.last_action_id, requests: posts, page_errors: pageErrors};
  } finally {
    await page.close();
  }
}

async function main() {
  await mkdir(output, {recursive: true});
  const catalog = await (await fetch(new URL("/api/scenes", base))).json();
  const meta = catalog.scenes.find(s => s.recording_id === source.recording && s.frame_id === source.frame);
  assert.ok(meta, "This live integration check requires the installed public source photo.");
  const browser = await chromium.launch({channel: process.env.PASSENGER_CHROMIUM ? undefined : "msedge",
    executablePath: process.env.PASSENGER_CHROMIUM || undefined, headless: true});
  try {
    const control = await traceOne(browser, meta, false);
    const fault = await traceOne(browser, meta, true);
    assert.deepEqual(control.saved_polygon, fault.saved_polygon);
    assert.deepEqual(control.correctly_rendered_pin, fault.correctly_rendered_pin);
    assert.deepEqual(control.correctly_rendered_tap, fault.correctly_rendered_tap);
    const result = {checked_utc: new Date().toISOString(), base, browser: browser.version(), source,
                    unretried_mutations: ["area", "confirm"], control, fault};
    await writeFile(path.join(output, "results.json"), JSON.stringify(result, null, 2) + "\n");
    console.log(JSON.stringify({result: "passed", regular: control.session, discarded_replies: fault.session,
                                pin: fault.correctly_rendered_pin, recoveries: fault.recovered}));
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
