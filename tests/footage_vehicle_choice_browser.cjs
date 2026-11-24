// Run against the normal app with both original EMT recordings. No hosted API call is sent.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

async function main() {
  const base = process.env.FOOTAGE_BASE_URL, out = process.env.FOOTAGE_BROWSER_OUT;
  if (!base) throw new Error("Set FOOTAGE_BASE_URL to the running footage application.");
  if (out) fs.mkdirSync(out,{recursive:true});
  const browser = await chromium.launch({channel:"msedge",headless:true});
  const results = [];
  try {
    for (const [recording, frame, noun, width] of [["video_054907",103,"truck",1366],["video_054604",126,"car",390]]) {
      const page = await browser.newPage({viewport:{width,height:850}});
      const errors = [];
      page.on("pageerror", error=>errors.push(error.message));
      const words = "Is the "+noun+" near the building?";
      const url = new URL("/footage",base);
      url.searchParams.set("recording",recording); url.searchParams.set("frame",String(frame));
      await page.goto(url.href);
      await page.waitForFunction(()=>document.querySelectorAll("#examples button:not(:disabled)").length>0);
      await page.locator("#question").fill(words);
      const first = page.waitForResponse(r=>r.url().endsWith("/api/footage") && r.request().method()==="POST");
      const started = Date.now();
      await page.locator("#go").click();
      const response = await first, data = await response.json();
      assert.equal(response.status(),200); assert.equal(data.answer_source,"local_tracks");
      const buttons = page.locator('.vehicle-choices button'), options = data.vehicle_choices.options;
      await buttons.first().waitFor();
      const ms = Date.now()-started;
      assert.equal(await buttons.count(),options.length); assert.ok(options.length>=2);
      await buttons.first().click();
      await page.locator("#vehicle-focus").waitFor();
      assert.equal(await page.locator("#box-label").textContent(),"Track "+options[0].track_id);
      await page.locator("#change-vehicle").click();
      const chosen = options[1];
      await buttons.nth(1).click();
      await page.locator("#vehicle-focus").waitFor();
      assert.equal(await page.locator("#box-label").textContent(),"Track "+chosen.track_id);
      const originalBox = await page.evaluate(()=>{
        const box=document.getElementById("evidence-box"), image=document.getElementById("clip-frame");
        const l=parseFloat(box.style.left),t=parseFloat(box.style.top),w=parseFloat(box.style.width),h=parseFloat(box.style.height);
        return [l*image.naturalWidth,t*image.naturalHeight,(l+w)*image.naturalWidth,(t+h)*image.naturalHeight].map(v=>v/100);
      });
      chosen.bbox_xyxy.forEach((expected,i)=>assert.ok(Math.abs(originalBox[i]-expected)<0.02));
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
      if (out) await page.screenshot({path:path.join(out,recording+"-selection.png")});
      const firstBody=response.request().postDataJSON();
      const selection={track_id:chosen.track_id,binding:data.vehicle_choices.binding};
      const local=await page.request.post(new URL("/api/footage",base).href,{data:{...firstBody,selection}});
      assert.equal(local.status(),200);
      const anchored=await local.json();
      assert.equal(anchored.selected_vehicle.track_id,chosen.track_id);
      assert.equal(anchored.original_question,words);
      assert.equal(anchored.answer.answer_status,"insufficient_evidence");
      assert.match(anchored.answer.answer,/do not measure the physical distance/);
      assert.deepEqual(anchored.evidence.map(e=>e.interval.track_id).sort(),[null,chosen.track_id].sort());
      assert.ok(anchored.evidence.every(e=>e.images.every(im=>im.frame_id===frame)));
      const stale=await page.request.post(new URL("/api/footage",base).href,{data:{...firstBody,selection,question:words.replace("building","shop")}});
      assert.equal(stale.status(),400);
      // Capture the explicit button action without forwarding a paid image request.
      let intent = null;
      await page.route("**/api/footage",async route=>{
        intent=route.request().postDataJSON();
        await route.fulfill({status:503,contentType:"application/json",body:JSON.stringify({error:"Test stopped before contacting Gemini."})});
      });
      await page.locator("#review-selected").click();
      await page.getByText("Test stopped before contacting Gemini.",{exact:true}).waitFor();
      assert.deepEqual(intent,{...firstBody,hosted:true,selection,image_relation:true});
      await page.unroute("**/api/footage");
      await page.locator("#answer-settings summary").click();
      await page.locator("#answer-mode").selectOption("local");
      await page.locator("#go").click();
      await buttons.first().waitFor();
      await buttons.first().click();
      await page.locator("#vehicle-focus").waitFor();
      await page.locator("#frame-picker summary").click();
      await page.locator("#frame-id").fill(String(frame+1)); await page.locator("#open-frame").click();
      await page.waitForFunction(n=>Number(new URL(location.href).searchParams.get("frame"))===n,frame+1);
      assert.ok(await page.locator("#vehicle-focus").isHidden()); assert.ok(await buttons.first().isDisabled());
      assert.deepEqual(errors,[]);
      results.push({recording,frame,viewport_width:width,question:words,source:data.answer_source,first_display_ms:ms,
        choices:options.map(({track_id,class_name,position})=>({track_id,class_name,position})),
        selected:chosen,original_box_verified:true,local_selected_status:anchored.answer.answer_status,local_selected_answer:anchored.answer.answer,
        local_evidence:anchored.evidence.map(e=>({track:e.interval.track_id,frames:e.images.map(im=>im.frame_id)})),
        changed_question_status:stale.status(),next_frame_disables_choices:true,verified_explicit_hosted_request_without_forwarding:true});
      await page.close();
    }
  } finally { await browser.close(); }
  if (out) fs.writeFileSync(path.join(out,"choices.json"),JSON.stringify(results,null,2)+"\n");
  process.stdout.write(JSON.stringify(results)+"\n");
}
main().catch(error=>{console.error(error); process.exitCode=1;});
