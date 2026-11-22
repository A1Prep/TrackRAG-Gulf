// Local Chromium drives the released upload and transcript-review controls.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const from = process.env.TRACKRAG_PLAYWRIGHT || "playwright";
const {chromium} = require(from);
const [base, directory] = process.argv.slice(2);
const read = name => JSON.parse(fs.readFileSync(path.join(directory, name), "utf8"));
const cohort = read("cohort.json"), bindings = read("bound.json");
const options = {headless:true, args:["--disable-gpu", "--disable-background-networking"]};
if (process.env.TRACKRAG_CHROMIUM) options.executablePath = process.env.TRACKRAG_CHROMIUM;

async function action(page, suffix, click) {
  const [response] = await Promise.all([page.waitForResponse(r => r.request().method()==="POST" &&
    new URL(r.url()).pathname.endsWith(suffix),{timeout:120000}), click()]);
  assert(response.ok(), String(response.status()));
  // Chrome can evict a delayed upload's response body from its inspector cache.
  // The ordinary, revision-bound session GET is also what this page uses on refresh.
  const endpoint=new URL(response.url()).pathname.replace(/\/[^/]+$/,"");
  const current=await page.request.get(base+endpoint);assert(current.ok());
  return current.json();
}

(async () => {
  const browser = await chromium.launch(options);
  try {
    for (const [index, frozen] of cohort.selected.entries()) {
      const bound = bindings.rows[index]; assert.equal(frozen.move, bound.move);
      if (!bound.wav) continue;
      const page = await browser.newPage();
      try {
        const pageErrors=[]; page.on("pageerror", e=>pageErrors.push(String(e)));
        await page.route("**/*", route => new URL(route.request().url()).origin===base ? route.continue() : route.abort());
        await page.goto(base);
        await page.waitForFunction(()=>!document.querySelector("#instruction").disabled &&
          document.querySelector("#frame").complete && document.querySelector("#frame").naturalWidth>0);
        const heard = await action(page,"/audio",()=>page.locator("#file").setInputFiles(path.join(directory,"wavs",bound.wav)));
        await page.waitForFunction(()=>!document.querySelector("#instruction").disabled);
        assert.equal(heard.audio_sha256, heard.voice_stage==="review" ? bound.wav_sha256 : null);
        assert.equal(heard.view.state,"idle"); assert.equal(heard.view.proposal_id,null);
        assert.equal(heard.view.request_id,null); assert.deepEqual(heard.view.turns,[]);
        assert.equal(await page.locator("#confirm").isVisible(),false);
        const chance=await page.request.post(base+"/api/sessions/"+heard.session+"/confirm", {
          headers:{"X-Session-Revision":String(heard.revision)},data:{proposal_id:"no-proposal"}});
        assert.equal(chance.status(),409);
        const input=page.locator("#instruction");
        const draft=await input.inputValue(); assert.equal(draft,heard.transcript || "");
        let corrected=null, button=null;
        if (heard.voice_stage==="review") {
          assert(await input.isEditable());assert(heard.transcript.trim());
          button=await page.locator("#submit").innerText();assert.equal(button,"Send reviewed transcript");
          await input.fill(frozen.reference);
          corrected=await input.inputValue();assert.equal(corrected,frozen.reference);
          assert.equal(await page.locator("#confirm").isVisible(),false);
        } else assert.equal(heard.voice_stage,"failed");
        const finish=await action(page,"/cancel",()=>page.locator("#cancel").click());
        assert.equal(finish.view.state,"cancelled");assert.equal(finish.view.proposal_id,null);
        assert.deepEqual(pageErrors,[]);
        const row={move:frozen.move,session:heard.session,stage:heard.voice_stage,error:heard.speech_error,
          audio_sha256:bound.wav_sha256,raw_displayed:draft,scripted_edited_value:corrected,reviewed_submit_label:button,
          no_turn_or_proposal_on_upload:true,confirmation_without_proposal_http:chance.status(),
          explicitly_cancelled:finish.view.state==="cancelled"};
        fs.writeFileSync(path.join(directory,"browser",bound.move+".json"),JSON.stringify(row,null,2)+"\n",{flag:"wx"});
        console.log(index+1, frozen.move, heard.voice_stage);
      } finally {await page.close();}
    }
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
