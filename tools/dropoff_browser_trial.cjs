// Replay a predeclared sequence through the passenger web app. Requires Playwright.
// Usage: node tools/dropoff_browser_trial.cjs PROTOCOL.json NEW_OUTPUT_DIR
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { chromium } = require("playwright");
const sha = bytes => crypto.createHash("sha256").update(bytes).digest("hex");
const sleep = ms => new Promise(resolve => setTimeout(resolve,ms));

async function main() {
  const [protocolFile, output] = process.argv.slice(2);
  if (!protocolFile || !output || fs.existsSync(output)) throw new Error("Supply a frozen protocol and a new output directory.");
  const manifest = fs.readFileSync(protocolFile), plan = JSON.parse(manifest);
  const sourceDir = path.dirname(path.resolve(protocolFile));
  fs.mkdirSync(output,{recursive:true}); fs.writeFileSync(path.join(output,"protocol.json"),manifest);
  const rows = [], consoleErrors = [];
  const report = {protocol_sha256:sha(manifest),started_utc:new Date().toISOString(),rows,consoleErrors};
  const save = () => fs.writeFileSync(path.join(output,"result.json"),JSON.stringify(report,null,2));
  const browserArgs = ["--disable-gpu","--disk-cache-size=1048576"];
  if (plan.microphone) {
    const micFile = path.resolve(sourceDir,plan.microphone.audio);
    if (sha(fs.readFileSync(micFile)) !== plan.microphone.audio_sha256) throw new Error("Virtual microphone source changed.");
    browserArgs.push("--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
      "--autoplay-policy=no-user-gesture-required", "--use-file-for-fake-audio-capture="+micFile);
  }
  const browser = await chromium.launch({executablePath:process.env.PASSENGER_CHROMIUM || undefined, headless:true, args:browserArgs});
  const page = await browser.newPage({viewport:{width:1280,height:860}, permissions:plan.microphone?["microphone"]:[]});
  page.setDefaultTimeout(20000);
  page.on("pageerror", error=>consoleErrors.push(error.message)); save();
  const response = (suffix, trigger, timeout=210000) => Promise.all([
    page.waitForResponse(r=>new URL(r.url()).pathname.endsWith(suffix) && r.request().method()==="POST",{timeout}),
    trigger()
  ]).then(async ([r]) => {
    if (!r.ok()) throw new Error(r.status()+" "+await r.text().catch(()=>"request failed"));
    // Large original frames can evict a completed POST body from Chromium's inspector.
    // Read the same server-side session view that the page renders, using the no-cache API.
    if (suffix==="/api/sessions") {
      const previous=r.request().postDataJSON().previous;
      await page.waitForFunction(old=>{
        const src=document.querySelector("#frame").getAttribute("src");
        return src && (!old || !src.includes("/"+old+"/")) && !document.querySelector("#scene").disabled;
      },previous);
    }
    const imageUrl=await page.locator("#frame").getAttribute("src");
    const snapshot=await page.request.get(new URL(imageUrl.replace(/\/image$/, ""),plan.base_url).href);
    if (!snapshot.ok()) throw new Error("Cannot read the current browser session: "+snapshot.status());
    const current = await snapshot.json();
    if (suffix === "/audio") {
      const audioBytes = r.request().postDataBuffer();
      // Chromium can omit Blob contents from its debugging protocol. The service verifies and
      // hashes the original WAV before exposing the transcript and its digest together.
      if (!/^[a-f0-9]{64}$/.test(current.audio_sha256 || "")) throw new Error("The browser audio receipt has no verified WAV digest.");
      if (audioBytes && sha(audioBytes)!==current.audio_sha256) throw new Error("Captured and stored audio hashes differ.");
      current.uploaded_audio_sha256 = current.audio_sha256;
    }
    return current;
  });
  async function clickAction(action, control) {
    return response("/"+action,()=>page.locator("#"+control).click());
  }
  async function selectScene(label) {
    const select = page.getByRole("combobox",{name:"Road scene"});
    const selected = await select.locator("option:checked").textContent();
    if (selected !== label) await response("/api/sessions",()=>select.selectOption({label}));
    await page.waitForFunction(()=>!document.querySelector("#scene").disabled && document.querySelector("#frame").complete && document.querySelector("#frame").naturalWidth>0);
  }
  async function draw(box) {
    await page.locator("#draw").click();
    const canvas=page.locator("#overlay");
    await canvas.scrollIntoViewIfNeeded();
    const b=await canvas.boundingBox();
    const dim=await canvas.evaluate(n=>[n.width,n.height]);
    const X=x=>b.x+b.width*x/dim[0], Y=y=>b.y+b.height*y/dim[1];
    await page.mouse.move(X(box[0]),Y(box[1])); await page.mouse.down();
    await page.mouse.move(X(box[2]),Y(box[3]),{steps:8}); await page.mouse.up();
    return clickAction("area","use-box");
  }
  async function canvasEvidence() {
    await page.waitForFunction(()=>!document.querySelector("#scene").disabled && document.querySelector("#frame").complete);
    return page.locator("#overlay").evaluate(n=>{
      const data=n.getContext("2d").getImageData(0,0,n.width,n.height).data;
      let painted=0;
      for (let alpha=3; alpha<data.length; alpha+=4) if (data[alpha]) painted++;
      const photo=document.querySelector("#frame");
      return {width:n.width,height:n.height,original_width:photo.naturalWidth,original_height:photo.naturalHeight,
        legend:document.querySelector("#legend").textContent,painted_pixels:painted};
    });
  }
  async function rejectStale(current, token) {
    const result = await page.request.post(plan.base_url+"/api/sessions/"+current.session+"/confirm",
      {headers:{"X-Session-Revision":String(current.revision)},data:{proposal_id:token}});
    const receipt={status:result.status(),body:await result.json()};
    if (receipt.status!==409) throw new Error("Stale confirmation was accepted or failed unexpectedly.");
    return receipt;
  }
  let lastKnownToken=null;
  try {
    await page.goto(plan.base_url); await page.waitForFunction(()=>!document.querySelector("#scene").disabled);
    for (const c of plan.cases) {
      const r={id:c.id,started_utc:new Date().toISOString()}; rows.push(r); save();
      if (c.unless_confirmed && rows.find(item=>item.id===c.unless_confirmed)?.after?.state==="confirmed") {
        r.skip="Predeclared positive control skipped after successful confirmation on this recording."; save(); continue;
      }
      try {
        if (c.open_scene) await selectScene(c.open_scene);
        if (await page.locator("#new-request").isVisible()) await page.locator("#new-request").click();
        const audio = path.resolve(sourceDir,c.audio); const hash=sha(fs.readFileSync(audio));
        if (hash!==c.audio_sha256) throw new Error("Audio hash mismatch.");
        let transcribed;
        if (c.input === "virtual_microphone") {
          if (!plan.microphone || hash !== plan.microphone.audio_sha256 || !Number.isFinite(c.capture_ms)
              || c.capture_ms<500 || c.capture_ms>55000) throw new Error("Invalid predeclared microphone input.");
          await page.getByRole("button",{name:"Record voice",exact:true}).click();
          await page.getByRole("button",{name:"Stop recording",exact:true}).waitFor();
          await sleep(c.capture_ms);
          transcribed=await clickAction("audio","mic");
        } else {
          transcribed=await response("/audio",()=>page.locator("#file").setInputFiles(audio));
          if (transcribed.uploaded_audio_sha256!==hash) throw new Error("Uploaded source bytes differ from the frozen WAV.");
        }
        r.input=c.input || "wav_upload"; r.source_audio_sha256=hash; r.audio_sha256=transcribed.uploaded_audio_sha256;
        r.recognized=transcribed.transcript; r.speech_error=transcribed.speech_error;
        if (!r.recognized) throw new Error("Voice input did not give a transcript.");
        const box=page.locator("#instruction");
        await page.waitForFunction(()=>document.querySelector("#instruction").value.length>0 && !document.querySelector("#instruction").disabled);
        if (c.passenger_edit) await box.fill(c.passenger_edit);
        r.sent_text=await box.inputValue();
        const submitted=await clickAction("submit","submit");
        r.initial=submitted.view; r.source_session=submitted.session; r.initial_utc=new Date().toISOString(); save();
        await page.waitForFunction(()=>!document.querySelector("#scene").disabled);
        r.overlay=await canvasEvidence();
        await page.screenshot({path:path.join(output,c.id+"-initial.png")});
        let current=submitted;
        if (c.marked_box_xyxy && ["proposed","needs_clarification","insufficient_visual_evidence"].includes(current.view.state)) {
          current=await draw(c.marked_box_xyxy); r.passenger_marked=current.view; save();
          r.passenger_overlay=await canvasEvidence();
          const stale=submitted.view.proposal_id || lastKnownToken || "never-issued";
          r.stale_check={token_source:submitted.view.proposal_id?"superseded same-turn":
            lastKnownToken?"earlier session or turn":"never-issued",...await rejectStale(current,stale)};
        }
        if (c.after==="confirm_if_proposed" && current.view.state==="proposed") {
          lastKnownToken=current.view.proposal_id;
          current=await clickAction("confirm","confirm"); r.after=current.view;
        } else if (c.after==="cancel" || (c.after==="confirm_if_proposed" && current.view.state!=="proposed")) {
          if (await page.getByRole("button",{name:"Cancel request"}).isVisible()) {
            current=await clickAction("cancel","cancel"); r.after=current.view;
          }
        } else { r.after=current.view; }
        if (c.stale_after_cancel && submitted.view.proposal_id && current.view.state==="cancelled") {
          r.stale_after_cancel=await rejectStale(current,submitted.view.proposal_id);
        }
        if (plan.final_screenshots) {
          r.final_overlay=await canvasEvidence();
          await page.screenshot({path:path.join(output,c.id+"-final.png")});
        }
      } catch(error) {
        r.error=error.message;
        await page.screenshot({path:path.join(output,c.id+"-failure.png")}).catch(()=>{});
      }
      r.finished_utc=new Date().toISOString(); save();
      // Yield one browser cycle between logged interactions.
      await sleep(20);
    }
  } finally {
    report.finished_utc=new Date().toISOString(); save(); await browser.close();
  }
  if (rows.some(r=>r.error) || consoleErrors.length) process.exitCode=1;
}
main().catch(e=>{console.error(e.message);process.exitCode=1;});
