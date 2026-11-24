const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

async function main() {
  const base = process.env.FOOTAGE_BASE_URL;
  if (!base) throw Error("Set FOOTAGE_BASE_URL to a running release with both EMT recordings.");
  const out = process.env.FOOTAGE_BROWSER_OUT;
  if (out) fs.mkdirSync(out,{recursive:true});
  const browser = await chromium.launch({channel:"msedge",headless:true});
  const errors = [], reports = [];
  try {
    for (const [name, viewport] of [["desktop",{width:1366,height:850}],["phone",{width:390,height:844}]]) {
      const page = await browser.newPage({viewport,deviceScaleFactor:1});
      page.on("pageerror",error=>errors.push(error.message));
      const url = new URL("/footage",base);
      if (name==="phone") url.searchParams.set("recording","video_054907");
      await page.goto(url.href);
      await page.waitForFunction(()=>document.querySelectorAll("#examples button:not(:disabled)").length >= 3);
      await page.locator("#clip-frame").waitFor({state:"visible"});
      const opened = await page.locator("#clip-position").innerText();
      const start = Number(new URL(page.url()).searchParams.get("frame"));
      assert.ok(start>1);
      assert.equal(await page.locator("#scene option").count(),2);
      assert.equal(await page.locator("#go").isDisabled(),true);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth),true);
      const initial = await page.locator("#examples button").allTextContents();
      if (out) await page.screenshot({path:path.join(out,name+"-open.png")});
      const first = page.locator("#clip-first-marker");
      assert.ok(await first.isVisible());
      const focusId = (await page.locator("#box-label").innerText()).split(" · ").at(-1);
      const labelInside = await page.evaluate(()=>{
        const frame = document.getElementById("clip-screen").getBoundingClientRect();
        const tag = document.getElementById("box-label").getBoundingClientRect();
        return tag.left >= frame.left-2 && tag.right <= frame.right+2;
      });
      assert.ok(labelInside);
      await first.click();
      await page.waitForFunction(id=>Number(new URL(location.href).searchParams.get("frame"))<id,start);
      assert.equal(await page.locator("#box-label").innerText(),"Track "+focusId);
      const reachedFirst = Number(new URL(page.url()).searchParams.get("frame"));
      await page.locator("#frame-picker summary").click();
      await page.locator("#frame-id").fill(String(start));
      await page.getByRole("button",{name:"Open frame",exact:true}).click();
      await page.waitForFunction(id=>Number(new URL(location.href).searchParams.get("frame"))===id,start);
      const answers = [];
      async function doSearch(trigger, test) {
        const wait=page.waitForResponse(r=>r.url().endsWith("/api/footage") && r.request().method()==="POST");
        await trigger();
        const response=await wait;
        assert.equal(response.status(),200);
        const data=await response.json();
        await page.locator(".answer-card").waitFor();
        await page.waitForFunction(()=>!document.getElementById("question").disabled);
        assert.equal(data.question.query_frame,start);
        assert.equal(data.hosted_requested,false);
        assert.ok(data.evidence.every(e=>e.interval.end_frame<=start && e.images.every(im=>im.frame_id<=start)));
        if(test) await test(data);
        answers.push({words:data.question.question,status:data.answer.answer_status,source:data.answer_source,
          answer:data.answer.answer,frames:data.evidence.flatMap(e=>e.images.map(im=>im.frame_id))});
        return data;
      }
      await doSearch(()=>page.locator("#examples button").first().click(),
        data=>assert.ok(data.answer.answer_status==="answered" && data.answer.cited_evidence_ids.length));
      const motion=await doSearch(()=>page.locator("#examples button").nth(1).click(),data=>{
        assert.equal(data.answer.answer_status,"answered");
        assert.match(data.answer.answer,/between.+frame.+and.+frame/);
        assert.ok(data.evidence[0].images.length>=2);
      });
      assert.equal(await page.locator(".answer-head h2").innerText(),initial[1]);
      assert.equal(await page.locator("#question").inputValue(),initial[1]);
      assert.ok(motion.answer.answer.toLowerCase().includes("track "+focusId));
      await page.locator(".moment-thumb img").evaluateAll(images=>Promise.all(images.map(img=>img.decode())));
      if (out) await page.screenshot({path:path.join(out,name+"-answer.png")});
      const im=motion.evidence[0].images[0];
      await page.locator(".moment-thumb").first().click();
      await page.waitForFunction(fid=>new URL(location.href).searchParams.get("frame")===String(fid),im.frame_id);
      await page.locator("#evidence-box:not(.suggested-track)").waitFor({state:"visible"});
      assert.equal(await page.locator("#box-label").innerText(),"Track "+motion.evidence[0].interval.track_id);
      assert.equal(await page.locator(".answer-card").count(),1);
      assert.ok((await page.locator("#clip-frame").getAttribute("src")).endsWith("/"+im.frame_id));
      if (name==="phone") {
        await page.getByRole("button",{name:"Back to answer"}).click();
        assert.equal(await page.locator("#back-to-answer").isVisible(),false);
      } else {
        await page.getByRole("button",{name:"Return to the question frame"}).click();
      }
      await page.waitForFunction(fid=>new URL(location.href).searchParams.get("frame")===String(fid),start);
      if (name==="desktop") {
        await doSearch(()=>page.locator("#examples button").nth(2).click(), data=>assert.equal(data.answer.answer_status,"answered"));
        for (const words of ["Who won the football world cup?","How many vehicles can you see at 25 seconds?"]) {
          await page.getByLabel("Your question").fill(words);
          await doSearch(()=>page.locator("#go").click(),data=>{
            assert.notEqual(data.answer.answer_status,"answered");
          });
          assert.equal(await page.getByRole("button",{name:/Ask Gemini about/}).count(),0);
        }
        await page.getByLabel("Your question").fill("   ");
        assert.equal(await page.locator("#go").isDisabled(),true);
        const choices=await page.locator("#scene option").evaluateAll(nodes=>nodes.map(n=>n.value));
        await page.locator("#scene").selectOption(choices[1]);
        await page.waitForFunction(()=>document.querySelector("#clip-name").textContent.includes("054907") &&
          document.querySelectorAll("#examples button:not(:disabled)").length>=2);
        assert.equal(await page.locator(".answer-card").count(),0);
        const paused = await page.locator("#clip-frame").getAttribute("src");
        await page.locator("#clip-play").click();
        await page.waitForFunction(old => document.getElementById("clip-frame").getAttribute("src") !== old,paused);
        const playbackFrame = await page.evaluate(() => {
          const id = Number(document.getElementById("clip-frame").src.split("/").at(-1));
          const link = document.getElementById("passenger-link");
          if (!link.title.includes("frame "+id+" of")) throw Error("Passenger link points to an earlier frame");
          link.click();
          return id;
        });
        await page.locator("#frame").waitFor();
        await page.waitForFunction(id => document.getElementById("frame").naturalWidth > 0 &&
          document.getElementById("scene").selectedOptions[0].textContent.includes("frame "+id+" "),playbackFrame);
        await page.locator("#footage-link").click();
        await page.waitForFunction(id => document.getElementById("clip-frame")?.src.endsWith("/"+id) &&
          !document.getElementById("clip-play").disabled,playbackFrame);
        assert.equal(new URL(page.url()).searchParams.get("recording"),"video_054907");
        assert.equal(new URL(page.url()).searchParams.get("frame"),String(playbackFrame));
      }
      reports.push({name,opened,initial,firstRecordedMarker:reachedFirst,highlightedTrack:focusId,answers});
      await page.close();
    }
    assert.deepEqual(errors,[]);
    if(out) fs.writeFileSync(path.join(out,"browser-observations.json"),JSON.stringify(reports,null,2));
    console.log(JSON.stringify(reports));
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
