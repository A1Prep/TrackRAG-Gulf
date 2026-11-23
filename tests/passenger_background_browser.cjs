// Run with Node, Playwright and optionally PASSENGER_CHROMIUM pointing to a browser executable.
const assert = require("node:assert/strict");
const {readFile} = require("node:fs/promises");
const {createServer} = require("node:http");
const path = require("node:path");
const {chromium} = require("playwright");

const assets = path.resolve(__dirname,"../trackrag/passenger_ui");
const scenes = ["first","second"].map((key,i) => ({key,recording_id:"kartaview_"+(4572137+i),frame_id:14,width:640,height:360}));
const words = "Just behind the last white car on the right";
const clone = value => JSON.parse(JSON.stringify(value));
const view = (state="idle",extra={}) => ({state,turns:[],can_request_area:true,can_request_pointing:true,...extra});
const latch = () => {
  let release, notify;
  return {hold:new Promise(done => { release=done; }),entered:new Promise(done => { notify=done; }),
    release:() => release(),notify:() => notify()};
};

async function browserCase(browser,url) {
  const page = await browser.newPage();
  const requests = [], sessions = new Map(), errors = [];
  const mock = {page,requests,errors,holdAck:null,holdPoll:null,polls:[],current:null};
  page.on("pageerror",error=>errors.push(error.message));
  await page.route("**/api/**",async route => {
    const req = route.request(), pathname = new URL(req.url()).pathname;
    const json = obj => route.fulfill({json:obj}).catch(error => {
      if (!/closed|abort|invalid interception/i.test(String(error))) throw error;
    });
    if (pathname==="/api/scenes") return json({scenes,footage_scenes:[],speech_available:true});
    if (pathname==="/api/sessions") {
      const {scene} = req.postDataJSON(), id="s"+(sessions.size+1);
      const s={session:id,scene,revision:0,view:view(),voice_stage:null,pending_grounding:null};
      sessions.set(id,s); mock.current=s; return json(s);
    }
    const [, , , id, action] = pathname.split("/"), s=sessions.get(id);
    if (action==="image") return route.fulfill({contentType:"image/svg+xml",body:'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="gray"/></svg>'});
    if (!action) {
      mock.polls.push(Date.now()); const snapshot=clone(s), gate=mock.holdPoll;
      if (gate) { mock.holdPoll=null; gate.notify(); await gate.hold; }
      return json(gate?.response ?? snapshot);
    }
    const body=action==="audio" ? {} : req.postDataJSON();
    requests.push({action,body,session:id});
    s.revision++;
    if (action==="begin-audio") s.voice_stage="recording";
    if (action==="audio") { s.voice_stage="review"; s.transcript="Just behind the last car"; }
    if (action==="submit") {
      s.voice_stage=null;
      const turns=[{role:"user",text:body.text}];
      if (body.background) {
        s.pending_grounding={request_id:id+"-request",text:body.text,mode:body.mode,elapsed_s:0};
        s.view=view("processing",{turns});
        const held=mock.holdAck,snapshot=clone(s);
        if (held) { mock.holdAck=null; held.notify(); await held.hold; }
        return json(snapshot);
      }
      s.view=view("needs_clarification",{turns,selected_by:"passenger",pair_choices:[]});
    }
    if (action==="take-over-area" || action==="cancel") {
      assert.equal(body.request_id,id+"-request");
      s.pending_grounding=null;
      s.view=view(action==="cancel" ? "cancelled":"needs_clarification",{turns:s.view.turns,selected_by:"passenger"});
    }
    if (action==="area") {
      assert.equal(body.box.length,4);
      s.view=view("proposed",{turns:s.view.turns,selected_by:"passenger",proposal_id:"mine",
        result:{destination_point_xy:[320,180],destination_box_xyxy:body.box}});
    }
    if (action==="confirm") {
      assert.equal(body.proposal_id,s.view.proposal_id); s.view.state="confirmed";
    }
    return json(s);
  });
  mock.finish = () => {
    const s=mock.current; s.revision++; s.pending_grounding=null;
    s.view=view("proposed",{turns:s.view.turns,proposal_id:"automatic",selected_by:"model",
      result:{destination_point_xy:[410,150],destination_box_xyxy:[400,140,420,160]}});
  };
  await page.goto(url);
  await page.waitForFunction(()=>document.querySelector('#frame').naturalWidth===640 && !document.querySelector('#scene').disabled);
  return mock;
}

async function submitTyped(page) {
  await page.locator("#instruction").fill(words); await page.locator("#submit").click();
}
async function reviewVoice(page) {
  await page.locator("#file").setInputFiles({name:"synthetic.wav",mimeType:"audio/wav",buffer:Buffer.from("RIFFfixtureWAVE")});
  await page.waitForFunction(()=>document.querySelector('#submit').textContent==='Send reviewed transcript');
  assert.equal(await page.locator("#instruction").inputValue(),"Just behind the last car");
  await page.locator("#instruction").fill(words);
}
async function heading(page,title) {
  await page.waitForFunction(value=>document.querySelector('#destination-title').textContent===value,title);
}

async function main() {
  const server=createServer(async (req,res)=>{
    const name=new URL(req.url,"http://localhost").pathname.slice(1)||"index.html";
    if (!/^[\w.-]+$/.test(name)) { res.writeHead(404).end(); return; }
    try { const data=await readFile(path.join(assets,name));
      res.writeHead(200,{"Content-Type":({".html":"text/html",".js":"text/javascript",".css":"text/css"})[path.extname(name)]||"application/octet-stream"}).end(data);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise(done=>server.listen(0,"127.0.0.1",done));
  const url="http://127.0.0.1:"+server.address().port;
  const browser=await chromium.launch({executablePath:process.env.PASSENGER_CHROMIUM||undefined,headless:true});
  try {
    // Completed typed request restores the normal proposal, without blocking browser navigation during acknowledgment or polling.
    {
      const s=await browserCase(browser,url), p=s.page, ack=s.holdAck=latch();
      await submitTyped(p); await ack.entered;
      assert.equal(await p.locator("#scene").isEnabled(),true); ack.release();
      await p.locator("#take-over-area").waitFor(); assert.equal(await p.locator("#cancel").isEnabled(),true);
      assert.equal(await p.locator("#last-request p").textContent(),words);
      assert.deepEqual(s.requests.at(-1).body,{text:words,mode:"typed",background:true});
      await p.waitForTimeout(1200); assert(s.polls.length>=2 && s.polls.length<=4);
      assert(s.polls[1]-s.polls[0]>=400); s.finish(); await heading(p,"Is this your intended location?");
      assert(await p.locator("#take-over-area").isHidden());
      assert.equal(await p.locator("#confirm").textContent(),"Save this location");await p.locator("#confirm").click();
      await heading(p,"Location saved");assert.match(await p.locator("#status").textContent(),/yellow pin.*intended drop-off/);
      assert(await p.locator("#stopping-note").isVisible());assert.deepEqual(s.errors,[]); await p.close();
    }
    // A snapshot already in flight cannot replace a passenger's deliberately marked location.
    for (const mode of ["typed","voice"]) {
      const s=await browserCase(browser,url),p=s.page,held=s.holdPoll=latch();
      if (mode === "typed") await submitTyped(p);
      else { await reviewVoice(p); await p.locator("#submit").click(); }
      await held.entered; s.finish(); held.response=clone(s.current);
      await p.locator("#take-over-area").click(); await heading(p,"Choose a spot on the photo");
      await p.locator("#overlay").press("Enter"); await p.locator("#use-box").click(); await heading(p,"Is this your intended location?");
      held.release(); await p.waitForTimeout(650);
      assert.match(await p.locator("#status").textContent(),/blue pin/);
      assert.equal(await p.locator("#legend").textContent(),"Blue pin at your mark");
      assert.deepEqual(s.errors,[]); await p.close();
    }
    // A reviewed, edited transcript is sent once as voice; a late poll cannot undo cancellation.
    {
      const s=await browserCase(browser,url),p=s.page,held=s.holdPoll=latch();
      await reviewVoice(p); await p.locator("#submit").click(); await held.entered;
      assert.deepEqual(s.requests.find(r=>r.action==="submit").body,{text:words,mode:"voice",background:true});
      assert.equal(await p.locator("#last-request span").textContent(),"Your reviewed transcript");
      assert.equal(await p.locator("#last-request p").textContent(),words);
      s.finish(); held.response=clone(s.current);
      await p.locator("#cancel").click(); await heading(p,"Request cancelled"); held.release(); await p.waitForTimeout(650);
      assert(await p.locator("#new-request").isVisible()); assert.deepEqual(s.errors,[]); await p.close();
    }
    // A previous scene's late acknowledgement or poll cannot remove the new scene's voice review.
    for (const phase of ["holdAck","holdPoll"]) {
      const s=await browserCase(browser,url),p=s.page,held=s[phase]=latch();
      await submitTyped(p); await held.entered;
      if (phase==="holdPoll") {s.finish(); held.response=clone(s.current);}
      await p.locator("#scene").selectOption("second");
      await p.waitForFunction(()=>location.search.includes('second') && !document.querySelector('#scene').disabled);
      await reviewVoice(p); held.release(); await p.waitForTimeout(650);
      assert.equal(await p.locator("#scene").inputValue(),"second");
      assert.equal(await p.locator("#instruction").inputValue(),words); await heading(p,"Did I hear you correctly?");
      assert.deepEqual(s.errors,[]); await p.close();
    }
    // Direct area/vehicle requests stay on their immediate server routes.
    for (const [button,field] of [["submit-area","mark_on_photo"],["submit-pointing","point_to_vehicles"]]) {
      const s=await browserCase(browser,url),p=s.page; await p.locator("#instruction").fill(words);
      await p.locator("#"+button).click(); await p.waitForFunction(()=>document.querySelector('#form').hidden);
      assert.deepEqual(s.requests[0].body,{text:words,mode:"typed",[field]:true}); assert.deepEqual(s.errors,[]); await p.close();
    }
    console.log("8 browser scenarios passed: typed completion, typed and reviewed-voice marking, reviewed-voice cancellation, stale acknowledgment, stale poll, direct area and vehicle selection.");
  } finally { await browser.close(); await new Promise(done=>server.close(done)); }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
