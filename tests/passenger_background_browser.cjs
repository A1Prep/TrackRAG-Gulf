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
const fixtureBays = [
  {id:"bay-1",polygon_xy:[[70,190],[260,160],[335,239],[70,300]],center_xy:[185,220],passenger_tap_xy:null},
  {id:"bay-2",polygon_xy:[[260,160],[435,135],[510,193],[335,239]],center_xy:[380,182],passenger_tap_xy:null}
];
function bayOffer(bay, selected_by) {
  const xs=bay.polygon_xy.map(p=>p[0]), ys=bay.polygon_xy.map(p=>p[1]);
  return {state:"proposed",bay_outline:bay,proposal_id:"offered-"+bay.id,selected_by,
    result:{destination_point_xy:bay.center_xy,destination_box_xyxy:[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)]}};
}
const latch = () => {
  let release, notify;
  return {hold:new Promise(done => { release=done; }),entered:new Promise(done => { notify=done; }),
    release:() => release(),notify:() => notify()};
};

async function browserCase(browser,url,{indexed=false,example=null}={}) {
  const page = await browser.newPage();
  const requests = [], sessions = new Map(), errors = [];
  const mock = {page,requests,errors,holdAck:null,holdPoll:null,polls:[],current:null};
  page.on("pageerror",error=>errors.push(error.message));
  await page.route("**/api/**",async route => {
    const req = route.request(), pathname = new URL(req.url()).pathname;
    const json = obj => route.fulfill({json:obj}).catch(error => {
      if (!/closed|abort|invalid interception/i.test(String(error))) throw error;
    });
    if (pathname==="/api/scenes") return json({scenes:indexed ? scenes.map(s=>({...s,parking_areas_indexed:true})) : scenes,
      footage_scenes:[],speech_available:true});
    if (pathname==="/api/sessions") {
      const {scene} = req.postDataJSON(), id="s"+(sessions.size+1);
      const s={session:id,scene,revision:0,view:view(),voice_stage:null,pending_grounding:null,example_request:example};
      sessions.set(id,s); mock.current=s; return json(s);
    }
    const [, , , id, action] = pathname.split("/"), s=sessions.get(id);
    if (action==="image") return route.fulfill({contentType:"image/svg+xml",body:'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="gray"/></svg>'});
    if (!action) {
      if (mock.failReadbacks > 0) { mock.failReadbacks--; return route.abort("failed"); }
      mock.polls.push(Date.now()); const snapshot=clone(s), gate=mock.holdPoll;
      if (gate) { mock.holdPoll=null; gate.notify(); await gate.hold; }
      return json(gate?.response ?? snapshot);
    }
    const body=action==="audio" ? {} : req.postDataJSON();
    requests.push({action,body,session:id});
    s.revision++;
    if (action==="begin-audio") s.voice_stage="recording";
    if (action==="audio") {
      s.voice_stage="review"; s.transcript="Just behind the last car";
      if(mock.loseAudioReply) {
        assert.equal(req.headers()["content-type"],"audio/wav");
        s.last_action_id=req.headers()["x-action-id"];
        assert.match(s.last_action_id,/^[a-f0-9-]{36}$/);
        mock.failReadbacks=mock.loseAudioReadbacks || 0;
        return route.abort("connectionreset");
      }
    }
    if (action==="submit") {
      s.voice_stage=null;
      if (mock.loseImmediateReply && body.text === "Yes") {
        s.last_action_id=req.headers()["x-action-id"];
        s.pending_grounding=null;
        s.view=view("needs_clarification",{turns:[...s.view.turns,{role:"user",text:body.text}],
          result:{clarification_question:"Please mark the ground you mean."}});
        mock.failReadbacks=3;
        return route.abort("failed");
      }
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
      const tap=[(body.box[0]+body.box[2])/2,(body.box[1]+body.box[3])/2];
      s.view=mock.manualBays ? view("needs_clarification",{turns:s.view.turns,selected_by:"passenger",
        bay_choices:(mock.manualBays===1?fixtureBays.slice(0,1):fixtureBays).map(bay=>({...bay,passenger_tap_xy:tap}))}) :
        view("proposed",{turns:s.view.turns,selected_by:"passenger",proposal_id:"mine",
          result:{destination_point_xy:tap,destination_box_xyxy:body.box}});
    }
    if (action==="select-bay") {
      assert.equal(req.headers()["x-session-revision"],String(s.revision-1));
      const bay=s.view.bay_choices.find(option=>option.id===body.bay_id);
      assert(bay); s.view=view("proposed",{turns:s.view.turns,...bayOffer(bay,s.view.selected_by)});
    }
    if (action==="confirm") {
      assert.equal(body.proposal_id,s.view.proposal_id); s.view.state="confirmed";
    }
    return json(s);
  });
  mock.finish = (extra={}) => {
    const s=mock.current; s.revision++; s.pending_grounding=null;
    s.view=view("proposed",{turns:s.view.turns,proposal_id:"automatic",selected_by:"model",
      result:{destination_point_xy:[410,150],destination_box_xyxy:[400,140,420,160]},...extra});
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
async function clickPhoto(page,[x,y]) {
  const canvas=page.locator("#overlay"), size=await canvas.boundingBox();
  await canvas.click({position:{x:x*size.width/640,y:y*size.height/360}});
}
async function pixelAlpha(page,point) {
  return page.locator("#overlay").evaluate((canvas,[x,y])=>canvas.getContext("2d").getImageData(x,y,1,1).data[3],point);
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
  const browser=await chromium.launch({executablePath:process.env.PASSENGER_CHROMIUM||undefined,headless:true,args:["--disable-gpu"]});
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
    // A broader search for marked bays always requires the passenger's explicit choice.
    for (const initial of [
      {state:"needs_clarification",result:{clarification_question:"Can you name another place?"}},
      {state:"error",result:null,error:{code:"timeout"},landmark_error_guidance:"The photo check timed out."}
    ]) {
      const s=await browserCase(browser,url,{indexed:true}), p=s.page;
      await submitTyped(p); await p.locator("#take-over-area").waitFor();
      s.finish({...initial,request_id:"initial",proposal_id:null});
      await p.locator("#other-bays").waitFor({state:"visible"});
      if (initial.state==="error") {
        await heading(p,"The photo check timed out");
        assert(await p.locator("#confirm").isHidden());
        assert.equal(await p.locator("#last-request p").textContent(),words);
      }
      assert.equal(s.requests.filter(r=>r.action==="submit").length,1);
      await p.locator("#other-bays").click();
      await p.locator("#take-over-area").waitFor();
      const submitted=s.requests.filter(r=>r.action==="submit");
      assert.deepEqual(submitted.at(-1).body,{text:"Find the marked parking bays.",mode:"typed",replaces_request_id:"initial",background:true});
      s.finish({state:"needs_clarification",request_id:"alternative",proposal_id:null,result:{clarification_question:"No marked bays."}});
      await p.waitForFunction(()=>document.querySelector("#destination-panel").dataset.state==="clarify");
      assert(await p.locator("#other-bays").isHidden()); assert.deepEqual(s.errors,[]); await p.close();
    }
    for (const [indexed,code] of [[false,"timeout"],[true,"input_or_audit_changed"]]) {
      const s=await browserCase(browser,url,{indexed}), p=s.page;
      await submitTyped(p); await p.locator("#take-over-area").waitFor();
      s.finish({state:"error",request_id:"initial",proposal_id:null,result:null,error:{code},
        landmark_error_guidance:code==="timeout" ? "The photo check timed out." : null});
      await heading(p,"The request couldn’t finish");
      assert(await p.locator("#other-bays").isHidden());
      assert(await p.locator("#confirm").isHidden()); await p.close();
    }
    // A first-time passenger can mark without typing, or run the photo example with one click.
    {
      const s=await browserCase(browser,url),p=s.page;
      await clickPhoto(p,[480,282]);
      await heading(p,"Choose a spot on the photo");
      assert.deepEqual(s.requests[0].body,{text:"Check my marked parking spot.",mode:"typed",mark_on_photo:true});
      assert(await p.locator("#use-box").isEnabled());
      assert(await p.locator("#confirm").isHidden());
      assert(!s.requests.some(r=>r.action==="area"));
      await p.locator("#use-box").click();
      await heading(p,"Is this your intended location?");
      const mark=s.requests.find(r=>r.action==="area").body.box;
      assert(Math.abs((mark[0]+mark[2])/2-480)<2);
      assert(Math.abs((mark[1]+mark[3])/2-282)<2);
      assert.deepEqual(s.errors,[]); await p.close();
    }
    {
      const s=await browserCase(browser,url),p=s.page;
      await p.setViewportSize({width:320,height:780});
      assert(await p.evaluate(()=>document.documentElement.scrollWidth <= innerWidth));
      await p.getByRole("button",{name:"Mark on photo",exact:true}).click();
      await heading(p,"Choose a spot on the photo");
      assert.deepEqual(s.requests[0].body,{text:"Check my marked parking spot.",mode:"typed",mark_on_photo:true});
      assert(await p.locator("#use-box").isDisabled());
      assert(await p.locator("#confirm").isHidden());
      assert.deepEqual(s.errors,[]); await p.close();
    }
    {
      const s=await browserCase(browser,url,{example:words}),p=s.page;
      await p.locator("#use-example").click(); await p.locator("#take-over-area").waitFor();
      assert.deepEqual(s.requests[0].body,{text:words,mode:"typed",background:true});
      assert.deepEqual(s.errors,[]); await p.close();
    }
    // Direct area/vehicle requests stay on their immediate server routes.
    for (const [button,field] of [["submit-area","mark_on_photo"],["submit-pointing","point_to_vehicles"]]) {
      const s=await browserCase(browser,url),p=s.page; await p.locator("#instruction").fill(words);
      if(button==="submit-pointing") {
        assert(await p.locator("#submit-pointing").isHidden());
        await p.locator("#other-photo-options summary").click();
      }
      await p.locator("#"+button).click(); await p.waitForFunction(()=>document.querySelector('#form').hidden);
      assert.deepEqual(s.requests[0].body,{text:words,mode:"typed",[field]:true}); assert.deepEqual(s.errors,[]); await p.close();
    }
    // Whisper completed, but Edge lost the HTTP reply. A previous unrelated typing draft must never become the transcript.
    for(const loseAudioReadbacks of [0,3]) {
      const s=await browserCase(browser,url),p=s.page;
      await p.locator("#instruction").fill("A different typed destination");
      s.loseAudioReply=true; s.loseAudioReadbacks=loseAudioReadbacks;
      await reviewVoice(p);
      assert.equal(s.requests.filter(r=>r.action==="audio").length,1);
      assert.equal(s.requests.filter(r=>r.action==="submit").length,0);
      assert(await p.locator("#error").isHidden()); await p.close();
    }
    for (const boxes of [[[44,80,125,146]], [[44,80,125,146],[331,93,412,163]]]) {
      const s=await browserCase(browser,url),p=s.page;
      await submitTyped(p); await p.locator("#take-over-area").waitFor();
      s.finish({state:"needs_clarification",references_only:true,proposal_id:null,
        result:{anchor_boxes_xyxy:boxes,clarification_question:"Could you mark another visible spot?"}});
      const name=boxes.length===1 ? "vehicle" : "vehicles";
      await heading(p,"Check the outlined "+name);
      assert.equal(await p.locator("#legend").textContent(), (boxes.length===1 ? "Vehicle" : "Vehicles")
        +" outlined in blue · Drop-off point unverified");
      for (const [x1,y1,,y2] of boxes) assert(await pixelAlpha(p,[x1,Math.floor((y1+y2)/2)])>0);
      assert(await p.locator("#confirm").isHidden()); assert.equal(await pixelAlpha(p,[260,250]),0);
      assert.deepEqual(s.errors,[]); await p.close();
    }
    // Answering a word or vehicle question sends a reply. It never invokes the location-save action.
    for (const scenario of [
      {field:{word_confirmation:true},heading:"Check one word",question:"Did you mean the white car?",button:"answer-yes",reply:"Yes"},
      {field:{references_only:true,reference_kind:"possible_vehicle"},heading:"Is this the vehicle you meant?",question:"Is that the silver car you meant?",button:"answer-no",reply:"No"},
      {field:{reference_kind:"vegetation",tree_side_choices:["left","right"]},heading:"Which tree do you mean?",
        question:"There are trees on both sides. Which tree do you mean?",button:"tree-right",reply:"The tree on the right"},
    ]) {
      const s=await browserCase(browser,url),p=s.page;
      await submitTyped(p); await p.locator("#take-over-area").waitFor();
      s.finish({state:"needs_clarification",proposal_id:null,...scenario.field,
        result:{clarification_question:scenario.question,anchor_boxes_xyxy:[[44,80,125,146]]}});
      await heading(p,scenario.heading);
      assert(await p.locator("#show-driver").isHidden());
      assert(await p.locator("#confirm").isHidden());
      await p.locator("#"+scenario.button).click(); await p.locator("#take-over-area").waitFor();
      assert.deepEqual(s.requests.at(-1).body,{text:scenario.reply,mode:"typed",background:true});
      s.finish({state:"needs_clarification",proposal_id:null,result:{clarification_question:"Please mark the ground you mean."},
        turns:[{role:"user",text:words},{role:"assistant",text:scenario.question},{role:"user",text:scenario.reply}]});
      await p.getByText("Your last reply",{exact:true}).waitFor();
      await p.locator("#history summary").click();
      assert((await p.locator("#messages .user").first().textContent()).endsWith(words));
      assert.equal(await p.locator("#last-request p").textContent(),scenario.reply);
      assert(await p.locator("#answer-yes").isHidden()); assert(await p.locator("#answer-no").isHidden());
      assert(!s.requests.some(r=>r.action==="confirm")); assert.deepEqual(s.errors,[]); await p.close();
    }
    // A completed rapid reply with a lost acknowledgement is recognized by its action ID.
    {
      const s=await browserCase(browser,url),p=s.page; await submitTyped(p); await p.locator("#take-over-area").waitFor();
      s.finish({state:"needs_clarification",word_confirmation:true,proposal_id:null,
        result:{clarification_question:"Did you mean the white car?"}});
      await heading(p,"Check one word"); s.loseImmediateReply=true; await p.locator("#answer-yes").click();
      await p.getByText("Your last reply",{exact:true}).waitFor();
      assert.equal(await p.locator("#last-request p").textContent(),"Yes");
      assert.equal(await p.locator("#status").textContent(),"Please mark the ground you mean.");
      assert(await p.locator("#error").isHidden()); assert(await p.locator("#form").isHidden());
      assert.equal(s.requests.filter(r=>r.body.text==="Yes").length,1); await p.close();
    }
    // A single verified tree can be shown to the driver. A general greenery outline cannot.
    for (const side of [null,"right"]) {
      const s=await browserCase(browser,url),p=s.page;
      await p.locator("#instruction").fill("Could you drop me next to the tree on the right?");
      await p.locator("#submit").click(); await p.locator("#take-over-area").waitFor();
      s.finish({state:"needs_clarification",proposal_id:null,references_only:true,reference_kind:"vegetation",
        tree_landmark_side:side,result:{anchor_boxes_xyxy:[[315,35,450,280]],clarification_question:"Where would you like to get out?"}});
      await heading(p,side ? "Tree on the right" : "I found the roadside greenery");
      assert(await p.locator("#confirm").isHidden()); assert(await p.locator("#tree-right").isHidden());
      if (side) {
        const actions=s.requests.length;
        await p.getByRole("button",{name:"Show this tree to the driver"}).click();
        assert.match(await p.locator("#driver-photo").getAttribute("aria-label"),/tree outlined in blue/);
        assert.match(await p.locator("#driver-note").innerText(),/tree.*exact drop-off point is still open/);
        await p.keyboard.press("Escape"); assert(await p.getByRole("dialog").isHidden()); assert.equal(s.requests.length,actions);
      } else assert(await p.locator("#show-driver").isHidden());
      await p.close();
    }
    // A rejected tap can return to a checked parking choice while preserving the original tree request.
    {
      const s=await browserCase(browser,url),p=s.page;
      const treeWords="Near the tree by the shops on the right";
      await p.locator("#instruction").fill(treeWords); await p.locator("#submit").click();
      await p.locator("#take-over-area").waitFor();
      s.finish({state:"needs_clarification",request_id:"tree-original",proposal_id:null,references_only:true,
        reference_kind:"vegetation",tree_parking_available:true,
        result:{anchor_boxes_xyxy:[[350,40,540,180]],clarification_question:"I couldn’t verify a parking spot at your mark."}});
      await p.getByRole("button",{name:"Show parking near this tree"}).waitFor();
      assert(await p.locator("#confirm").isHidden());
      await p.getByRole("button",{name:"Show parking near this tree"}).click();
      await p.locator("#take-over-area").waitFor();
      assert.equal(s.requests.at(-1).body.text,treeWords);
      assert.equal(s.requests.at(-1).body.replaces_request_id,"tree-original");
      assert.deepEqual(s.errors,[]); await p.close();
    }
    // A same-row alternative keeps its qualification when the passenger reviews and saves the ring.
    {
      const s=await browserCase(browser,url), p=s.page;
      const detail="The parking is in the same row. I couldn’t verify that it is immediately beside the sedan.";
      await submitTyped(p); await p.locator("#take-over-area").waitFor();
      const offered=bayOffer(fixtureBays[0],"model");
      offered.result.anchor_boxes_xyxy=[[350,100,490,185]];
      s.finish({...offered,near_vehicle_parking:true,near_vehicle_detail:detail});
      await heading(p,"Is this your intended bay?");
      assert((await p.locator("#status").innerText()).includes(detail));
      assert.match(await p.locator("#legend").innerText(),/Blue outline around the vehicle/);
      await p.locator("#confirm").click(); await heading(p,"Location saved");
      assert((await p.locator("#status").innerText()).includes(detail));
      assert.deepEqual(s.errors,[]); await p.close();
    }
    // A foliage outline remains visible when the passenger saves a parking point near it.
    {
      const s=await browserCase(browser,url), p=s.page;
      await p.locator("#instruction").fill("Near the trees"); await p.locator("#submit").click();
      await p.locator("#take-over-area").waitFor();
      const offered=bayOffer(fixtureBays[0],"model");
      offered.result.anchor_boxes_xyxy=[[360,50,530,135]];
      s.finish({...offered,reference_kind:"vegetation"}); await heading(p,"Is this your intended bay?");
      assert.match(await p.locator("#status").innerText(),/greenery/); assert(await pixelAlpha(p,[360,90])>0);
      await p.locator("#confirm").click(); await heading(p,"Location saved");
      assert.match(await p.locator("#legend").innerText(),/greenery in blue/); assert(await pixelAlpha(p,[360,90])>0);
      assert.deepEqual(s.errors,[]); await p.close();
    }
    // Inspect a small matched vehicle, pan the zoomed photo, then mark the original image deliberately.
    {
      const s=await browserCase(browser,url), p=s.page;
      await submitTyped(p); await p.locator("#take-over-area").waitFor();
      s.finish({state:"needs_clarification",proposal_id:null,references_only:true,
        result:{anchor_boxes_xyxy:[[482,112,540,160]],clarification_question:"I found the vehicle. Can you mark the ground?"}});
      await heading(p,"Check the outlined vehicle");
      const actions=s.requests.length;
      await p.locator("#show-driver").click(); await p.getByRole("dialog").waitFor();
      assert.equal(await p.locator("#driver-words").innerText(),words);
      assert.match(await p.locator("#driver-note").innerText(),/exact drop-off point is still open/);
      assert(await p.locator("#driver-photo").evaluate(c=>c.getContext("2d").getImageData(0,0,1,1).data[3]===255));
      await p.keyboard.press("Escape"); assert(await p.getByRole("dialog").isHidden());
      assert.equal(s.requests.length,actions,"showing the landmark must not submit anything");
      await p.locator("#zoom-in").click();
      const placement = await p.locator("#photo-viewport").evaluate(v => {
        const r=v.getBoundingClientRect(), c=v.querySelector("canvas").getBoundingClientRect();
        return {x:c.x+511*c.width/640,y:c.y+150*c.height/360, scroll:v.scrollLeft, width:r.width};
      });
      assert(placement.scroll>placement.width*.26,"zoom should bring the right-side vehicle into view");
      await p.mouse.move(placement.x,placement.y); await p.mouse.down();
      await p.mouse.move(placement.x+40,placement.y+4,{steps:5}); await p.mouse.up();
      assert(await p.locator("#draw-tools").isHidden(),"dragging to pan must not place a mark");
      const near = await p.locator("#overlay").evaluate(c => {
        const r=c.getBoundingClientRect(); return {x:r.x+.79*r.width,y:r.y+.51*r.height};
      });
      await p.mouse.click(near.x,near.y); await p.locator("#use-box").waitFor();
      assert.equal(s.requests.filter(r=>r.action==="area").length,0,"a tap is a draft until the user checks it");
      await p.locator("#use-box").click(); await heading(p,"Is this your intended location?");
      const b=s.requests.find(r=>r.action==="area").body.box;
      assert(Math.abs((b[0]+b[2])/2-640*.79)<2 && Math.abs((b[1]+b[3])/2-360*.51)<2);
      await p.close();
    }
    // An explicitly marked point keeps its authorship and uncertainty through Save.
    {
      const s=await browserCase(browser,url),p=s.page;
      await submitTyped(p); await p.locator("#take-over-area").waitFor();
      const marked={...fixtureBays[0],extent:"interior_point_only",passenger_tap_xy:fixtureBays[0].center_xy};
      s.finish({...bayOffer(marked,"passenger"),passenger_marked_vehicle_location:true});
      await heading(p,"Is this the spot you meant?");
      assert.match(await p.locator("#legend").innerText(),/your chosen spot/);
      assert.match(await p.locator("#status").innerText(),/parking pavement you marked.*vehicle is still uncertain/);
      await p.locator("#confirm").click();await heading(p,"Your marked location was saved");
      assert.match(await p.locator("#status").innerText(),/exact point you chose.*vehicle is still uncertain/);
      await p.close();
    }
    // Single automatic bay displays its perspective polygon and centre only; new input clears both.
    {
      const s=await browserCase(browser,url),p=s.page; await submitTyped(p); await p.locator("#take-over-area").waitFor();
      s.finish(bayOffer(fixtureBays[0],"model")); await heading(p,"Is this your intended bay?");
      assert.equal(await p.locator("#confirm").textContent(),"Save this bay");
      assert.equal(await p.locator("#legend").textContent(),"Gold ring at bay centre");
      assert(await pixelAlpha(p,[120,237])>0); assert.equal(await pixelAlpha(p,[320,170]),0);
      await p.locator("#zoom-in").click(); assert.equal(await p.locator("#zoom-level").textContent(),"1.5×");
      assert(await pixelAlpha(p,[120,237])>0); await p.locator("#confirm").click(); await heading(p,"Location saved");
      assert.match(await p.locator("#status").textContent(),/outlined parking bay as your intended drop-off/);
      await p.locator("#new-request").click(); assert.equal(await pixelAlpha(p,[120,237]),0);
      assert.deepEqual(s.errors,[]); await p.close();
    }
    // A stripe can border two bays. A button or a tap inside a single bay selects only that candidate.
    for (const mode of ["manual buttons","automatic photo","automatic label"]) {
      const s=await browserCase(browser,url),p=s.page,manual=mode.startsWith("manual");
      if (mode==="automatic label") await p.evaluate(()=>{
        const ctx=document.querySelector("#overlay").getContext("2d"),draw=ctx.fillText.bind(ctx);
        ctx.fillText=function(label,x,y) {
          if (label==="Bay A") {
            const t=ctx.getTransform(); window.bayALabel=[t.a*(x+2)+t.e,t.d*y+t.f];
          }
          return draw(label,x,y);
        };
      });
      if (manual) {
        s.manualBays=true; await p.setViewportSize({width:390,height:844});
        await p.locator("#instruction").fill(words); await p.locator("#submit-area").click();
        await clickPhoto(p,[297.5,199.5]); await p.locator("#use-box").click();
      } else {
        await submitTyped(p); await p.locator("#take-over-area").waitFor();
        s.finish({state:"needs_clarification",proposal_id:null,result:null,bay_choices:fixtureBays});
      }
      await heading(p,"Which parking bay do you mean?");
      assert.deepEqual(await p.locator("#bay-buttons button").allTextContents(),["Bay A","Bay B"]);
      assert(await p.locator("#confirm").isHidden()); assert(await pixelAlpha(p,[120,237])>0);
      await clickPhoto(p,[297.5,199.5]); assert.equal(s.requests.filter(r=>r.action==="select-bay").length,0);
      const index=manual?1:0;
      if (manual) await p.getByRole("button",{name:"Choose bay B"}).press("Enter");
      else if (mode==="automatic label") await clickPhoto(p,await p.evaluate(()=>window.bayALabel));
      else await clickPhoto(p,fixtureBays[index].center_xy);
      await heading(p,"Is this your intended bay?"); assert(await p.locator("#bay-tools").isHidden());
      assert.deepEqual(s.requests.at(-1).body,{bay_id:fixtureBays[index].id});
      assert.equal(await p.locator("#legend").textContent(),manual?"Gold ring at bay centre · Blue diamond at your tap":"Gold ring at bay centre");
      assert.equal(await pixelAlpha(p,manual?[120,237]:[470,185]),0);
      await p.locator("#confirm").click(); await heading(p,"Location saved"); assert.deepEqual(s.errors,[]); await p.close();
    }
    // A visible bay beside an ambiguous stripe needs a choice before the save action appears.
    {
      const s=await browserCase(browser,url),p=s.page;s.manualBays=1;
      await p.locator("#instruction").fill(words);await p.locator("#submit-area").click();
      await clickPhoto(p,[72,220]);await p.locator("#use-box").click();await heading(p,"Did you mean this side of the line?");
      assert.match(await p.locator("#status").textContent(),/Bay A is the visible bay beside your mark/);
      assert.deepEqual(await p.locator("#bay-buttons button").allTextContents(),["Bay A"]);assert(await p.locator("#confirm").isHidden());
      await p.getByRole("button",{name:"Choose bay A"}).click();await heading(p,"Is this your intended bay?");
      assert(await p.locator("#confirm").isVisible());assert.deepEqual(s.errors,[]);await p.close();
    }
    console.log("Browser scenarios passed, including quick replies and one- and two-vehicle references without destination pins.");
  } finally { await browser.close(); await new Promise(done=>server.close(done)); }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
