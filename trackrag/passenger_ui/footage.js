import {FootagePlayer} from "./footage-player.js";

const get = id => document.getElementById(id);
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function action(label, fn, className = "text-btn") {
  const btn = node("button", label, className);
  btn.type = "button";
  btn.addEventListener("click", fn);
  return btn;
}
async function fetchJson(url, opts) {
  const response = await fetch(url, opts);
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body.error || "Could not open this recording.");
    error.status = response.status;
    throw error;
  }
  return body;
}
function formatTime(time) {
  const n = Math.round(10 * time);
  return Math.floor(n / 600) + ":" + String(Math.floor(n / 10) % 60).padStart(2, "0") + "." + n % 10;
}
const scene = get("scene"), question = get("question"), mode = get("answer-mode"), results = get("results");
const recordings = new Map(), bookmarks = new Map();
let busy = true, loading = true, sequence = 0, searchRun = null, contexts = {}, subjects = {};
let lastHints = "", highlight = null, answerMoment = null, submittedHint = null, firstMarker = null;
let chosenVehicle = null;
const player = new FootagePlayer(updatePlayer);
const recordingName = rec => "Recording " + rec.recording_id.replace(/^video_/, "");
const className = text => (text || "road user").toLowerCase().replace(/_/g, " ");
function canAsk() {
  return !busy && !loading && player.displayed && (!player.pending || player.playing);
}
function syncButtons() {
  get("go").disabled = !canAsk() || !question.value.trim();
  for (const btn of get("examples").children) btn.disabled = !canAsk();
}
function editing(enabled) {
  busy = !enabled;
  for (const control of [scene, question, mode, get("frame-id"), get("open-frame")]) control.disabled = !enabled;
  player.setLocked(!enabled);
  syncButtons();
}
function forgetVehicle() {
  chosenVehicle = null;
  get("vehicle-focus").hidden = true;
  for (const button of results.querySelectorAll('.vehicle-choices button[aria-pressed="true"]')) {
    button.setAttribute("aria-pressed","false");
  }
}
function clearResult() {
  results.replaceChildren();
  answerMoment = null;
  highlight = null;
  forgetVehicle();
  get("evidence-box").hidden = true;
  get("back-to-answer").hidden = true;
  get("progress").textContent = "";
}
function submitWords(words, displayWords = words, selection = null, imageRelation = false) {
  if (!canAsk()) return;
  question.value = displayWords;
  submittedHint = {words,displayWords,scene:scene.value,frame:player.displayed.id,selection,imageRelation};
  syncButtons();
  get("search").requestSubmit();
}
function examples(row, visibleTrack) {
  const context = contexts[row.id] || {};
  const subject = context.current || subjects[row.id];
  const hints = [];
  if (mode.value === "gemini") {
    hints.push(["Describe the road at this moment", "What can you see on the road at " + formatTime(row.time) + "?"]);
    if (subject) hints.push(["Describe the tracked " + className(subject.class_name), "What does track " + subject.id + " look like in this frame?"]);
    return hints;
  }
  hints.push(["Which road users are here?", "Which types of road users are detected in this frame?"]);
  if (subject) {
    const name = className(subject.class_name), track = "track " + subject.id;
    if (subject.first_frame < row.id) {
      hints.push(["How did the " + name + " move?", "How did " + track + " move across the image?"]);
    } else {
      hints.push(["Where is the " + name + " in the frame?", "Which side of the frame is " + track + " on?"]);
    }
    if (subject.first_frame !== undefined) {
      const outlined = visibleTrack === subject.id;
      const target = outlined ? "the highlighted " + name : name + " " + subject.id;
      const recent = subject.first_frame > player.frames[0].id && currentObservedSeconds(row,subject) <= 2;
      const label = outlined && recent ? "When did the " + name + " first appear?" : "When was " + target + " first detected?";
      hints.push([label, "When was " + track + " first detected?"]);
    }
  } else hints.push(["How many vehicles are visible?", "How many vehicles are detected in this frame?"]);
  return hints;
}
function refreshHints(row, visibleTrack) {
  const hints = examples(row, visibleTrack);
  const key = JSON.stringify(hints);
  if (lastHints === key) return;
  lastHints = key;
  const buttons = hints.map(([label, words]) => {
    const btn = action(label, () => submitWords(words,label), "footage-question");
    btn.setAttribute("aria-label", label);
    btn.title = words;
    return btn;
  });
  get("examples").replaceChildren(...buttons);
}
function positionBox(element,rect,rec) {
  if (!Array.isArray(rect) || rect.length!==4 || !rect.every(Number.isFinite)) return false;
  const [x0,y0,x1,y1]=rect;
  if (x0<0 || y0<0 || x1<=x0 || y1<=y0 || x1>rec.width || y1>rec.height) return false;
  Object.assign(element.style,{left:100*x0/rec.width+"%",top:100*y0/rec.height+"%",
    width:100*(x1-x0)/rec.width+"%",height:100*(y1-y0)/rec.height+"%"});
  return true;
}
function vehicleCrop(option) {
  const image = get("clip-frame");
  if (!image.complete || !image.naturalWidth) return null;
  const [left,top,right,bottom] = option.bbox_xyxy;
  const crop = node("canvas",undefined,"vehicle-crop");
  crop.width=140; crop.height=84; crop.setAttribute("aria-hidden","true");
  const context=crop.getContext("2d");
  if (!context) return null;
  const w=Math.min(image.naturalWidth,Math.max(24,(right-left)*1.55,(bottom-top)*1.55*crop.width/crop.height));
  const h=Math.min(image.naturalHeight,w*crop.height/crop.width);
  const x=Math.max(0,Math.min(image.naturalWidth-w,(left+right-w)/2));
  const y=Math.max(0,Math.min(image.naturalHeight-h,(top+bottom-h)/2));
  context.drawImage(image,x,y,w,h,0,0,crop.width,crop.height);
  context.strokeStyle="#ffdc62"; context.lineWidth=1.5;
  context.strokeRect((left-x)*crop.width/w,(top-y)*crop.height/h,(right-left)*crop.width/w,(bottom-top)*crop.height/h);
  return crop;
}
function revealBox(row, rec) {
  const box = get("evidence-box");
  box.hidden = true;
  const active = highlight?.key === scene.value && highlight.frame === row.id ? highlight : null;
  const subject = !answerMoment && !player.playing ? contexts[row.id]?.current : null;
  const rect = active?.rect || subject?.box;
  if (!positionBox(box,rect,rec)) return;
  const [x0,y0] = rect;
  box.classList.toggle("suggested-track",!active);
  box.classList.toggle("near-right",x0/rec.width > .63);
  box.classList.toggle("near-top",y0/rec.height < .10);
  get("box-label").textContent = active ? "Track " + active.track : className(subject.class_name) + " · " + subject.id;
  box.hidden = false;
  return active?.track ?? subject.id;
}
function currentObservedSeconds(row,subject,frames=player.frames) {
  const first = frames.find(f=>f.id===subject.first_frame);
  return first ? row.time-first.time : 0;
}
function showFirstMarker(row) {
  const marker = get("clip-first-marker"), subject = contexts[row.id]?.current;
  marker.hidden = true; firstMarker = null;
  if (busy || player.playing || !subject?.first_box || subject.first_frame >= row.id || player.frames.length < 2) return;
  const at = player.frames.findIndex(f=>f.id===subject.first_frame);
  if (at < 0) return;
  const label = "Go to the first detection of the highlighted track ("+subject.id+") at "+formatTime(player.frames[at].time);
  marker.title = label; marker.setAttribute("aria-label",label);
  const fraction = at/(player.frames.length-1);
  marker.style.left = "calc("+100*fraction+"% + "+(8-16*fraction)+"px)";
  firstMarker = {index:at,id:subject.first_frame,track:subject.id,rect:subject.first_box};
  marker.hidden=false;
}
function representative(frames,rec) {
  const threshold = frames[0].time + (frames.at(-1).time-frames[0].time)*.35;
  return frames.find(f=>{
    const data = contexts[f.id], subject = data?.current;
    if (f.time<threshold || !subject || data.count<3 || currentObservedSeconds(f,subject,frames)<.8) return false;
    const [x0,y0,x1,y1]=subject.box;
    return (x1-x0)*(y1-y0) >= rec.width*rec.height*.002;
  }) || frames.find(f=>f.time>=threshold && contexts[f.id]?.current) || frames[0];
}
function updatePlayer(writeAddress = true) {
  syncButtons();
  const row = player.displayed, rec = recordings.get(scene.value);
  if (chosenVehicle && (row?.id!==answerMoment?.frame || player.playing)) forgetVehicle();
  const atQuestion = row && row.id===answerMoment?.frame && scene.value===answerMoment.key && !player.playing && !player.pending;
  for (const button of results.querySelectorAll(".vehicle-choices button")) button.disabled = busy || !atQuestion;
  get("vehicle-focus").hidden = !atQuestion || !chosenVehicle || busy;
  if (!row || !rec) { get("evidence-box").hidden = true; return; }
  if (!writeAddress) return;
  const visibleTrack = revealBox(row, rec);
  showFirstMarker(row);
  const back = get("return-question-frame");
  if (back) back.hidden = answerMoment?.frame === row.id;
  const ctx = contexts[row.id];
  get("clip-info").textContent = (player.frames.length ? formatTime(player.frames.at(-1).time) + " · " + player.frames.length + " frames" : "") +
    (ctx ? " · " + ctx.count + (ctx.count === 1 ? " road user" : " road users") + " tracked here" : "");
  get("ask-time").textContent = formatTime(row.time);
  get("frame-id").value = String(row.id);
  refreshHints(row, visibleTrack);
  syncButtons();
  const passengerLink = get("passenger-link");
  const saved = bookmarks.get(rec.recording_id + ":" + row.id);
  passengerLink.href = saved ? "/?scene="+encodeURIComponent(saved) : "/api/footage-preview/"+scene.value+"/dropoff/"+row.id;
  passengerLink.title = "Open passenger drop-off on frame " + row.id + " of " + recordingName(rec);
  if (player.playing) return;
  const url = new URL(location.href);
  url.searchParams.set("scene",scene.value);
  url.searchParams.set("recording",rec.recording_id);
  url.searchParams.set("frame",String(row.id));
  if (url.href !== location.href) history.replaceState(null,"",url);
}
async function selectScene(target = null) {
  const run = ++sequence, anchor = recordings.get(scene.value);
  loading = true;
  clearResult();
  if (submittedHint) question.value = "";
  get("examples").replaceChildren();
  lastHints = ""; subjects = {}; contexts = {};
  submittedHint=null; firstMarker=null; get("clip-first-marker").hidden=true;
  get("ask-time").textContent = "";
  get("passenger-link").href = "/";
  get("passenger-link").removeAttribute("title");
  get("clip-name").textContent = "EMT " + recordingName(anchor).toLowerCase();
  get("clip-info").textContent = "";
  get("frame-picker").open = false;
  player.reset(scene.value,anchor);
  syncButtons();
  if (target !== null) void player.cover(target);
  try {
    const manifest = await fetchJson("/api/footage-preview/"+anchor.key+"/timeline");
    if (run !== sequence) return;
    const frames = manifest.frames;
    if (!frames.length) throw new Error("This original recording has no readable frames.");
    subjects = manifest.subjects || {};
    contexts = manifest.frame_context || {};
    const featured = representative(frames,anchor);
    const available = target === null || frames.some(row=>row.id===target);
    await player.load(frames, target === null ? featured.id : available ? target : anchor.frame_id);
    if (run !== sequence) return;
    loading = false; updatePlayer();
    if (!available && player.displayed) get("progress").textContent = "Frame "+target+" is unavailable. Showing frame "+player.displayed.id+".";
  } catch(error) {
    if (run !== sequence) return;
    loading = false; get("progress").textContent = error.message; syncButtons();
  }
}
async function keyForFrame(anchor, frame) {
  const label = recordings.get(anchor).recording_id+":"+frame;
  if (bookmarks.has(label)) return bookmarks.get(label);
  const bound = await fetchJson("/api/frames", {method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({scene:anchor,frame_id:frame,for_footage:true})});
  if (bound.frame_id!==frame || bound.recording_id!==recordings.get(anchor).recording_id)
    throw new Error("The recording changed. Select the displayed frame again.");
  bookmarks.set(label,bound.key);
  return bound.key;
}
function updateMode() {
  const hosted = mode.value === "gemini";
  get("mode-label").textContent = "Search settings · " + (hosted ? "Gemini visual review" : "Recorded tracks");
  get("mode-hint").textContent = hosted ? "Gemini receives privacy-processed evidence frames to describe visible details." :
    "Use recorded tracks for counts, positions and observed movement.";
  get("hint-description").textContent = hosted ? "Choose a question to send for visual review." : "Choose a question to search the frames seen so far.";
  if (player.displayed) updatePlayer();
}
async function inspect(frame, track = null, rect = null) {
  if (busy || loading || !answerMoment || answerMoment.key !== scene.value || frame > answerMoment.frame) return;
  const index = player.frames.findIndex(f => f.id === frame);
  if (index < 0) return;
  player.pause(false);
  if (chosenVehicle && (frame!==answerMoment.frame || track!==chosenVehicle.option.track_id)) forgetVehicle();
  highlight = rect && track !== null ? {key:scene.value,frame,track,rect} : null;
  await player.display(index);
  if (player.displayed?.id === frame) {
    get("back-to-answer").hidden = false;
    const back = get("return-question-frame");
    if (back) back.hidden = false;
    get("clip-screen").scrollIntoView({block:"center"});
    get("clip-play").focus({preventScroll:true});
  }
}
function addEvidence(card, item, cited) {
  const group = node("section", undefined, "evidence-group");
  const interval = item.interval, track = interval.track_id;
  const times = player.frames.filter(f => f.id===interval.start_frame || f.id===interval.end_frame);
  const time = times.map(f => formatTime(f.time)).join(" – ");
  group.append(node("h3", (track == null ? "Scene evidence" : "Track " + track) + (time ? " · "+time : "")));
  group.append(node("p", (cited ? "Cited in the answer. " : "Retrieved for comparison. ") +
    "Select a frame to see that moment in the player.", "evidence-note"));
  const strip = node("div", undefined, "moment-strip");
  for (const im of item.images) {
    const moment = player.frames.find(f => f.id === im.frame_id);
    if (!moment) continue;
    const label = formatTime(moment.time) + " · Frame " + im.frame_id;
    const thumb = action("", () => void inspect(im.frame_id,track,im.bbox_xyxy), "moment-thumb");
    thumb.setAttribute("aria-label","View "+(track==null ? "scene" : "track "+track)+" at "+label+" in player");
    const rec = recordings.get(scene.value);
    const panel = node("span",undefined,"moment-image"); panel.style.aspectRatio=rec.width+"/"+rec.height;
    const img = node("img"); img.alt = ""; img.loading = cited ? "eager" : "lazy"; img.src = im.url;
    panel.append(img);
    const marker = node("span",undefined,"moment-object");
    if (track!=null && positionBox(marker,im.bbox_xyxy,rec)) panel.append(marker);
    thumb.append(panel,node("span",label,"evidence-caption"));
    const tile = node("div",undefined,"moment-tile");
    const full = node("a","Open evidence image");
    full.href = im.url; full.target = "_blank"; full.rel = "noopener";
    full.setAttribute("aria-label","Open evidence image for frame "+im.frame_id+" in a new tab");
    tile.append(thumb,full); strip.append(tile);
  }
  group.append(strip); card.append(group);
}
function showAnswer(body, current, rec, displayWords) {
  const {answer,evidence} = body, status = answer.answer_status;
  const words = body.original_question || body.question.question;
  answerMoment = {key:scene.value,frame:current.id};
  const card = node("article",undefined,"answer-card"), header = node("div",undefined,"answer-head");
  const titles = {answered:"Answer",insufficient_evidence:"More evidence needed",error:"Search unavailable"};
  header.append(node("h2",body.vehicle_choices ? "Choose a vehicle" : body.image_plane_review && status==="answered" ? "Positions in the image" : status==="answered" ? displayWords : titles[status] || (evidence.length ? "Frames found" : "No matching evidence")),
    node("span", {local_tracks:"Recorded tracks",gemini:"Gemini visual review",evidence_only:"Frame search"}[body.answer_source], "answer-source"));
  card.append(header, node("p",recordingName(rec)+" · Asked at "+formatTime(current.time)+" · Frame "+current.id,"result-meta"));
  if (status !== "answered" || body.image_plane_review) card.append(node("p",displayWords,"asked-words"));
  if (body.image_plane_review) card.append(node("p","The camera does not measure the physical distance to the landmark.","result-meta"));
  let text = answer.answer || (evidence.length ? "The search found the frames below. They do not establish an answer to this question." :
    "No frames support that question. Try a visible road user, a track number or an earlier time in this clip.");
  if (status === "error") text = evidence.length ? "The answer service could not finish. You can inspect the available frames or retry." : "The answer service could not finish. You can retry this question.";
  card.append(node("p",text,"answer-text"));
  if (body.vehicle_choices) {
    const group = node("div",undefined,"vehicle-choices");
    group.setAttribute("role","group"); group.setAttribute("aria-label","Vehicles in the paused frame");
    for (const option of body.vehicle_choices.options) {
      const label = className(option.class_name) + " · " + option.position + " · Track " + option.track_id;
      const button = action(label,async ()=>{
        if (!canAsk() || player.playing || player.displayed.id!==current.id || answerMoment?.key!==scene.value) return;
        chosenVehicle = {option, selection:{binding:body.vehicle_choices.binding,track_id:option.track_id}, words, displayWords, imageRelation:body.image_relation_available};
        for (const item of group.children) item.setAttribute("aria-pressed",String(item===button));
        get("vehicle-focus-note").textContent = "Outlined " + label + ". " + (body.image_relation_available
          ? "The camera does not measure the distance to the landmark. Gemini can describe their positions in this image."
          : "Recorded tracks show its position. Gemini can check your question against privacy-processed images.");
        get("review-selected").textContent = body.image_relation_available ? "Compare image positions with Gemini" : "Ask Gemini about this vehicle";
        get("review-selected").hidden = false; get("change-vehicle").hidden = false;
        await inspect(current.id,option.track_id,option.bbox_xyxy);
        if (chosenVehicle?.option===option) get("review-selected").focus({preventScroll:true});
      },"subtle");
      const crop=vehicleCrop(option);
      if (crop) button.prepend(crop);
      button.setAttribute("aria-pressed","false");
      group.append(button);
    }
    card.append(group);
  } else if (body.selected_vehicle) {
    const option = body.selected_vehicle;
    highlight = {key:scene.value,frame:current.id,track:option.track_id,rect:option.bbox_xyxy};
    card.append(node("p","Selected track "+option.track_id+" · "+className(option.class_name)+" · "+option.position,"result-meta"));
  }
  const actions = node("div",undefined,"answer-actions");
  if (status === "error") actions.append(action("Retry this search",()=>submitWords(words,displayWords,body.selection,body.image_plane_review)));
  if (body.image_relation_available && !body.vehicle_choices && !body.image_plane_review)
    actions.append(action("Compare image positions with Gemini",()=>{mode.value="gemini";updateMode();submitWords(words,displayWords,body.selection,true);}));
  if (body.can_answer_from_tracks) actions.append(action("Answer from recorded tracks",()=>{mode.value="local";updateMode();submitWords(body.question.question,displayWords);}));
  if (body.answer_source==="evidence_only" && evidence.length && /\b(colou?r|sign|lettering|look like|appearance|paint|wearing)\b/i.test(body.question.question)) {
    actions.append(action("Ask Gemini about these frames",()=>{mode.value="gemini";updateMode();submitWords(body.question.question,displayWords);}));
  }
  if (evidence.length) {
    const back = action("Return to the question frame",()=>void inspect(current.id));
    back.id = "return-question-frame"; back.hidden = true;
    actions.append(back);
  }
  card.append(actions);
  for (const item of evidence) addEvidence(card,item, (answer.cited_evidence_ids || []).includes(item.id));
  results.replaceChildren(card);
  results.focus({preventScroll:true});
  results.scrollIntoView({block:"start"});
}
editing(false);
try {
  const catalog = await fetchJson("/api/footage-catalog").catch(error=>{
    if(error.status === 404) return fetchJson("/api/scenes");
    throw error;
  });
  if (!catalog.footage_available) throw new Error("Footage search needs the original recordings and saved predictions.");
  const covered = new Set(catalog.footage_scenes);
  const frames = catalog.scenes.filter(f=>covered.has(f.key)).sort((a,b)=>a.recording_id.localeCompare(b.recording_id)||a.frame_id-b.frame_id);
  const params = new URLSearchParams(location.search), requested = frames.find(f=>f.key===params.get("scene"));
  const recordingId = params.get("recording") || requested?.recording_id;
  for(const f of frames) {
    bookmarks.set(f.recording_id+":"+f.frame_id,f.key);
    if ([...recordings.values()].some(row=>row.recording_id===f.recording_id)) continue;
    recordings.set(f.key,f);
    const opt = node("option",recordingName(f)); opt.value=f.key; scene.append(opt);
    if(f.recording_id===recordingId) scene.value=f.key;
  }
  if (!scene.value) throw new Error("No original recordings are attached.");
  const anchor = recordings.get(scene.value), requestedFrame = params.get("frame");
  const initial = requestedFrame!==null && /^\d+$/.test(requestedFrame) ? Number(requestedFrame) :
    requested?.recording_id===anchor.recording_id ? requested.frame_id : null;
  get("frame-picker").hidden = false;
  editing(true); updateMode();
  await selectScene(initial);
} catch(error) {
  get("progress").textContent = error.message;
}
scene.addEventListener("change",()=>void selectScene());
question.addEventListener("input",()=>{submittedHint=null;clearResult();syncButtons();});
mode.addEventListener("change",()=>{submittedHint=null;clearResult();updateMode();});
get("back-to-answer").onclick=async ()=>{
  if (answerMoment) await inspect(answerMoment.frame,chosenVehicle?.option.track_id,chosenVehicle?.option.bbox_xyxy);
  if (answerMoment) {
    get("back-to-answer").hidden=true;
    results.focus({preventScroll:true});
    results.scrollIntoView({block:"start"});
  }
};
get("review-selected").onclick=()=>{
  if (!chosenVehicle || !canAsk() || player.playing || player.displayed.id!==answerMoment?.frame) return;
  const {words,displayWords,selection,imageRelation}=chosenVehicle;
  mode.value="gemini"; updateMode(); submitWords(words,displayWords,selection,imageRelation);
};
get("change-vehicle").onclick=()=>{
  const selected=results.querySelector('.vehicle-choices button[aria-pressed="true"]');
  if (selected) {selected.scrollIntoView({block:"center"});selected.focus({preventScroll:true});}
};
get("clip-first-marker").onclick=async ()=>{
  const target = firstMarker;
  if (!target || busy || loading || player.pending) return;
  player.pause(false);
  highlight = {key:scene.value,frame:target.id,track:target.track,rect:target.rect};
  await player.display(target.index);
  if (player.displayed?.id===target.id) get("clip-play").focus({preventScroll:true});
};
get("open-frame").onclick = async () => {
  if (busy || loading) return;
  player.pause(false);
  const index = player.frames.findIndex(f=>f.id===get("frame-id").valueAsNumber);
  if(index<0) {get("progress").textContent="Enter a frame present in this recording."; return;}
  highlight=null; await player.display(index); get("frame-picker").open=false;
};
get("search").addEventListener("submit",async event=>{
  event.preventDefault();
  if (!canAsk() || !question.value.trim()) return;
  player.pause();
  const current=player.displayed, anchor=scene.value, hosted=mode.value==="gemini", displayWords=question.value.trim();
  const bound = submittedHint?.frame===current.id && submittedHint.scene===anchor && submittedHint.displayWords===displayWords;
  const words = bound ? submittedHint.words : displayWords;
  const selection = bound ? submittedHint.selection : null;
  const imageRelation = bound && submittedHint.imageRelation;
  const controller=new AbortController();
  searchRun=controller; clearResult(); editing(false);
  get("cancel-search").hidden=false;
  get("progress").textContent=hosted ? "Asking Gemini about the selected evidence…" : "Checking recorded tracks up to "+formatTime(current.time)+"…";
  try {
    const key=await keyForFrame(anchor,current.id);
    if (controller.signal.aborted) return;
    const body=await fetchJson("/api/footage",{method:"POST",headers:{"Content-Type":"application/json"},signal:controller.signal,
      body:JSON.stringify({scene:key,question:words,hosted,...(selection ? {selection} : {}),...(imageRelation ? {image_relation:true} : {})})});
    if (searchRun!==controller) return;
    if (body.question.query_frame!==current.id || body.question.recording_id!==recordings.get(anchor).recording_id ||
        body.evidence.some(e => e.interval.end_frame>current.id || e.images.some(im=>im.frame_id>current.id))) {
      throw new Error("The evidence did not match this recording and playhead. Choose the moment and search again.");
    }
    get("progress").textContent="";
    showAnswer(body,current,recordings.get(anchor),displayWords);
  } catch(error) {
    if (searchRun===controller && !controller.signal.aborted) get("progress").textContent=error.message;
  } finally {
    if (searchRun===controller) {searchRun=null; get("cancel-search").hidden=true; editing(true); updatePlayer();}
  }
});
get("cancel-search").onclick=()=>{
  if (!searchRun) return;
  searchRun.abort(); searchRun=null;
  get("cancel-search").hidden=true; editing(true);
  get("progress").textContent="Search cancelled. You can edit your question or play the recording.";
};
