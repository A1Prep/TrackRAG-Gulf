import {FootagePlayer} from "./footage-player.js";

const get = id => document.getElementById(id);
function node(tag, text) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  return item;
}
async function fetchJson(url, opts) {
  const response = await fetch(url,opts);
  const body = await response.json();
  if (!response.ok) {
    const error=new Error(body.error || "Could not search the recording.");
    error.status=response.status; throw error;
  }
  return body;
}
const scene = get("scene"), question = get("question"), mode = get("answer-mode"), results = get("results");
let busy = true, loading = true, turn = 0, shown = "", exampleKey = "", searchRun = null, subjects = {};
const recordings = new Map(), bookmarks = new Map();
const player = new FootagePlayer(updatePlayer);
function editing(enabled) {
  busy = !enabled;
  for (const control of [scene, question, mode, get("frame-id"), get("open-frame")]) control.disabled = !enabled;
  player.setLocked(!enabled);
  get("go").disabled = !enabled || !player.displayed;
  for (const button of get("examples").children) button.disabled = !enabled;
}
function clearResults() {
  results.replaceChildren();
  get("progress").textContent = "";
}
function examples(frame) {
  const subject=subjects[frame.id], clock=formatTime(frame.time);
  if (mode.value==="gemini") return [
    "What can you see on the road at "+clock+"?",
    ...(subject ? ["What does track "+subject.id+" look like in this frame?"] : [])];
  return ["Which types of road users are detected in this frame?",
    subject ? "Which side of the frame is track "+subject.id+" on?" : "How many vehicles are detected in this frame?"];
}
function formatTime(seconds) {
  const tenths=Math.round(seconds*10), sec=Math.floor(tenths/10)%60;
  return Math.floor(tenths/600)+":"+String(sec).padStart(2,"0")+(tenths%10 ? "."+tenths%10 : "");
}
function updatePlayer(writeAddress = true) {
  const row=player.displayed, recording=recordings.get(scene.value);
  get("go").disabled=busy || loading || !row || (player.pending && !player.playing);
  if (!row || !recording || !writeAddress) return;
  const key=recording.recording_id+":"+row.id;
  if (shown!==key) { shown=key; clearResults(); }
  get("frame-id").value=String(row.id);
  const hints=examples(row);
  if (exampleKey!==hints.join("|")) {
    exampleKey=hints.join("|"); get("examples").replaceChildren();
    for(const hint of hints) {
      const button=node("button",hint); button.type="button"; button.disabled=busy;
      button.onclick=()=>{question.value=hint; clearResults(); question.focus();};
      get("examples").append(button);
    }
  }
  if (player.playing) return;
  const url = new URL(location.href);
  url.searchParams.set("scene",scene.value); url.searchParams.set("recording",recording.recording_id);
  url.searchParams.set("frame",String(row.id));
  if (url.href!==location.href) history.replaceState(null,"",url);
  const saved=bookmarks.get(key);
  get("passenger-link").href=saved ? "/?scene="+encodeURIComponent(saved)
    : "/api/footage-preview/"+scene.value+"/dropoff/"+row.id;
}
async function selectScene(target) {
  const run=++turn, anchor=recordings.get(scene.value);
  loading=true; clearResults(); get("examples").replaceChildren(); exampleKey=shown=""; subjects={};
  player.reset(scene.value,anchor); get("go").disabled=true;
  void player.cover(target);
  try {
    const manifest=await fetchJson("/api/footage-preview/"+anchor.key+"/timeline");
    if (run!==turn) return;
    subjects=manifest.subjects || {};
    const available=manifest.frames.some(row=>row.id===target);
    await player.load(manifest.frames,available ? target : anchor.frame_id);
    if (run!==turn) return;
    loading=false; updatePlayer();
    if (!available && player.displayed) get("progress").textContent="Frame "+target+" is unavailable. Showing frame "+player.displayed.id+".";
  } catch(error) {
    if (run!==turn) return;
    loading=false; get("progress").textContent=error.message; get("go").disabled=true;
  }
}
async function keyForFrame(anchor, frame) {
  const label=recordings.get(anchor).recording_id+":"+frame;
  if (bookmarks.has(label)) return bookmarks.get(label);
  const bound=await fetchJson("/api/frames",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({scene:anchor,frame_id:frame,for_footage:true})});
  if (bound.frame_id!==frame || bound.recording_id!==recordings.get(anchor).recording_id)
    throw new Error("The recording changed. Select the displayed frame again.");
  bookmarks.set(label,bound.key); return bound.key;
}
editing(false);
try {
  const catalog = await fetchJson("/api/footage-catalog").catch(error=>{
    if(error.status===404) return fetchJson("/api/scenes");
    throw error;
  });
  if (!catalog.footage_available) throw new Error("Footage search needs the original recordings and saved predictions.");
  const covered = new Set(catalog.footage_scenes);
  const frames=catalog.scenes.filter(f=>covered.has(f.key)).sort((a,b)=>
    a.recording_id.localeCompare(b.recording_id) || a.frame_id-b.frame_id);
  const params=new URLSearchParams(location.search), requested=frames.find(f=>f.key===params.get("scene"));
  const recordingId=params.get("recording") || requested?.recording_id;
  for(const f of frames) {
    bookmarks.set(f.recording_id+":"+f.frame_id,f.key);
    if ([...recordings.values()].some(row=>row.recording_id===f.recording_id)) continue;
    recordings.set(f.key,f);
    const option=node("option","Recording "+f.recording_id.replace(/^video_/,""));
    option.value=f.key; scene.append(option);
    if(f.recording_id===recordingId) scene.value=f.key;
  }
  if (!scene.value) throw new Error("No original recordings are attached.");
  const anchor=recordings.get(scene.value);
  const requestedFrame=params.get("frame");
  const initial=requestedFrame!==null && /^\d+$/.test(requestedFrame) ? Number(requestedFrame)
    : requested?.recording_id===anchor.recording_id ? requested.frame_id : anchor.frame_id;
  get("frame-picker").hidden=false;
  editing(true);
  await selectScene(initial);
} catch (error) {
  get("progress").textContent = error.message;
}
scene.addEventListener("change", () => void selectScene(recordings.get(scene.value).frame_id));
question.addEventListener("input", clearResults);
mode.addEventListener("change", () => {clearResults(); updatePlayer();});
get("open-frame").onclick=async () => {
  if (busy || loading) return;
  player.pause(false);
  const id=get("frame-id").valueAsNumber, index=player.frames.findIndex(f=>f.id===id);
  if(index<0) {get("progress").textContent="Enter a frame present in this recording."; return;}
  await player.display(index); get("frame-picker").open=false;
};
get("search").addEventListener("submit",async event => {
  event.preventDefault();
  if (busy || loading || !player.displayed || (player.pending && !player.playing)) return;
  player.pause();
  const current=player.displayed, anchor=scene.value, hosted=mode.value==="gemini";
  const controller = new AbortController();
  searchRun=controller;
  clearResults();
  editing(false);
  get("cancel-search").hidden=false;
  get("progress").textContent = hosted ? "Finding frames and asking Gemini…" : "Searching the recorded tracks…";
  try {
    const sourceKey=await keyForFrame(anchor,current.id);
    if(controller.signal.aborted) return;
    const body = await fetchJson("/api/footage",{method:"POST",headers:{"Content-Type":"application/json"},
      signal:controller.signal, body:JSON.stringify({scene:sourceKey,question:question.value,hosted})});
    if(searchRun!==controller) return;
    if (body.question.query_frame!==current.id || body.question.recording_id!==recordings.get(anchor).recording_id)
      throw new Error("The search returned a different frame or recording.");
    const labels = {answered:hosted ? "Gemini answer" : "Answer from recorded tracks",
      error:"The search couldn’t finish",insufficient_evidence:"Insufficient evidence"};
    if (body.answer.answer_status !== "not_requested") {
      const failed=body.answer.answer_status==="error";
      const message=failed ? (body.evidence.length ? "You can review the retrieved frames below or retry this question." :
        "Retry the question or try different wording at this frame.") :
        body.answer.answer || "The available evidence does not establish an answer.";
      results.append(node("h2",labels[body.answer.answer_status]),node("p",message));
      if(failed) {
        const retry=node("button","Retry search"); retry.type="button"; retry.className="subtle";
        retry.onclick=()=>get("search").requestSubmit(); results.append(retry);
        if (body.can_answer_from_tracks) {
          const local=node("button","Answer from saved tracks"); local.type="button"; local.className="subtle";
          local.onclick=()=>{mode.value="local";updatePlayer();get("search").requestSubmit();};
          results.append(local);
        }
      }
    } else {
      results.append(node("h2",body.evidence.length ? "Retrieved evidence" : "No matching evidence"),
        node("p","For colours, signs and visual details, ask Gemini about the frames."));
      const ask=node("button","Ask Gemini about this question");
      ask.type="button"; ask.className="subtle";
      ask.onclick=()=>{mode.value="gemini";updatePlayer();get("search").requestSubmit();};
      results.append(ask);
    }
    const count = body.evidence.length;
    get("progress").textContent = scene.selectedOptions[0].textContent+" · Paused at "+formatTime(current.time)+" · Frame "+current.id+
      (count ? " · " + count + " evidence interval" + (count === 1 ? "" : "s") : " · No matching evidence");
    for (const item of body.evidence) {
      const article = node("article");
      article.className = "card review-form";
      const cited = body.answer.cited_evidence_ids.includes(item.id) ? " · cited in the answer" : "";
      article.append(node("h3","Evidence "+(body.evidence.indexOf(item)+1)+cited),
        node("p","Frames "+item.interval.start_frame+"–"+item.interval.end_frame));
      const strip = node("div");
      strip.className = "evidence-strip";
      for (const im of item.images) {
        const link = node("a");
        link.href = im.url;
        link.target = "_blank";
        link.rel = "noopener";
        link.setAttribute("aria-label", "Open evidence at frame " + im.frame_id + " in a new tab");
        const photo = node("img");
        photo.alt = "Evidence for frame " + im.frame_id;
        photo.src = im.url;
        const moment=player.frames.find(row=>row.id===im.frame_id);
        link.append(photo,node("span","Evidence image · Frame "+im.frame_id));
        const tile=node("div"); tile.append(link);
        if(moment) {
          const replay=node("a","View original at "+formatTime(moment.time));
          const url=new URL(location.href); url.searchParams.set("frame",String(moment.id));
          replay.href=url.href; replay.target="_blank"; replay.rel="noopener"; tile.append(replay);
        }
        strip.append(tile);
      }
      article.append(strip);
      results.append(article);
    }
  } catch (error) {
    if(searchRun===controller && !controller.signal.aborted) get("progress").textContent = error.message;
  } finally {
    if(searchRun===controller) {searchRun=null; get("cancel-search").hidden=true; editing(true);}
  }
});
get("cancel-search").onclick=()=>{
  if(!searchRun) return;
  searchRun.abort(); searchRun=null;
  get("cancel-search").hidden=true; editing(true);
  get("progress").textContent="Search cancelled. You can edit the question or replay the recording.";
};
