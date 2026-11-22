import {addFrame, openFrame} from "./frames.js";
import {VehiclePairPicker, drawChosenLandmarks} from "./pair-choices.js";
import {PhotoTools, positionWords} from "./photo-tools.js";

const el = id => document.getElementById(id);
const sceneSelect = el("scene"), image = el("frame"), canvas = el("overlay"), input = el("instruction");
let catalog, session, busy = false, capture = null, editing = false, previousDraft = null;
let editTarget = null, retired = false;
let photoState = "loading", retry = 0, notice = "", failure = "", historyRevision = "";
const pairs = new VehiclePairPicker(render);
const photo = new PhotoTools(canvas, el("photo-viewport"), render, () => busy || photoState !== "ready", pairs);
const terminal = state => ["confirmed", "cancelled"].includes(state);
const lastUser = turns => (turns || []).filter(t => t.role === "user").at(-1);
function text(node, value = "") {
  if (node.textContent !== value) node.textContent = value;
}
async function api(path, options) {
  const res = await fetch(path, options), data = await res.json();
  if (!res.ok) {
    const error = new Error(data.error || "The request could not be completed.");
    error.status = res.status; throw error;
  }
  return data;
}
function post(path, body, extra = {}) {
  return api(path, {method:"POST", headers:{"Content-Type":"application/json", ...extra}, body:JSON.stringify(body)});
}
function endpoint(action) {
  return "/api/sessions/" + session.session + (action ? "/" + action : "");
}
async function action(name, body = {}) {
  session = await post(endpoint(name), body, {"X-Session-Revision":String(session.revision)});
}
function activeScene() { return catalog?.scenes.find(s => s.key === session?.scene); }
function resetSelection() { photo.reset(); pairs.reset(); }
function staleEdit() {
  return editing && (editTarget !== session?.view.request_id || (editing !== "new" && terminal(session?.view.state)));
}
function focus(id) { el(id).focus({preventScroll:true}); }
function focusResult() {
  const node = el("destination-title");
  node.focus({preventScroll:true});
  const bounds = node.getBoundingClientRect();
  if (bounds.top < 0 || bounds.bottom > innerHeight) node.scrollIntoView({block:"nearest"});
}
function history(v) {
  const revision = session?.session + ":" + session?.revision;
  if (revision === historyRevision) return;
  historyRevision = revision;
  const messages = el("messages"); messages.replaceChildren();
  for (const turn of v.turns || []) {
    const bubble = document.createElement("p"), label = document.createElement("b");
    bubble.className = "bubble " + (turn.role === "user" ? "user" : "assistant");
    text(label, turn.role === "user" ? "You" : "Passenger assistant");
    bubble.append(label, document.createTextNode(turn.text)); messages.append(bubble);
  }
  el("history").hidden = !v.turns?.length;
}
function canMark(v) {
  return ["proposed","needs_clarification","insufficient_visual_evidence"].includes(v.state)
    || Boolean(v.landmark_error_guidance || v.parking_recovery);
}
function suggestion(v) {
  const point = v.result?.destination_point_xy;
  if (!point) return "Check the outlined area on the photo.";
  const where = positionWords(point, canvas.width, canvas.height);
  if (v.selected_by === "passenger") return "The road check placed the blue pin in the " + where + " of the photo, inside your mark.";
  if (v.selected_by === "passenger_landmarks") return "The yellow pin in the " + where + " uses the two vehicles you chose.";
  return "Check the yellow pin in the " + where + " of the photo.";
}
function response(v, stage) {
  if (photoState === "failed") return ["error", retired ? "Reopen this scene" : "Reload the road photo",
    retired ? "This session ended. Reopen the scene to start a new request." : "Open the photo again to check a location."];
  if (busy && notice) return ["working", notice, ""];
  if (!session) return ["idle", "Where would you like to get out?", ""];
  if (staleEdit()) return ["error", "This request has changed", "Your draft is below. Go back to the current request to edit it again."];
  if (capture) return ["listening", "Listening…", "Stop recording when you have finished speaking."];
  if (stage === "review") return ["review", "Did I hear you correctly?", "Edit any words below, then send the transcript."];
  if (stage === "failed" || stage === "recording") return ["clarify", "Try voice again or type", "Your description can also be sent as text."];
  if (photo.active) return ["mark", "Choose a spot on the photo", "Check your mark to see where the pin lands."];
  if (pairs.open) return ["mark", "Which vehicles do you mean?", "Choose two vehicles and a position in the photo controls."];
  if (editing === "reply") return ["clarify", "Tell me more", v.result?.clarification_question || v.turns?.filter(t=>t.role==="assistant").at(-1)?.text || ""];
  if (editing) return ["idle", editing === "new" ? "Where would you like to get out?" : "Edit your description", ""];
  if (v.state === "proposed") return ["proposed", "Is this your spot?", suggestion(v)];
  if (v.state === "confirmed") {
    const point = v.result?.destination_point_xy;
    const location = point ? " in the " + positionWords(point, canvas.width, canvas.height) + " of the photo" : "";
    return ["confirmed", "Spot confirmed", "You confirmed the " + (v.selected_by === "passenger" ? "blue" : "yellow") + " pin" + location + "."];
  }
  if (v.state === "cancelled") return ["idle", "Request cancelled", "You can start again on this photo."];
  if (v.state === "error") return ["error", "The request couldn’t finish", v.landmark_error_guidance ||
    (v.parking_recovery ? "You can still mark the intended spot on the photo or edit your description." : "Try the description again or choose another scene.")];
  if (canMark(v)) return ["clarify", "Which spot do you mean?", v.result?.clarification_question ||
    v.turns?.filter(t => t.role === "assistant").at(-1)?.text || "Mark the location on the photo or edit your description."];
  return ["idle", "Where would you like to get out?", ""];
}
function legend(v) {
  const node = el("legend");
  const area = Boolean(v.result?.destination_box_xyxy) && photoState === "ready" && !editing && !session?.voice_stage;
  node.dataset.kind = !photo.active && area && !pairs.open ? (v.selected_by === "passenger" ? "manual" : "auto") : "none";
  let label = "Original photo";
  if (photoState === "loading") label = "Opening photo…";
  else if (photoState === "failed") label = "Photo unavailable";
  else if (pairs.open) label = "Numbers mark selectable vehicles";
  else if (photo.active) label = "Your mark · preview";
  else if (area) {
    label = v.selected_by === "passenger" ? "Your marked spot" : v.selected_by === "passenger_landmarks" ? "Spot from your vehicles" : "Suggested spot";
    if (v.state === "confirmed") label += " · confirmed";
    if (v.result.anchor_boxes_xyxy?.length === 2 && ["parking_model","passenger_landmarks"].includes(v.selected_by)) label += " · A and B mark the vehicles";
  } else if (!editing && !session?.voice_stage && v.area_landmarks) label = "A and B mark the vehicles";
  text(node, label);
}
function render() {
  const v = session?.view || {}, stage = session?.voice_stage, current = activeScene();
  const opened = Boolean(session), ready = photoState === "ready", micOn = Boolean(capture);
  const marking = photo.active || pairs.open, locked = busy || micOn, prior = lastUser(v.turns);
  const formOpen = !marking && (editing || Boolean(stage) || !v.state || v.state === "idle" || (v.state === "error" && !canMark(v)));
  const decision = opened && !formOpen && !marking && !terminal(v.state);
  const [kind, heading, status] = response(v, stage);
  el("destination-panel").dataset.state = kind;
  text(el("destination-title"), heading); text(el("status"), status);
  const err = failure || session?.speech_error?.message || "";
  text(el("error"), err); el("error").hidden = !err;
  el("last-request").hidden = !prior || formOpen;
  if (prior) text(el("last-request").querySelector("p"), prior.text);
  history(v);

  sceneSelect.disabled = locked || !opened;
  el("prev-scene").disabled = sceneSelect.disabled || sceneSelect.selectedIndex <= 0;
  el("next-scene").disabled = sceneSelect.disabled || sceneSelect.selectedIndex >= sceneSelect.options.length-1;
  el("frame-picker").hidden = !current || !catalog.footageRecordings.has(current.recording_id);
  el("open-frame").disabled = sceneSelect.disabled;
  el("photo-unavailable").hidden = photoState !== "failed";
  el("retry-photo").disabled = locked;
  text(el("retry-photo"), retired ? "Reopen scene" : "Reload photo");
  el("form").hidden = !formOpen;
  el("speech-controls").hidden = !catalog?.speech_available;
  el("mic").disabled = busy || !opened || !ready || marking || (staleEdit() && !micOn);
  el("upload").disabled = locked || !opened || !ready || marking || staleEdit();
  text(el("mic"), micOn ? "Stop recording" : "Record voice");
  input.disabled = locked || !opened || marking;
  el("submit").disabled = input.disabled || !ready || !input.value.trim() || staleEdit();
  text(el("submit"), stage === "review" ? "Send reviewed transcript" : editing === "replace" ? "Send new description" : editing === "reply" ? "Send reply" : "Find my spot");
  text(el("input-label"), stage === "review" ? "Your transcript" : editing === "reply" ? "Your reply" : "Your description");
  el("edit-original").hidden = editing !== "reply" || Boolean(stage) || Boolean(input.value.trim());
  el("edit-original").disabled = locked;
  el("submit-pointing").hidden = !formOpen || !v.can_request_pointing || !input.value.trim() || Boolean(stage && stage !== "review");
  el("submit-pointing").disabled = input.disabled || !ready || staleEdit();
  el("discard-edit").hidden = !formOpen || (!editing && !stage) || micOn;
  el("discard-edit").disabled = locked;
  text(el("discard-edit"), staleEdit() ? "Back to current request" : stage ? "Discard recording" : "Back without changing");

  el("confirm").hidden = !(decision && v.state === "proposed" && v.proposal_id);
  el("confirm").disabled = locked || !ready;
  el("draw").hidden = !(decision && canMark(v));
  el("draw").disabled = locked || !ready;
  el("draw").className = v.state === "proposed" ? "subtle" : "primary";
  text(el("draw"), v.state === "proposed" ? "Change spot on photo" : "Mark my spot on the photo");
  el("edit-words").hidden = !decision || !prior;
  el("edit-words").disabled = locked;
  text(el("edit-words"), v.state === "needs_clarification" || v.state === "insufficient_visual_evidence" ? "Answer in words" : "Edit description");
  el("choose-pair").hidden = !decision || (v.pair_choices?.length || 0) < 2;
  el("choose-pair").disabled = locked || !ready;
  el("new-request").hidden = !terminal(v.state) || formOpen;
  el("new-request").disabled = locked || !ready;
  el("cancel").hidden = !opened || (!stage && (!v.state || v.state === "idle" || terminal(v.state))) || marking;
  el("cancel").disabled = locked;
  pairs.controls(locked || !ready);
  photo.controls(ready, locked);
  legend(v); paint();
}
async function perform(work, message = "", after) {
  if (busy) return;
  busy = true; failure = ""; notice = message; render();
  let success = false;
  try { await work(); success = true; }
  catch (error) {
    failure = error.message || "The action could not be completed.";
    if (session) {
      try { session = await api(endpoint()); sceneSelect.value = session.scene; }
      catch (syncError) { if (syncError.status === 404) { retired=true; photoState="failed"; resetSelection(); } }
    }
  } finally { busy = false; notice = ""; render(); }
  if (success && after) after();
}
function loadPhoto() {
  photoState = "loading"; image.src = endpoint("image") + (retry ? "?retry=" + retry : "");
}
async function chooseScene() {
  const newSession = await post("/api/sessions", {scene:sceneSelect.value, previous:retired ? undefined : session?.session});
  session = newSession; input.value = ""; editing = false; previousDraft = editTarget = null; retired = false; retry = 0;
  resetSelection(); photo.fit(); el("frame-picker").open = el("history").open = false;
  const f = activeScene();
  canvas.width = f.width; canvas.height = f.height;
  image.parentElement.style.aspectRatio = el("photo-viewport").style.aspectRatio = f.width + " / " + f.height;
  image.alt = "Road scene from " + sceneSelect.selectedOptions[0].textContent;
  const url = new URL(location.href); url.searchParams.set("scene", f.key); window.history.replaceState(null, "", url);
  loadPhoto();
}
function stepScene(amount) {
  if (sceneSelect.disabled) return;
  sceneSelect.selectedIndex += amount; perform(chooseScene, "Opening the photo…");
}
el("prev-scene").onclick = () => stepScene(-1);
el("next-scene").onclick = () => stepScene(1);
sceneSelect.onchange = () => perform(chooseScene, "Opening the photo…");
el("open-frame").onclick = () => perform(async () => {
  const f = await openFrame(sceneSelect,el("frame-id"));
  if (!catalog.scenes.some(s => s.key === f.key)) catalog.scenes.push(f);
  await chooseScene();
}, "Opening the frame…");
function currentImage() { return session && image.currentSrc && new URL(image.currentSrc).pathname === endpoint("image"); }
image.addEventListener("load", () => {
  if (!currentImage()) return;
  const f = activeScene(); photoState = image.naturalWidth === f.width && image.naturalHeight === f.height ? "ready" : "failed";
  render();
});
image.addEventListener("error", () => { if (currentImage()) { photoState = "failed"; resetSelection(); render(); } });
el("retry-photo").onclick = () => {
  if (retired) perform(chooseScene, "Reopening the scene…");
  else { retry++; loadPhoto(); render(); }
};
window.addEventListener("resize", () => paint());

function paint() {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (photoState !== "ready" || !canvas.clientWidth) return;
  const v = session.view, scale = photo.frameRatio();
  if (editing || session.voice_stage) return;
  if (pairs.open) { pairs.paint(ctx,scale); return; }
  if (v.area_landmarks) drawChosenLandmarks(ctx, v.area_landmarks.boxes_xyxy, scale);
  const box = photo.active ? photo.box : v.result?.destination_box_xyxy;
  if (!box) { photo.drawCursor(ctx,scale); return; }
  if (!photo.active && ["passenger_landmarks","parking_model"].includes(v.selected_by))
    drawChosenLandmarks(ctx, v.result.anchor_boxes_xyxy, scale);
  const [x1,y1,x2,y2] = box, passenger = photo.active || v.selected_by === "passenger";
  const color = passenger ? "#66eff1" : "#ffd34f";
  ctx.save();
  ctx.fillStyle = passenger ? "rgba(61,220,226,.15)" : "rgba(255,211,79,.10)";
  ctx.fillRect(x1,y1,x2-x1,y2-y1);
  if (!passenger || photo.active) ctx.setLineDash([7*scale,4*scale]);
  for (const [stroke,width] of [["#142a25",4.5],[color,2.5]]) {
    ctx.lineWidth=width*scale; ctx.strokeStyle=stroke; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  }
  const point = photo.active ? [(x1+x2)/2,(y1+y2)/2] : v.result.destination_point_xy;
  if (point) {
    ctx.setLineDash([]); ctx.translate(point[0],point[1]); ctx.scale(scale,scale);
    // The centre stays on the exact model or passenger point at every zoom.
    ctx.beginPath(); ctx.arc(0,0,10.5,0,2*Math.PI);
    ctx.lineWidth=3.6; ctx.strokeStyle="#16352d"; ctx.stroke();
    ctx.lineWidth=1.8; ctx.strokeStyle="#ffffff"; ctx.stroke();
    ctx.beginPath(); ctx.arc(0,0,6,0,2*Math.PI);
    ctx.fillStyle=color; ctx.fill(); ctx.lineWidth=1.8; ctx.strokeStyle="#16352d"; ctx.stroke();
  }
  ctx.restore();
}
el("zoom-in").onclick = () => photo.setZoom(1, photo.active ? photo.cursor : session?.view.result?.destination_point_xy);
el("zoom-out").onclick = () => photo.setZoom(-1, photo.active ? photo.cursor : session?.view.result?.destination_point_xy);
el("pan").onclick = () => {
  photo.panning = !photo.panning; render();
  if (!photo.panning) photo.focusPhoto();
};
el("draw").onclick = () => photo.begin(session?.view.result?.destination_point_xy);
el("discard-box").onclick = () => { photo.reset(); render(); focus("draw"); };
el("use-box").onclick = () => {
  if (photo.box && photoState === "ready") perform(async () => {
    await action("area", {box:photo.box}); photo.reset();
  }, "Checking your marked spot…", focusResult);
};
el("choose-pair").onclick = () => { pairs.begin(session.view.pair_choices); el("pair-first").focus(); };
el("discard-pair").onclick = () => { pairs.reset(); render(); focus("choose-pair"); };
el("use-pair").onclick = () => {
  if (!pairs.hasPair() || photoState !== "ready") return;
  const request = {track_ids:pairs.values(), placement:pairs.position.value};
  perform(async () => { await action("select-pair",request); pairs.reset(); }, "Checking the road beside those vehicles…", focusResult);
};
el("confirm").onclick = () => {
  if (photoState !== "ready" || editing || session.voice_stage || !session.view.proposal_id) return;
  const id = session.view.proposal_id;
  perform(() => action("confirm",{proposal_id:id}), "Saving your choice…", focusResult);
};
el("cancel").onclick = () => perform(async () => {
  await action("cancel"); input.value=""; resetSelection(); previousDraft=null; editing=false;
}, "Cancelling the request…", focusResult);
function startEditing(mode, words) {
  editing=mode; editTarget=session.view.request_id; input.value=words; failure=""; render(); input.focus();
}
function editOriginal() { startEditing("replace", session.view.turns.find(turn=>turn.role==="user")?.text || ""); }
el("edit-words").onclick = () => ["needs_clarification","insufficient_visual_evidence"].includes(session.view.state)
  ? startEditing("reply","") : editOriginal();
el("edit-original").onclick = editOriginal;
el("new-request").onclick = () => startEditing("new","");
el("discard-edit").onclick = () => {
  if (session.voice_stage && !staleEdit()) {
    perform(async () => {
      await action("discard-audio"); input.value = previousDraft?.text || "";
      editing = previousDraft?.editing || false; editTarget = previousDraft?.target ?? null; previousDraft = null;
    }, "Discarding the recording…", () => el("form").hidden ? focusResult() : input.focus());
  } else { input.value=""; editing=false; previousDraft=editTarget=null; render(); focusResult(); }
};
function sendInstruction(pointing) {
  if (!input.value.trim() || photoState !== "ready" || staleEdit()) return;
  const payload = {text:input.value.trim(), mode:session.voice_stage === "review" ? "voice" : "typed"};
  if (editing === "replace") payload.replaces_request_id = editTarget;
  if (pointing) payload.point_to_vehicles = true;
  perform(async () => {
    await action("submit",payload); input.value=""; editing=false; previousDraft=null; resetSelection();
    if (pointing && session.view.pair_choices?.length >= 2) pairs.begin(session.view.pair_choices);
  }, pointing ? "Opening the vehicle choices…" : "Finding your spot…", () => pairs.open ? el("pair-first").focus() : focusResult());
}
el("form").onsubmit = e => { e.preventDefault(); sendInstruction(false); };
el("submit-pointing").onclick = () => sendInstruction(true);
input.addEventListener("input", render);

function encodeWav(chunks, rate) {
  const n = chunks.reduce((v,a) => v + a.length, 0), buffer = new ArrayBuffer(44 + 2*n), d = new DataView(buffer);
  function ascii(offset, word) { for (let i=0;i<word.length;i++) d.setUint8(offset+i, word.charCodeAt(i)); }
  ascii(0,"RIFF"); d.setUint32(4,36+2*n,true); ascii(8,"WAVE"); ascii(12,"fmt ");
  d.setUint32(16,16,true); d.setUint16(20,1,true); d.setUint16(22,1,true); d.setUint32(24,rate,true);
  d.setUint32(28,2*rate,true); d.setUint16(32,2,true); d.setUint16(34,16,true); ascii(36,"data"); d.setUint32(40,2*n,true);
  let offset=44;
  for (const chunk of chunks) for (const sample of chunk) {
    const v=Math.max(-1,Math.min(1,sample)); d.setInt16(offset,Math.round(v*(v<0 ? 32768 : 32767)),true); offset+=2;
  }
  return new Blob([buffer], {type:"audio/wav"});
}
function checkAudio(blob) {
  if (!blob.size || blob.size > 12*1024*1024) throw new Error("Choose a nonempty WAV up to 12 MiB.");
}
async function beginAudio() {
  const draft = previousDraft || {text:input.value, editing, target:editTarget};
  await action("begin-audio");
  previousDraft = draft; resetSelection();
}
async function sendAudio(blob) {
  checkAudio(blob);
  session = await api(endpoint("audio"), {method:"POST", headers:{"Content-Type":"audio/wav",
    "X-Session-Revision":String(session.revision)}, body:blob});
  input.value = session.voice_stage === "review" ? session.transcript : (previousDraft?.text || "");
}
el("upload").onclick = () => el("file").click();
el("file").onchange = () => {
  const file = el("file").files[0]; el("file").value="";
  if (file) perform(async () => { checkAudio(file); await beginAudio(); await sendAudio(file); }, "Transcribing the audio…", () => input.focus());
};
async function startMic() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode)
    throw new Error("The microphone is unavailable. You can type or upload a WAV file.");
  let stream, context, expired=false, timer;
  try {
    const permission = navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true}});
    permission.then(s => { if (expired) s.getTracks().forEach(t => t.stop()); }, () => {});
    stream = await Promise.race([permission,new Promise((_,reject) => {
      timer=setTimeout(() => { expired=true; reject(new Error("Microphone permission timed out. You can type or upload a WAV file.")); }, 20000);
    })]);
    clearTimeout(timer);
    context = new AudioContext({sampleRate:16000}); await context.audioWorklet.addModule("/capture.js");
    const source = context.createMediaStreamSource(stream), node = new AudioWorkletNode(context,"passenger-capture");
    const gain = context.createGain(); gain.gain.value=0;
    const blocks=[]; node.port.onmessage = e => { blocks.push(e.data); };
    source.connect(node); node.connect(gain); gain.connect(context.destination); await context.resume();
    // Keep the draft and the active proposal until microphone setup has actually succeeded.
    await beginAudio();
    capture={stream,context,source,node,blocks,rate:context.sampleRate,timer:setTimeout(stopMic,55000)};
  } catch(error) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (context) await context.close();
    if (error.name === "NotAllowedError") throw new Error("Microphone access was denied. You can type or upload a WAV file.");
    throw error;
  } finally { clearTimeout(timer); }
}
async function stopMic() {
  if (!capture || busy) return;
  const c=capture; capture=null;
  await perform(async () => {
    clearTimeout(c.timer); c.stream.getTracks().forEach(t => t.stop()); c.source.disconnect(); c.node.disconnect();
    await c.context.close(); await sendAudio(encodeWav(c.blocks,c.rate));
  }, "Transcribing your voice…", () => input.focus());
}
el("mic").onclick = () => { if (capture) stopMic(); else perform(startMic,"Opening the microphone…"); };
window.addEventListener("pagehide", () => { if (capture) capture.stream.getTracks().forEach(t => t.stop()); });
perform(async () => {
  catalog=await api("/api/scenes");
  const covered = new Set(catalog.footage_scenes);
  catalog.footageRecordings = new Set(catalog.scenes.filter(f => covered.has(f.key)).map(f => f.recording_id));
  el("footage-link").hidden=!catalog.footage_available;
  for (const f of catalog.scenes) addFrame(sceneSelect,f);
  const requested = new URLSearchParams(window.location.search).get("scene");
  if (catalog.scenes.some(f => f.key === requested)) sceneSelect.value = requested;
  await chooseScene();
}, "Opening the photo…");
