import {addFrame, openFrame} from "./frames.js";
import {noSound, openMicrophone} from "./capture.js";
import {VehiclePairPicker, drawChosenLandmarks} from "./pair-choices.js";
import {PhotoTools, destinationLabel, drawDestination, positionWords} from "./photo-tools.js";

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
  if (!point) return "Choose a location on the photo.";
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
  if (capture) {
    const source = capture.mic.label ? "Recording with " + capture.mic.label + ". " : "The input level should move as you speak. ";
    return ["listening", "Listening…", source + "Press Stop recording to review your words."];
  }
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
function legend(v, offerMark) {
  const node = el("legend");
  const pin = v.result?.destination_point_xy && photoState === "ready" && !editing && !session?.voice_stage;
  node.dataset.kind = !photo.active && pin && !pairs.open ? (v.selected_by === "passenger" ? "manual" : "auto") : "none";
  let label = "Original photo";
  if (photoState === "loading") label = "Opening photo…";
  else if (photoState === "failed") label = "Photo unavailable";
  else if (pairs.open) label = "Numbers mark selectable vehicles";
  else if (photo.active) label = "Your mark · preview";
  else if (pin) {
    label = destinationLabel(v);
    if (v.state === "confirmed") label += " · confirmed";
    if (v.result.anchor_boxes_xyxy?.length === 2 && ["parking_model","passenger_landmarks"].includes(v.selected_by)) label += " · A and B mark the vehicles";
  } else if (!editing && !session?.voice_stage && v.area_landmarks) label = "A and B mark the vehicles";
  if (offerMark && !photo.zoom) label = v.area_landmarks ? "A and B mark the vehicles · Tap the photo to mark your spot" : "Tap the photo to mark your spot";
  text(node, label);
  return label;
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
  el("mic-choice").hidden = !catalog?.speech_available || el("mic-input").options.length < 2;
  el("mic-input").disabled = locked || !opened;
  el("mic-monitor").hidden = !micOn;
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
  el("submit-area").hidden = !formOpen || !v.can_request_area || !input.value.trim() || Boolean(stage && stage !== "review");
  el("submit-area").disabled = input.disabled || !ready || staleEdit();
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
  const offerMark = decision && ready && !locked && ["needs_clarification","insufficient_visual_evidence"].includes(v.state);
  photo.controls(ready, locked, legend(v, offerMark), offerMark);
  paint();
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
  el("footage-link").href = catalog.searchableScenes.has(f.key) ? "/footage?scene=" + encodeURIComponent(f.key) : "/footage";
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
  if (!catalog.scenes.some(s => s.key === f.key)) {
    catalog.scenes.push(f);
    // Additional frames come from the attached recording and its verified predictions.
    catalog.searchableScenes.add(f.key);
  }
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
  if (photo.active) {
    if (!photo.box) { photo.drawCursor(ctx,scale); return; }
    const [x1,y1,x2,y2] = photo.box;
    drawDestination(ctx, [(x1+x2)/2,(y1+y2)/2], photo.box, "mark_preview", scale);
    return;
  }
  if (!v.result?.destination_point_xy) return;
  if (["passenger_landmarks","parking_model"].includes(v.selected_by)) drawChosenLandmarks(ctx, v.result.anchor_boxes_xyxy, scale);
  drawDestination(ctx, v.result.destination_point_xy, v.result.destination_box_xyxy, v.selected_by, scale);
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
function sendInstruction(choice = "automatic") {
  if (!input.value.trim() || photoState !== "ready" || staleEdit()) return;
  const payload = {text:input.value.trim(), mode:session.voice_stage === "review" ? "voice" : "typed"};
  if (editing === "replace") payload.replaces_request_id = editTarget;
  if (choice === "vehicles") payload.point_to_vehicles = true;
  if (choice === "area") payload.mark_on_photo = true;
  perform(async () => {
    await action("submit",payload); input.value=""; editing=false; previousDraft=null; resetSelection();
    if (choice === "vehicles" && session.view.pair_choices?.length >= 2) pairs.begin(session.view.pair_choices);
    if (choice === "area" && session.view.state === "needs_clarification" && session.view.selected_by === "passenger") photo.begin();
  }, choice === "area" ? "Opening the photo to mark…" : choice === "vehicles" ? "Opening the vehicle choices…" : "Finding your spot…",
  () => photo.active ? photo.focusPhoto() : pairs.open ? el("pair-first").focus() : focusResult());
}
el("form").onsubmit = e => { e.preventDefault(); sendInstruction(); };
el("submit-pointing").onclick = () => sendInstruction("vehicles");
el("submit-area").onclick = () => sendInstruction("area");
input.addEventListener("input", render);

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
async function refreshMicrophones() {
  if (!catalog?.speech_available || !navigator.mediaDevices?.enumerateDevices) return;
  // The browser can hide device names until the passenger grants microphone access.
  const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
  const select = el("mic-input"), chosen = select.value;
  const inputs = devices.filter(d => d.kind === "audioinput" && d.label && d.deviceId && !["default", "communications"].includes(d.deviceId));
  const systemInput = devices.find(d => d.kind === "audioinput" && d.deviceId === "default");
  select.replaceChildren(new Option(systemInput?.label || "System default", ""), ...inputs.map(d => new Option(d.label, d.deviceId)));
  if (inputs.some(d => d.deviceId === chosen)) select.value = chosen;
  el("mic-choice").hidden = !inputs.length;
}
async function startMic() {
  const meter = el("mic-level");
  meter.value = 0;
  const mic = await openMicrophone(el("mic-input").value, level => { meter.value = level; });
  try {
    await beginAudio();
    capture = {mic, timer: setTimeout(stopMic, 55000)};
    void refreshMicrophones();
  } catch (error) { mic.close(); throw error; }
}
async function stopMic() {
  if (!capture || busy) return;
  const {mic, timer} = capture;
  capture = null; clearTimeout(timer);
  await perform(async () => {
    let recording;
    try { recording = await mic.wav(); }
    catch (error) { input.value = previousDraft?.text || ""; throw error; }
    finally { mic.close(); }
    await sendAudio(recording.wav);
    if (session.speech_error?.code === "no_speech") {
      failure = recording.silent ? noSound : "The microphone picked up sound, but no words were recognized. Check the input level and record again.";
    }
  }, "Transcribing your voice…", () => input.focus());
}
el("mic").onclick = () => { if (capture) stopMic(); else perform(startMic,"Opening the microphone…"); };
navigator.mediaDevices?.addEventListener?.("devicechange", refreshMicrophones);
window.addEventListener("pagehide", () => { if (capture) { clearTimeout(capture.timer); capture.mic.close(); capture=null; } });
perform(async () => {
  catalog=await api("/api/scenes");
  const covered = new Set(catalog.footage_scenes);
  catalog.searchableScenes = covered;
  catalog.footageRecordings = new Set(catalog.scenes.filter(f => covered.has(f.key)).map(f => f.recording_id));
  el("footage-link").hidden=!catalog.footage_available;
  for (const f of catalog.scenes) addFrame(sceneSelect,f);
  const requested = new URLSearchParams(window.location.search).get("scene");
  if (catalog.scenes.some(f => f.key === requested)) sceneSelect.value = requested;
  await chooseScene();
  void refreshMicrophones();
}, "Opening the photo…");
