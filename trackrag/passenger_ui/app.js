import {addFrame, openFrame} from "./frames.js";
import {noSound, openMicrophone} from "./capture.js";
import {VehiclePairPicker, drawChosenLandmarks, drawVehicleReferences} from "./pair-choices.js";
import {PhotoTools, bayAtPoint, bayChoices, bayName, boxCentre, destinationLabel, drawBays,
  drawCheckedMark, drawDestination, offeredBay, positionWords} from "./photo-tools.js";

const el = id => document.getElementById(id);
const sceneSelect = el("scene"), image = el("frame"), canvas = el("overlay"), input = el("instruction");
let catalog, session, busy = false, capture = null, editing = false, previousDraft = null;
let editTarget = null, retired = false;
let photoState = "loading", retry = 0, notice = "", failure = "", historyRevision = "";
let submitting = null, groundingPoll = null;
let checkedMark = null, bayLabelBoxes = [];
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
function post(path, body, extra = {}, signal) {
  return api(path, {method:"POST", headers:{"Content-Type":"application/json", ...extra}, body:JSON.stringify(body), signal});
}
function endpoint(action) {
  return "/api/sessions/" + session.session + (action ? "/" + action : "");
}
async function postAction(name, body, signal, actionId=crypto.randomUUID()) {
  const sid=session.session, revision=session.revision;
  try {
    return await post(endpoint(name),body,{"X-Session-Revision":String(revision),"X-Action-ID":actionId},signal);
  } catch(error) {
    if(signal?.aborted || (error.status && error.status<500)) throw error;
    // A reply can be lost after the server applied the tap or confirmation. Read the
    // receipt for that exact action; submitting a second tap could change the choice.
    for(const delay of [0,350,900]) {
      if(delay) await new Promise(resolve=>setTimeout(resolve,delay));
      if(session?.session!==sid || signal?.aborted) throw error;
      let saved;
      try {saved=await api("/api/sessions/"+sid,{signal:signal || AbortSignal.timeout(3500)});}
      catch(_) {continue;}
      if(saved.last_action_id===actionId && saved.revision>revision) return saved;
      if(saved.revision>revision) break;
    }
    throw error;
  }
}
async function action(name, body = {}) {session=await postAction(name,body);}
function stopPolling() {
  if (!groundingPoll) return;
  clearTimeout(groundingPoll.timer);
  groundingPoll.controller.abort(); groundingPoll = null;
}
function watchGrounding() {
  if (!session?.pending_grounding || busy || retired) { stopPolling(); return; }
  if (groundingPoll) return;
  const watch = {sid:session.session, controller:new AbortController(), timer:null};
  groundingPoll = watch;
  async function refresh() {
    let delay = 500;
    try {
      const next = await api("/api/sessions/" + watch.sid,{signal:watch.controller.signal});
      if (watch !== groundingPoll || next.session !== session?.session || next.revision < session.revision) return;
      session = next; failure = ""; render();
    } catch (error) {
      if (watch !== groundingPoll || watch.controller.signal.aborted) return;
      if (error.status === 404) { retired=true; photoState="failed"; resetSelection(); }
      else { failure="The connection was interrupted. Checking your request again…"; delay=1500; }
      render();
    } finally {
      if (watch === groundingPoll) watch.timer = setTimeout(refresh,delay);
    }
  }
  watch.timer = setTimeout(refresh,500);
}
function activeScene() { return catalog?.scenes.find(s => s.key === session?.scene); }
function resetSelection() { photo.reset(); pairs.reset(); checkedMark=null; }
function currentMark(v) {
  return checkedMark?.session===session?.session && checkedMark?.request===v.request_id && v.selected_by==="passenger" ? checkedMark : null;
}
function visibleChoices(v) {
  if (photoState !== "ready" || photo.active || pairs.open || editing || submitting || session?.voice_stage || session?.pending_grounding) return [];
  return bayChoices(v,canvas.width,canvas.height);
}
function tapName(v) { return currentMark(v)?.dragged ? "Your mark" : "Your tap"; }
function staleEdit() {
  return editing && (editTarget !== session?.view.request_id || (editing !== "new" && terminal(session?.view.state)));
}
function focus(id) { el(id).focus({preventScroll:true}); }
function focusResult() {
  const node = el("bay-tools").hidden ? el("destination-title") : el("bay-tools");
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
function suggestion(v, bay) {
  const point = v.result?.destination_point_xy;
  if (!point) return "Choose a location on the photo.";
  const where = positionWords(point, canvas.width, canvas.height);
  if (v.rear_vehicle && bay) return "The gold ring marks the open ground behind the outlined " + v.rear_vehicle + ". Is this where you want to get out?";
  if (bay?.extent === "interior_point_only") return (bay.passenger_tap_xy ? "The blue diamond marks your tap. " : "")
    + "The gold ring shows the spot in the " + where + " of the photo. Check that it is where you meant.";
  if (bay?.extent === "visible_parking_surface") return (bay.passenger_tap_xy ? tapName(v) + " is the blue diamond. " : "")
    + "The gold ring is inside the visible parking area in the " + where + ". The photo leaves its full edges uncertain.";
  if (bay) return (bay.passenger_tap_xy ? tapName(v) + " is the blue diamond. " : "")
    + "The gold ring marks the centre of the outlined parking bay in the " + where + " of the photo.";
  if (v.selected_by === "passenger") return "The blue pin is at the centre of your mark, in the " + where + " of the photo. Save it if this is where you want to get out.";
  if (v.selected_by === "passenger_landmarks") return "The yellow pin in the " + where + " uses the two vehicles you chose.";
  return "Check the yellow pin in the " + where + " of the photo.";
}
function response(v, stage, bay, choices) {
  if (photoState === "failed") return ["error", retired ? "Reopen this scene" : "Reload the road photo",
    retired ? "This session ended. Reopen the scene to start a new request." : "Open the photo again to check a location."];
  if (busy && notice) return ["working", notice, ""];
  if (!session) return ["idle", "Where would you like to get out?", ""];
  if (submitting) return ["working", "Sending your request…", ""];
  if (session.pending_grounding) return ["working", "Finding your spot…", v.can_request_area
    ? "You can mark the spot on the photo now or wait for a suggestion." : "You can cancel while we check the photo."];
  if (staleEdit()) return ["error", "This request has changed", "Your draft is below. Go back to the current request to edit it again."];
  if (capture) {
    const source = capture.mic.label ? "Recording with " + capture.mic.label + ". " : "The input level should move as you speak. ";
    return ["listening", "Listening…", source + "Press Stop recording to review your words."];
  }
  if (stage === "review") return ["review", "Did I hear you correctly?", "Edit any words below, then send the transcript."];
  if (stage === "failed" || stage === "recording") return ["clarify", "Try voice again or type", "Your description can also be sent as text."];
  if (photo.active) return ["mark", "Choose a spot on the photo", "Mark where you want to get out. For a painted parking bay, tap inside the bay you mean."];
  if (pairs.open) return ["mark", "Which vehicles do you mean?", "Choose two vehicles and a position in the photo controls."];
  if (editing === "reply") return ["clarify", "Tell me more", v.result?.clarification_question || v.turns?.filter(t=>t.role==="assistant").at(-1)?.text || ""];
  if (editing) return ["idle", editing === "new" ? "Where would you like to get out?" : "Edit your description", ""];
  if (choices.length) {
    if (v.bay_reference_uncertain) return ["clarify", choices.some(c=>c.extent==="interior_point_only") ? (choices.length===1 ? "Is this the spot you meant?" : "Which spot did you mean?") : choices.some(c=>c.extent==="visible_parking_surface") ? "Does a highlighted area match?" : choices.length===1 ? "Is this the bay you meant?" : "Which outlined bay did you mean?",
      v.result.clarification_question + " You can also mark another spot."];
    if (choices.some(c=>c.extent==="interior_point_only")) return ["clarify","Which spot did you mean?",v.result.clarification_question];
    if (choices.some(c=>c.extent==="visible_parking_surface")) return ["clarify","Which highlighted area do you mean?",
      "Choose a label to review its pin, or mark a different spot on the photo."];
    if (choices.length === 1) return ["clarify", "Did you mean this side of the line?",
      "Bay A is the visible bay beside your mark. Choose it or mark a different spot."];
    const why = choices.some(choice => choice.passenger_tap_xy) ? "More than one bay is near your mark." : "More than one bay matches your description.";
    return ["clarify", "Which parking bay do you mean?", why + (choices.length === 2 ? " Choose bay A or bay B." : " Choose the labelled bay you mean.")];
  }
  if (v.state === "proposed") return ["proposed", bay?.extent === "interior_point_only" ? "Is this the spot you meant?" : bay?.extent === "visible_parking_surface" ? "Is this the parking area you meant?" : bay ? "Is this your intended bay?" : "Is this your intended location?", suggestion(v,bay)];
  if (v.state === "confirmed") {
    if (bay?.extent === "interior_point_only") return ["confirmed","Location saved","You chose the spot at the gold ring as your intended drop-off."];
    if (bay?.extent === "visible_parking_surface") return ["confirmed", "Location saved", "You chose the gold ring inside the outlined parking area as your intended drop-off."];
    if (bay) return ["confirmed", "Location saved", "You chose the outlined parking bay as your intended drop-off. The gold ring marks its centre."];
    const point = v.result?.destination_point_xy;
    const location = point ? " in the " + positionWords(point, canvas.width, canvas.height) + " of the photo" : "";
    return ["confirmed", "Location saved", "You chose the " + (v.selected_by === "passenger" ? "blue" : "yellow") + " pin" + location + " as your intended drop-off."];
  }
  if (v.state === "cancelled") return ["idle", "Request cancelled", "You can start again on this photo."];
  if (v.state === "error") return ["error", "The request couldn’t finish", v.landmark_error_guidance ||
    (v.parking_recovery ? "You can still mark the intended spot on the photo or edit your description." : "Try the description again or choose another scene.")];
  if (v.word_confirmation) return ["clarify", "Check one word", v.result.clarification_question];
  if (v.references_only && v.result?.anchor_boxes_xyxy?.length) {
    if (v.reference_kind === "vegetation") return ["clarify", "I found the roadside greenery", v.result.clarification_question];
    if (v.reference_kind === "possible_vehicle") return ["clarify", "Is this the vehicle you meant?", v.result.clarification_question];
    const vehicles = v.result.anchor_boxes_xyxy.length === 1 ? "vehicle" : "vehicles";
    return ["clarify", "I found the " + vehicles + " you described", v.result.clarification_question];
  }
  if (canMark(v)) return ["clarify", currentMark(v) ? "Choose another location" : "Which spot do you mean?", v.result?.clarification_question ||
    v.turns?.filter(t => t.role === "assistant").at(-1)?.text || "Mark the location on the photo or edit your description."];
  return ["idle", "Where would you like to get out?", ""];
}
function legend(v, offerMark, bay, choices) {
  const node = el("legend");
  const pin = ["proposed","confirmed"].includes(v.state) && v.result?.destination_point_xy
    && photoState === "ready" && !editing && !submitting && !session?.voice_stage && !session?.pending_grounding;
  node.dataset.kind = !photo.active && pin && !pairs.open ? (v.selected_by === "passenger" ? "manual" : "auto") : "none";
  let label = "Original photo";
  if (photoState === "loading") label = "Opening photo…";
  else if (photoState === "failed") label = "Photo unavailable";
  else if (pairs.open) label = "Numbers mark selectable vehicles";
  else if (photo.active) label = "Centre dot · your point to check";
  else if (choices.length) {
    label = choices.map((choice,i) => bayName(i,choice)).join(" / ");
    if (v.vehicle_ground_choice) label += " · Blue outline around the vehicle";
    if (choices.some(choice => choice.passenger_tap_xy)) label += " · Blue diamond at " + tapName(v).toLowerCase();
  }
  else if (pin) {
    if (bay) {
      node.dataset.kind = "bay";
      label = bay.extent === "interior_point_only" ? "Gold ring at the selected spot" : bay.extent === "visible_parking_surface" ? "Visible parking area · Gold ring at selected location" : "Gold ring at bay centre";
      if (v.rear_vehicle) label = "Gold ring behind the vehicle · Blue outline around the " + v.rear_vehicle;
      if (bay.passenger_tap_xy) label += " · Blue diamond at " + tapName(v).toLowerCase();
    } else label = destinationLabel(v);
    if (v.state === "confirmed") label += " · saved location";
    if (!bay && v.result.anchor_boxes_xyxy?.length === 2 && ["parking_model","passenger_landmarks"].includes(v.selected_by)) label += " · A and B mark the vehicles";
  } else if (!editing && !session?.voice_stage && v.area_landmarks) label = "A and B mark the vehicles";
  if (!choices.length && currentMark(v) && !pin && !photo.active && !editing) label="Cross = your last mark · no destination pin";
  else if (v.references_only && v.result?.anchor_boxes_xyxy?.length && !photo.active && !editing && !session?.pending_grounding) {
    const outlines = v.result.anchor_boxes_xyxy.length === 1 ? "Blue outline shows the vehicle" : "Blue outlines show the vehicles";
    label = v.reference_kind === "vegetation" ? "Roadside greenery in blue · Drop-off point unverified" :
      v.reference_kind === "possible_vehicle" ? "Blue outline shows a possible match · Drop-off point unverified" :
      outlines + " you described · Drop-off point unverified";
  }
  else if (offerMark && !photo.zoom) label = v.area_landmarks ? "A and B mark the vehicles · Tap the photo to mark your spot" : "Tap the photo to mark your spot";
  text(node, label);
  return label;
}
function render() {
  const v = session?.view || {}, stage = session?.voice_stage, current = activeScene();
  const opened = Boolean(session), ready = photoState === "ready", micOn = Boolean(capture);
  const pending = session?.pending_grounding;
  const bay = offeredBay(v,canvas.width,canvas.height), choices = visibleChoices(v);
  const marking = photo.active || pairs.open, locked = busy || micOn || Boolean(submitting), prior = lastUser(v.turns);
  const formOpen = !pending && !marking && (editing || Boolean(stage) || !v.state || v.state === "idle" || (v.state === "error" && !canMark(v)));
  const decision = opened && !pending && !formOpen && !marking && !terminal(v.state);
  const [kind, heading, status] = response(v, stage, bay, choices);
  el("destination-panel").dataset.state = kind;
  text(el("destination-title"), heading); text(el("status"), status);
  const observed=bay?.observed_landmarks || (choices.length===1 ? choices[0].observed_landmarks : null);
  el("pictured-landmarks").hidden=!observed?.length || Boolean(marking || formOpen);
  if(observed?.length) text(el("pictured-landmarks"),"Visible landmarks near the highlight include "+observed.join(", ")+".");
  el("stopping-note").hidden = !["proposed","confirmed"].includes(kind);
  const err = failure || session?.speech_error?.message || "";
  text(el("error"), err); el("error").hidden = !err;
  const shownWords = pending?.text ?? prior?.text;
  const firstRequest = v.turns?.find(turn => turn.role === "user");
  const showOriginal = !formOpen && !pending && firstRequest && prior && firstRequest !== prior && firstRequest.text !== shownWords;
  el("original-request").hidden = !showOriginal;
  if (showOriginal) text(el("original-request").querySelector("p"), firstRequest.text);
  el("last-request").hidden = !shownWords || formOpen;
  if (shownWords) {
    text(el("last-request").querySelector("span"), pending?.mode === "voice" ? "Your reviewed transcript" : showOriginal ? "Your last reply" : "Your words");
    text(el("last-request").querySelector("p"), shownWords);
  }
  history(v);

  sceneSelect.disabled = busy || micOn || !opened;
  el("prev-scene").disabled = sceneSelect.disabled || sceneSelect.selectedIndex <= 0;
  el("next-scene").disabled = sceneSelect.disabled || sceneSelect.selectedIndex >= sceneSelect.options.length-1;
  el("next-outlined").disabled=sceneSelect.disabled;
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
  el("mic").disabled = busy || Boolean(submitting) || !opened || !ready || marking || (staleEdit() && !micOn);
  el("upload").disabled = locked || !opened || !ready || marking || staleEdit();
  text(el("mic"), micOn ? "Stop recording" : "Record voice");
  input.disabled = locked || !opened || marking;
  el("submit").disabled = input.disabled || !ready || !input.value.trim() || staleEdit();
  el("show-parking-areas").hidden = !(current?.parking_areas_indexed && formOpen && (!v.state || v.state === "idle")
    && !editing && !stage && !micOn && !input.value.trim());
  el("show-parking-areas").disabled = input.disabled || !ready;
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
  text(el("confirm"),bay && !bay.extent ? "Save this bay" : "Save this location");
  const yesNo = decision && v.state === "needs_clarification" && (v.word_confirmation || v.reference_kind === "possible_vehicle");
  for (const id of ["answer-yes", "answer-no"]) {
    el(id).hidden = !yesNo;
    el(id).disabled = locked || !ready;
  }
  text(el("answer-yes"), v.word_confirmation ? "Yes, I meant a car" : "Yes, that vehicle");
  el("draw").hidden = !(decision && canMark(v));
  el("draw").disabled = locked || !ready;
  el("take-over-area").hidden = !pending;
  el("take-over-area").disabled = locked || !ready || !v.can_request_area;
  el("draw").className = v.state === "proposed" || choices.length || yesNo ? "subtle" : "primary";
  text(el("draw"), choices.length ? "Mark a different spot" : v.state === "proposed" ? "Change spot on photo" : "Mark my spot on the photo");
  el("edit-words").hidden = !decision || !prior;
  el("edit-words").disabled = locked;
  text(el("edit-words"), v.state === "needs_clarification" || v.state === "insufficient_visual_evidence" ? "Answer in words" : "Edit description");
  el("choose-pair").hidden = !decision || (v.pair_choices?.length || 0) < 2;
  el("choose-pair").disabled = locked || !ready;
  el("new-request").hidden = !terminal(v.state) || formOpen;
  el("new-request").disabled = locked || !ready;
  el("cancel").hidden = !opened || (!pending && !stage && (!v.state || v.state === "idle" || terminal(v.state))) || marking;
  el("cancel").disabled = locked;
  pairs.controls(locked || !ready);
  const offerMark = decision && ready && !locked && !choices.length && ["needs_clarification","insufficient_visual_evidence"].includes(v.state);
  photo.controls(ready, locked, legend(v, offerMark, bay, choices), offerMark);
  const buttons = el("bay-buttons");
  const choiceKey = JSON.stringify(choices.map(({id,center_xy}) => [id,center_xy]));
  if (buttons.dataset.choices !== choiceKey) {
    buttons.dataset.choices = choiceKey;
    buttons.replaceChildren(...choices.map((choice,index) => {
      const button = document.createElement("button"); button.type = "button"; button.dataset.bayId = choice.id;
      const name=bayName(index,choice);
      text(button,name); button.setAttribute("aria-label","Choose "+name[0].toLowerCase()+name.slice(1));
      const [across,down] = choice.center_xy.map((value,i) => Math.round(100*value/(i ? canvas.height : canvas.width)));
      button.setAttribute("aria-description","Centre at " + across + "% across and " + down + "% down the photo.");
      return button;
    }));
  }
  for (const button of buttons.children) button.disabled = locked;
  el("bay-tools").hidden = !choices.length;
  canvas.classList.toggle("choosing-bay",choices.length > 0);
  paint();
  watchGrounding();
}
async function perform(work, message = "", after) {
  if (busy) return;
  stopPolling();
  if (submitting) { submitting.abort(); submitting=null; }
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
el("next-outlined").onclick=()=>{
  if(sceneSelect.disabled || !catalog) return;
  const start=catalog.scenes.findIndex(f=>f.key===sceneSelect.value);
  for(let i=1;i<=catalog.scenes.length;i++) {
    const next=catalog.scenes[(start+i)%catalog.scenes.length];
    if(next.parking_areas_indexed && next.key!==sceneSelect.value) {
      sceneSelect.value=next.key; perform(chooseScene,"Opening the photo…"); return;
    }
  }
};
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
  bayLabelBoxes = [];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (photoState !== "ready" || !canvas.clientWidth) return;
  const v = session.view, scale = photo.frameRatio();
  if (editing || submitting || session.voice_stage || session.pending_grounding) return;
  if (pairs.open) { pairs.paint(ctx,scale); return; }
  const choices = visibleChoices(v), bay = offeredBay(v,canvas.width,canvas.height);
  if (v.area_landmarks && (photo.active || (!choices.length && !bay))) drawChosenLandmarks(ctx,v.area_landmarks.boxes_xyxy,scale);
  if ((v.references_only || v.vehicle_ground_choice) && v.state==="needs_clarification" && !bay)
    drawVehicleReferences(ctx,v.result?.anchor_boxes_xyxy,scale,v.reference_kind==="vegetation" ? "Greenery" : "Vehicle");
  if (v.rear_vehicle && bay && ["proposed","confirmed"].includes(v.state) && !photo.active)
    drawVehicleReferences(ctx,v.result.anchor_boxes_xyxy,scale,"Vehicle");
  if (photo.active) {
    if (!photo.box) { photo.drawCursor(ctx,scale); return; }
    drawDestination(ctx, boxCentre(photo.box), photo.dragged ? photo.box : null, "mark_preview", scale);
    return;
  }
  const mark=currentMark(v);
  if (choices.length) { bayLabelBoxes = drawBays(ctx,choices,scale,true,tapName(v)); return; }
  if (!["proposed","confirmed"].includes(v.state) || !v.result?.destination_point_xy) {
    if (mark) drawCheckedMark(ctx,boxCentre(mark.box),scale);
    return;
  }
  if (bay) { drawBays(ctx,[bay],scale,false,tapName(v)); return; }
  if (["passenger_landmarks","parking_model"].includes(v.selected_by)) drawChosenLandmarks(ctx, v.result.anchor_boxes_xyxy, scale);
  const showBox = v.selected_by!=="passenger" || mark?.dragged;
  drawDestination(ctx, v.result.destination_point_xy, showBox ? v.result.destination_box_xyxy : null, v.selected_by, scale);
}
el("zoom-in").onclick = () => photo.setZoom(1, photo.active ? photo.cursor : session?.view.result?.destination_point_xy);
el("zoom-out").onclick = () => photo.setZoom(-1, photo.active ? photo.cursor : session?.view.result?.destination_point_xy);
el("pan").onclick = () => {
  photo.panning = !photo.panning; render();
  if (!photo.panning) photo.focusPhoto();
};
el("draw").onclick = () => photo.begin(session?.view.result?.destination_point_xy);
function selectBay(id) {
  if (busy || !visibleChoices(session?.view || {}).some(bay => bay.id === id)) return;
  perform(() => action("select-bay",{bay_id:id}), "Opening the selected bay…",focusResult);
}
el("bay-buttons").onclick = event => {
  const button = event.target.closest("button");
  if (button && !button.disabled) selectBay(button.dataset.bayId);
};
// In a zoomed photo a drag pans. Only a tap within one candidate selects a bay.
let bayPointer = null;
photo.viewport.addEventListener("pointerdown", event => {
  bayPointer = event.target === canvas && event.isPrimary && event.button === 0 && !busy && visibleChoices(session?.view || {}).length
    ? {id:event.pointerId,x:event.clientX,y:event.clientY,revision:session.revision,session:session.session} : null;
},true);
photo.viewport.addEventListener("pointermove", event => {
  if (bayPointer?.id === event.pointerId && Math.hypot(bayPointer.x-event.clientX,bayPointer.y-event.clientY) > 5) bayPointer=null;
});
photo.viewport.addEventListener("pointercancel", () => { bayPointer=null; });
photo.viewport.addEventListener("pointerup", event => {
  const pressed = bayPointer; bayPointer = null;
  if (!pressed || pressed.id !== event.pointerId || pressed.session !== session?.session || pressed.revision !== session.revision) return;
  const point = photo.point(event);
  const label = bayLabelBoxes.find(({box:[l,t,r,b]}) => point[0] >= l && point[0] <= r && point[1] >= t && point[1] <= b);
  const id = label ? label.id : bayAtPoint(visibleChoices(session.view),point,1.5*photo.frameRatio())?.id;
  if (id) selectBay(id);
},true);
el("take-over-area").onclick = () => perform(async () => {
  await action("take-over-area",{request_id:session.pending_grounding.request_id}); photo.begin();
}, "Opening the photo to mark…", () => photo.focusPhoto());
el("discard-box").onclick = () => { photo.reset(); render(); focus("draw"); };
el("use-box").onclick = () => {
  if (photo.box && photoState === "ready") perform(async () => {
    const box=photo.box.slice(), dragged=photo.dragged;
    await action("area", {box});
    checkedMark={session:session.session,request:session.view.request_id,box,dragged}; photo.reset();
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
  perform(() => action("confirm",{proposal_id:id}), "Saving your location…", focusResult);
};
el("cancel").onclick = () => perform(async () => {
  const pending = session.pending_grounding;
  await action("cancel",pending ? {request_id:pending.request_id} : {});
  input.value=""; resetSelection(); previousDraft=null; editing=false;
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
async function sendAutomatic(payload) {
  const controller = new AbortController(), sid = session.session;
  const actionId = crypto.randomUUID(), revision = session.revision;
  submitting = controller; failure=""; render();
  let accepted = false;
  try {
    const next = await postAction("submit",{...payload,background:true},controller.signal,actionId);
    if (submitting !== controller || next.session !== sid) return;
    session = next; accepted = true;
  } catch (error) {
    if (submitting !== controller) return;
    failure = error.message || "Your request could not be sent.";
    try {
      const next = await api("/api/sessions/"+sid,{signal:controller.signal});
      if (submitting === controller && next.session === sid) {
        session = next;
        accepted = next.last_action_id === actionId && next.revision > revision;
        if (accepted) failure="";
      }
    } catch (_) { /* Keep the submitted draft available when the connection is down. */ }
  } finally {
    if (submitting === controller) {
      submitting = null;
      if (accepted) { input.value=""; editing=false; previousDraft=editTarget=null; resetSelection(); }
      render();
      if (accepted && document.activeElement !== sceneSelect) focusResult();
    }
  }
}
function sendInstruction(choice = "automatic") {
  if (busy || submitting || session?.pending_grounding || !input.value.trim() || photoState !== "ready" || staleEdit()) return;
  const payload = {text:input.value.trim(), mode:session.voice_stage === "review" ? "voice" : "typed"};
  if (editing === "replace") payload.replaces_request_id = editTarget;
  if (choice === "vehicles") payload.point_to_vehicles = true;
  if (choice === "area") payload.mark_on_photo = true;
  if (choice === "automatic") { void sendAutomatic(payload); return; }
  perform(async () => {
    await action("submit",payload); input.value=""; editing=false; previousDraft=null; resetSelection();
    if (choice === "vehicles" && session.view.pair_choices?.length >= 2) pairs.begin(session.view.pair_choices);
    if (choice === "area" && session.view.state === "needs_clarification" && session.view.selected_by === "passenger") photo.begin();
  }, choice === "area" ? "Opening the photo to mark…" : "Opening the vehicle choices…",
  () => photo.active ? photo.focusPhoto() : pairs.open ? el("pair-first").focus() : focusResult());
}
el("form").onsubmit = e => { e.preventDefault(); sendInstruction(); };
for (const [id, answer] of [["answer-yes", "Yes"], ["answer-no", "No"]]) {
  const button = el(id);
  button.onclick = () => {
    if (button.hidden || button.disabled) return;
    startEditing("reply", answer);
    sendInstruction();
  };
}
el("show-parking-areas").onclick = () => {
  const button = el("show-parking-areas");
  if (button.hidden || button.disabled || input.value.trim()) return;
  input.value = "Find the marked parking bays.";
  sendInstruction();
};
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
window.addEventListener("pagehide", () => {
  stopPolling();
  if (submitting) { submitting.abort(); submitting=null; }
  if (capture) { clearTimeout(capture.timer); capture.mic.close(); capture=null; }
});
window.addEventListener("pageshow", () => { if (session) render(); });
perform(async () => {
  catalog=await api("/api/scenes?preview=1");
  const covered = new Set(catalog.footage_scenes);
  catalog.searchableScenes = covered;
  catalog.footageRecordings = new Set(catalog.scenes.filter(f => covered.has(f.key)).map(f => f.recording_id));
  el("footage-link").hidden=!catalog.footage_available;
  for (const f of catalog.scenes) addFrame(sceneSelect,f);
  const requested = new URLSearchParams(window.location.search).get("scene");
  if (catalog.scenes.some(f => f.key === requested)) sceneSelect.value = requested;
  await chooseScene();
  // The original gallery can be used while individual parking outlines are verified.
  void api("/api/scenes").then(full=>{
    const byKey=new Map(catalog.scenes.map(f=>[f.key,f]));
    for(const frame of full.scenes) {
      const current=byKey.get(frame.key);
      if(current) {current.parking_areas_indexed=frame.parking_areas_indexed; addFrame(sceneSelect,current);}
    }
    el("next-outlined").hidden=!full.scenes.some(f=>f.parking_areas_indexed);
    render();
  }).catch(()=>{});
  void refreshMicrophones();
}, "Opening the photo…");
