import {addFrame, openFrame} from "./frames.js";
import {VehiclePairPicker, drawChosenLandmarks} from "./pair-choices.js";

const el = id => document.getElementById(id);
const sceneSelect = el("scene"), image = el("frame"), canvas = el("overlay"), input = el("instruction");
let catalog, session, busy = false, capture = null, drawing = false, startPoint = null, rectangle = null;
let notice = "", failure = "";
const pairs = new VehiclePairPicker(render);

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "The request failed.");
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
function activeScene() { return catalog.scenes.find(s => s.key === session?.scene); }
function resetDraw() { drawing = false; startPoint = rectangle = null; }
function resetSelection() { resetDraw(); pairs.reset(); }
function text(node, content) { node.textContent = content || ""; }
function addMessage(turn) {
  const bubble = document.createElement("p");
  bubble.className = "bubble " + (turn.role === "user" ? "user" : "assistant");
  const speaker = document.createElement("b");
  text(speaker, turn.role === "user" ? "You" : "Passenger assistant");
  bubble.append(speaker, document.createTextNode(turn.text));
  el("messages").appendChild(bubble);
}
function showLegend(view, area) {
  const legend = el("legend");
  legend.replaceChildren();
  if (pairs.open) {
    text(legend, "Numbered outlines show vehicles detected in this frame.");
    return;
  }
  if (drawing || !area) {
    const landmarks = view.area_landmarks;
    text(legend, landmarks ? (landmarks.source === "passenger_landmarks" ? "A and B mark your two chosen vehicles."
      : "A and B mark the two vehicles matched in your instruction.") : drawing ? "Marking a passenger-selected area" : "Original frame");
    return;
  }
  const marked = view.selected_by === "passenger";
  for (const [shape, label] of [["point", "Dot marks the proposed road location"],
    ["region", marked ? "Outline shows your marked area" : "Outline shows the approximate area"]]) {
    const key = document.createElement("span"), symbol = document.createElement("i");
    key.className="legend-key"; symbol.className="key-"+shape+(marked?" passenger":"");
    symbol.setAttribute("aria-hidden","true"); key.append(symbol,document.createTextNode(label)); legend.append(key);
  }
  if (view.selected_by === "passenger_landmarks") {
    const anchors = document.createElement("span");
    text(anchors,"A and B mark your two chosen vehicles."); legend.append(anchors);
  }
}
function render() {
  const v = session?.view || {}, stage = session?.voice_stage;
  const open = Boolean(session), micOn = Boolean(capture);
  sceneSelect.disabled = busy || micOn || !open;
  el("open-frame").disabled = sceneSelect.disabled;
  el("speech-controls").hidden = !catalog?.speech_available;
  el("mic").disabled = busy || !open || drawing || pairs.open;
  el("upload").disabled = busy || micOn || !open || drawing || pairs.open;
  text(el("mic"), micOn ? "Stop recording" : "Record voice");
  input.disabled = busy || micOn || !open || drawing || pairs.open;
  el("submit").disabled = input.disabled || !input.value.trim();
  text(el("submit"), stage === "review" ? "Send reviewed transcript" :
    v.state === "needs_clarification" ? "Send clarification" : "Send instruction");
  text(el("input-label"), stage === "review" ? "Review and edit the transcript" :
    v.state === "needs_clarification" ? "Your clarification" : "Passenger instruction");
  const statuses = {confirmed:"Destination confirmed.", cancelled:"Request cancelled.",
    needs_clarification:"Clarification requested.", proposed:"Check the proposed road location on the image.",
    insufficient_visual_evidence:"The requested area could not be identified.", error:"The request could not be completed."};
  const voiceStatus = stage === "review" ? "Review the words before sending." :
    stage === "failed" ? "Record again or type the instruction." : stage === "recording" ? "Waiting for new audio." : "";
  const pairStatus = v.state==="proposed" && v.selected_by==="passenger_landmarks" ?
    "Check the road location calculated from the two vehicles you chose." : "";
  const areaStatus = !drawing && !pairs.open && (v.area_landmarks ?
    "The road location is still unclear. Mark where you want to get out beside A and B." : v.unmatched_described_pair ?
    "You described two vehicles. Select their outlines or mark the intended road area directly." : "");
  text(el("status"), notice || voiceStatus || v.landmark_error_guidance || areaStatus || pairStatus || statuses[v.state]);
  const interpretationError = v.landmark_error_guidance ? "" : v.can_request_pointing && v.error?.code==="invalid_model_output"
    ? "I couldn’t match your words to the picture. You can describe the place again or choose two vehicles."
    : v.error?.message;
  const err = failure || session?.speech_error?.message || interpretationError || "";
  text(el("error"), err); el("error").hidden = !err;
  el("messages").replaceChildren();
  (v.turns || []).forEach(addMessage);
  el("messages").scrollTop = el("messages").scrollHeight;
  const canMark = !stage && (["proposed","needs_clarification","insufficient_visual_evidence"].includes(v.state)
    || Boolean(v.landmark_error_guidance));
  const pendingWords = Boolean(input.value.trim());
  const recoverWords = !stage && (v.pair_choices?.length||0)<2 && ["error","proposed","needs_clarification","insufficient_visual_evidence"].includes(v.state)
    && v.turns?.some(turn=>turn.role==="user");
  const pointButton = el("submit-pointing");
  pointButton.hidden = !v.can_request_pointing || (!pendingWords && !recoverWords) || drawing || pairs.open;
  pointButton.disabled = input.disabled || (Boolean(stage) && stage!=="review");
  text(pointButton, pendingWords ? "Send and choose two vehicles" : "Review my words and choose two vehicles");
  el("draw").hidden = !canMark || drawing || pairs.open;
  el("draw").disabled = busy || micOn || pendingWords;
  text(el("draw"), v.area_landmarks || v.unmatched_described_pair || v.landmark_error_guidance ? "Mark the intended road area" : "Mark an area myself");
  el("choose-pair").hidden = Boolean(stage) || drawing || pairs.open || (v.pair_choices?.length||0)<2;
  el("choose-pair").disabled = busy || micOn || pendingWords;
  text(el("choose-pair"), v.area_landmarks ? "Change the two vehicles" : "Choose two vehicles");
  el("confirm").hidden = v.state !== "proposed" || !v.proposal_id || Boolean(stage) || drawing || pairs.open;
  el("confirm").disabled = busy || micOn || pendingWords;
  el("cancel").hidden = !open || (!stage && ["idle","cancelled","confirmed"].includes(v.state));
  el("cancel").disabled = busy || micOn;
  el("draw-tools").hidden = !drawing;
  el("use-box").disabled = busy || !rectangle || pendingWords;
  el("discard-box").disabled = busy;
  pairs.controls(busy || micOn || pendingWords);
  const area = v.result?.destination_box_xyxy;
  showLegend(v, area);
  canvas.classList.toggle("drawing", drawing);
  canvas.classList.toggle("choosing-pair", pairs.open);
  paint();
}
async function perform(work, message = "") {
  if (busy) return;
  busy = true; failure = ""; notice = message; render();
  try { await work(); }
  catch (error) {
    failure = error.message || "The action could not be completed.";
    if (session) {
      try {
        session = await api(endpoint());
        sceneSelect.value = session.scene;
      } catch { /* A retired tab must reopen a frame. */ }
    }
  } finally { busy = false; notice = capture ? "Recording. Stop when you have finished speaking." : ""; render(); }
}
async function chooseScene() {
  const old = session?.session;
  input.value = ""; resetSelection();
  session = await post("/api/sessions", {scene:sceneSelect.value, previous:old});
  const f = activeScene();
  canvas.width = f.width; canvas.height = f.height;
  image.parentElement.style.aspectRatio = f.width + " / " + f.height;
  image.src = endpoint("image");
  image.alt = "Original image from " + sceneSelect.selectedOptions[0].textContent;
}
el("open-frame").onclick = () => perform(async () => {
  const f = await openFrame(sceneSelect,el("frame-id"));
  if (!catalog.scenes.some(s => s.key === f.key)) catalog.scenes.push(f);
  await chooseScene();
  el("frame-picker").open = false;
}, "Opening the original frame…");
function paint() {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const v = session?.view;
  const scale = canvas.width / Math.max(canvas.clientWidth, 1);
  if (pairs.open) { pairs.paint(ctx,scale); return; }
  const box = drawing ? rectangle : v?.result?.destination_box_xyxy;
  if (v?.area_landmarks) drawChosenLandmarks(ctx, v.area_landmarks.boxes_xyxy, scale);
  if (!box) return;
  if (!drawing && v.selected_by === "passenger_landmarks") drawChosenLandmarks(ctx,v.result.anchor_boxes_xyxy,scale);
  const [x1,y1,x2,y2] = box;
  const passenger = drawing || v.selected_by === "passenger";
  ctx.save();
  if (passenger) {
    ctx.fillStyle = "rgba(66,232,248,.12)";
    ctx.fillRect(x1, y1, x2-x1, y2-y1);
  }
  if (!passenger || drawing) ctx.setLineDash([8*scale,5*scale]);
  ctx.lineWidth = 5 * scale; ctx.strokeStyle = "#14241e"; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.lineWidth = 2.6 * scale; ctx.strokeStyle = passenger ? "#52effa" : "#ffd22d";
  ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  const point = drawing ? null : v.result.destination_point_xy;
  if (point) {
    ctx.setLineDash([]); ctx.beginPath(); ctx.arc(point[0],point[1],6*scale,0,2*Math.PI);
    ctx.fillStyle = ctx.strokeStyle; ctx.fill(); ctx.lineWidth = 2*scale; ctx.strokeStyle="#12241a"; ctx.stroke();
  }
  ctx.restore();
}
function imagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  return [Math.max(0,Math.min(canvas.width, (event.clientX-rect.left)*canvas.width/rect.width)),
    Math.max(0,Math.min(canvas.height, (event.clientY-rect.top)*canvas.height/rect.height))];
}
function updateBox(event, final = false) {
  const [x,y] = imagePoint(event), [a,b] = startPoint;
  const bounds = canvas.getBoundingClientRect();
  const sx = canvas.width / bounds.width, sy = canvas.height / bounds.height;
  if (final && Math.hypot((x-a)/sx, (y-b)/sy) <= 3) {
    // A tap previews a small visible region. The server still checks road support inside the requested mark.
    rectangle = [Math.max(0,Math.floor(x-6*sx)), Math.max(0,Math.floor(y-6*sy)),
      Math.min(canvas.width,Math.ceil(x+6*sx)), Math.min(canvas.height,Math.ceil(y+6*sy))];
  } else {
    rectangle = [Math.floor(Math.min(x,a)), Math.floor(Math.min(y,b)), Math.ceil(Math.max(x,a)), Math.ceil(Math.max(y,b))];
  }
  if (rectangle[2] <= rectangle[0] || rectangle[3] <= rectangle[1]) rectangle = null;
  paint(); el("use-box").disabled = !rectangle;
}
canvas.addEventListener("pointerdown", e => {
  if (pairs.open && !busy && e.isPrimary) { e.preventDefault(); pairs.pickAt(imagePoint(e)); return; }
  if (!drawing || busy || !e.isPrimary) return;
  e.preventDefault(); canvas.setPointerCapture(e.pointerId); startPoint=imagePoint(e); rectangle=null; paint();
});
canvas.addEventListener("pointermove", e => { if (drawing && startPoint && e.isPrimary) updateBox(e); });
canvas.addEventListener("pointerup", e => {
  if (!drawing || !startPoint || !e.isPrimary) return;
  updateBox(e,true); startPoint=null;
});
canvas.addEventListener("pointercancel", () => { startPoint=rectangle=null; render(); });
el("draw").onclick = () => { if (!input.value.trim()) { drawing=true; rectangle=null; render(); } };
el("discard-box").onclick = () => { resetDraw(); render(); };
el("use-box").onclick = () => {
  if (!input.value.trim()) perform(async () => { await action("area", {box:rectangle}); resetDraw(); });
};
el("choose-pair").onclick = () => { if (!input.value.trim()) pairs.begin(session.view.pair_choices); };
el("use-pair").onclick = () => {
  if (!pairs.hasPair() || input.value.trim()) return;
  const request = {track_ids:pairs.values(), placement:pairs.position.value};
  perform(async () => { await action("select-pair",request); pairs.reset(); },
    "Checking the road beside your two vehicles…");
};
el("confirm").onclick = () => { const proposal = session.view.proposal_id;
  perform(async () => { await action("confirm", {proposal_id:proposal}); }); };
el("cancel").onclick = () => perform(async () => {
  const retiringAudio = Boolean(session.voice_stage);
  await action("cancel"); if (retiringAudio) input.value=""; resetSelection();
});
function sendInstruction(pointing) {
  if (!input.value.trim()) return;
  const words = input.value.trim(), mode = session.voice_stage === "review" ? "voice" : "typed";
  const payload = {text:words, mode};
  if (pointing) payload.point_to_vehicles = true;
  perform(async () => {
    await action("submit", payload); input.value=""; resetSelection();
    if (pointing && session.view.pair_choices?.length>=2) pairs.begin(session.view.pair_choices);
  }, pointing ? "Opening the vehicle choices…" : "Checking the instruction against this frame…");
}
el("form").onsubmit = e => { e.preventDefault(); sendInstruction(false); };
el("submit-pointing").onclick = () => {
  if (!input.value.trim()) {
    const prior = session.view.turns.filter(turn=>turn.role==="user").at(-1);
    if (prior) { input.value=prior.text; input.focus(); render(); }
    return;
  }
  sendInstruction(true);
};
input.addEventListener("input", render);
sceneSelect.onchange = () => perform(chooseScene);
image.addEventListener("error", () => { failure="Could not open the bound original frame."; render(); });
window.addEventListener("resize", paint);

function encodeWav(chunks, rate) {
  const n = chunks.reduce((v,a) => v + a.length, 0), buffer = new ArrayBuffer(44 + 2*n), d = new DataView(buffer);
  function ascii(offset, word) { for (let i=0;i<word.length;i++) d.setUint8(offset+i, word.charCodeAt(i)); }
  ascii(0,"RIFF"); d.setUint32(4,36+2*n,true); ascii(8,"WAVE"); ascii(12,"fmt ");
  d.setUint32(16,16,true); d.setUint16(20,1,true); d.setUint16(22,1,true);
  d.setUint32(24,rate,true); d.setUint32(28,2*rate,true); d.setUint16(32,2,true); d.setUint16(34,16,true);
  ascii(36,"data"); d.setUint32(40,2*n,true);
  let offset=44;
  for (const chunk of chunks) for (const sample of chunk) {
    const v=Math.max(-1,Math.min(1,sample)); d.setInt16(offset,Math.round(v*(v<0 ? 32768 : 32767)),true); offset+=2;
  }
  return new Blob([buffer], {type:"audio/wav"});
}
async function sendAudio(blob) {
  if (!blob.size || blob.size > 12*1024*1024) throw new Error("Choose a nonempty WAV up to 12 MiB.");
  session = await api(endpoint("audio"), {method:"POST", headers:{"Content-Type":"audio/wav",
    "X-Session-Revision":String(session.revision)}, body:blob});
  input.value = session.transcript || "";
}
async function beginAudio() {
  input.value=""; resetSelection();
  await action("begin-audio");
}
el("upload").onclick = () => el("file").click();
el("file").onchange = () => {
  const file = el("file").files[0]; el("file").value="";
  if (file) perform(async () => { await beginAudio(); await sendAudio(file); }, "Transcribing the recording…");
};
async function startMic() {
  await beginAudio();
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error("Microphone unavailable. Upload a WAV or type instead.");
  let stream, context, expired=false, timer;
  try {
    const permission = navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true}});
    permission.then(s => { if (expired) s.getTracks().forEach(t => t.stop()); }, () => {});
    stream = await Promise.race([permission,new Promise((_,reject) => {
      timer=setTimeout(() => { expired=true; reject(new Error("Microphone permission timed out. Upload a WAV or type instead.")); }, 20000);
    })]);
    clearTimeout(timer);
    context = new AudioContext({sampleRate:16000}); await context.audioWorklet.addModule("/capture.js");
    const source = context.createMediaStreamSource(stream), node = new AudioWorkletNode(context,"passenger-capture");
    const gain = context.createGain(); gain.gain.value=0;
    const blocks=[]; node.port.onmessage = e => { blocks.push(e.data); };
    source.connect(node); node.connect(gain); gain.connect(context.destination); await context.resume();
    capture={stream,context,source,node,blocks,rate:context.sampleRate,timer:setTimeout(stopMic,55000)};
  } catch(error) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (context) await context.close();
    throw error;
  } finally { clearTimeout(timer); }
}
async function stopMic() {
  if (!capture || busy) return;
  const c = capture; capture=null;
  await perform(async () => {
    clearTimeout(c.timer); c.stream.getTracks().forEach(t => t.stop()); c.source.disconnect(); c.node.disconnect();
    await c.context.close(); await sendAudio(encodeWav(c.blocks,c.rate));
  }, "Transcribing your recording…");
}
el("mic").onclick = () => { if (capture) stopMic(); else perform(startMic,"Opening the microphone…"); };
window.addEventListener("pagehide", () => { if (capture) capture.stream.getTracks().forEach(t => t.stop()); });
perform(async () => {
  catalog=await api("/api/scenes");
  el("footage-link").hidden=!catalog.footage_available;
  el("frame-picker").hidden=!catalog.footage_available;
  for (const f of catalog.scenes) addFrame(sceneSelect,f);
  await chooseScene();
});
