const el = id => document.getElementById(id);
const sceneSelect = el("scene"), image = el("frame"), canvas = el("overlay"), input = el("instruction");
let catalog, session, busy = false, capture = null, drawing = false, startPoint = null, rectangle = null;
let notice = "", failure = "";

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
function text(node, content) { node.textContent = content || ""; }
function addMessage(turn) {
  const bubble = document.createElement("p");
  bubble.className = "bubble " + (turn.role === "user" ? "user" : "assistant");
  const speaker = document.createElement("b");
  text(speaker, turn.role === "user" ? "You" : "Passenger assistant");
  bubble.append(speaker, document.createTextNode(turn.text));
  el("messages").appendChild(bubble);
}
function render() {
  const v = session?.view || {}, stage = session?.voice_stage;
  const open = Boolean(session), micOn = Boolean(capture);
  sceneSelect.disabled = busy || micOn || !open;
  el("speech-controls").hidden = !catalog?.speech_available;
  el("mic").disabled = busy || !open || drawing;
  el("upload").disabled = busy || micOn || !open || drawing;
  text(el("mic"), micOn ? "Stop recording" : "Record voice");
  input.disabled = busy || micOn || !open || drawing;
  el("submit").disabled = input.disabled || !input.value.trim();
  text(el("submit"), stage === "review" ? "Send reviewed transcript" :
    v.state === "needs_clarification" ? "Send clarification" : "Send instruction");
  text(el("input-label"), stage === "review" ? "Review and edit the transcript" :
    v.state === "needs_clarification" ? "Your clarification" : "Passenger instruction");
  const statuses = {confirmed:"Destination confirmed.", cancelled:"Request cancelled.",
    needs_clarification:"Clarification requested.", proposed:"Review the area on the image.",
    insufficient_visual_evidence:"The requested area could not be identified.", error:"The request could not be completed."};
  const voiceStatus = stage === "review" ? "Review the words before sending." :
    stage === "failed" ? "Record again or type the instruction." : stage === "recording" ? "Waiting for new audio." : "";
  text(el("status"), notice || voiceStatus || statuses[v.state]);
  const err = failure || session?.speech_error?.message || v.error?.message || "";
  text(el("error"), err); el("error").hidden = !err;
  el("messages").replaceChildren();
  (v.turns || []).forEach(addMessage);
  el("messages").scrollTop = el("messages").scrollHeight;
  const canMark = !stage && ["proposed","needs_clarification","insufficient_visual_evidence"].includes(v.state);
  el("draw").hidden = !canMark || drawing;
  el("draw").disabled = busy || micOn;
  el("confirm").hidden = v.state !== "proposed" || !v.proposal_id || Boolean(stage) || drawing;
  el("confirm").disabled = busy || micOn || Boolean(input.value.trim());
  el("cancel").hidden = !open || (!stage && ["idle","cancelled","confirmed"].includes(v.state));
  el("cancel").disabled = busy || micOn;
  el("draw-tools").hidden = !drawing;
  el("use-box").disabled = busy || !rectangle;
  el("discard-box").disabled = busy;
  const area = v.result?.destination_box_xyxy;
  text(el("legend"), drawing ? "Marking a passenger-selected area" : !area ? "Original frame" :
    (v.selected_by === "passenger" ? "Passenger-marked area" : "Model-proposed area") +
    (v.state === "confirmed" ? " · confirmed" : ""));
  canvas.classList.toggle("drawing", drawing);
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
  input.value = ""; resetDraw();
  session = await post("/api/sessions", {scene:sceneSelect.value, previous:old});
  const f = activeScene();
  canvas.width = f.width; canvas.height = f.height;
  image.parentElement.style.aspectRatio = f.width + " / " + f.height;
  image.src = endpoint("image");
  image.alt = "Original image from " + sceneSelect.selectedOptions[0].textContent;
}
function paint() {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const v = session?.view;
  const box = drawing ? rectangle : v?.result?.destination_box_xyxy;
  if (!box) return;
  const scale = canvas.width / Math.max(canvas.clientWidth, 1);
  const [x1,y1,x2,y2] = box;
  const passenger = drawing || v.selected_by === "passenger";
  ctx.save();
  ctx.fillStyle = passenger ? "rgba(66,232,248,.22)" : "rgba(255,208,35,.22)";
  ctx.fillRect(x1, y1, x2-x1, y2-y1);
  ctx.lineWidth = 5 * scale; ctx.strokeStyle = "#14241e"; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.lineWidth = 2.6 * scale; ctx.strokeStyle = passenger ? "#52effa" : "#ffd22d";
  if (drawing) ctx.setLineDash([7*scale,4*scale]);
  ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  const point = drawing ? null : v.result.destination_point_xy;
  if (point) {
    ctx.setLineDash([]); ctx.beginPath(); ctx.arc(point[0],point[1],5*scale,0,2*Math.PI);
    ctx.fillStyle = ctx.strokeStyle; ctx.fill(); ctx.lineWidth = 2*scale; ctx.strokeStyle="#12241a"; ctx.stroke();
  }
  ctx.restore();
}
function imagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  return [Math.max(0,Math.min(canvas.width, (event.clientX-rect.left)*canvas.width/rect.width)),
    Math.max(0,Math.min(canvas.height, (event.clientY-rect.top)*canvas.height/rect.height))];
}
function updateBox(event) {
  const [x,y] = imagePoint(event), [a,b] = startPoint;
  rectangle = [Math.floor(Math.min(x,a)), Math.floor(Math.min(y,b)), Math.ceil(Math.max(x,a)), Math.ceil(Math.max(y,b))];
  if (rectangle[2] <= rectangle[0] || rectangle[3] <= rectangle[1]) rectangle = null;
  paint(); el("use-box").disabled = !rectangle;
}
canvas.addEventListener("pointerdown", e => {
  if (!drawing || busy || !e.isPrimary) return;
  e.preventDefault(); canvas.setPointerCapture(e.pointerId); startPoint=imagePoint(e); rectangle=null; paint();
});
canvas.addEventListener("pointermove", e => { if (drawing && startPoint && e.isPrimary) updateBox(e); });
canvas.addEventListener("pointerup", e => {
  if (!drawing || !startPoint || !e.isPrimary) return;
  updateBox(e); startPoint=null;
});
canvas.addEventListener("pointercancel", () => { startPoint=rectangle=null; render(); });
el("draw").onclick = () => { drawing=true; rectangle=null; render(); };
el("discard-box").onclick = () => { resetDraw(); render(); };
el("use-box").onclick = () => perform(async () => { await action("area", {box:rectangle}); resetDraw(); input.value=""; });
el("confirm").onclick = () => { const proposal = session.view.proposal_id;
  perform(async () => { await action("confirm", {proposal_id:proposal}); }); };
el("cancel").onclick = () => perform(async () => { await action("cancel"); input.value=""; resetDraw(); });
el("form").onsubmit = e => {
  e.preventDefault(); if (!input.value.trim()) return;
  const words = input.value.trim(), mode = session.voice_stage === "review" ? "voice" : "typed";
  perform(async () => { await action("submit", {text:words, mode}); input.value=""; resetDraw(); },
    "Checking the instruction against this frame…");
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
  input.value=""; resetDraw();
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
  for (const f of catalog.scenes) {
    const opt=document.createElement("option"); opt.value=f.key;
    opt.textContent=f.recording_id.replace(/^video_/, "Recording ")+" · frame "+f.frame_id+" · "+f.timestamp_s.toFixed(1)+" s";
    sceneSelect.appendChild(opt);
  }
  await chooseScene();
});
