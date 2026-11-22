"use strict";

const $ = id => document.getElementById(id);
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
let view = null, frameIds = [], epoch = 0, sceneToken = null;
let dirty = false, fresh = true, renderedRequest = null, audioId = null, renderedAudio = null;
let speechPending = false, actionQueue = Promise.resolve(), qaSignature = null, initialized = false;
let recorder = null, microphone = null, audioUrl = null;
const seenErrors = new Set();

function notice(message) {
  $("notice").textContent = message || "";
  $("notice").hidden = !message;
}

async function readResponse(response) {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function post(path, body, headers = null) {
  const send = async () => readResponse(await fetch(path, {
    method: "POST", headers: headers || {"Content-Type": "application/json"},
    body: headers ? body : JSON.stringify(body),
  }));
  const pending = actionQueue.then(send);
  actionQueue = pending.catch(() => {});
  return pending;
}

async function refresh() {
  const capturedEpoch = epoch;
  try {
    const result = await readResponse(await fetch("/api/state", {cache: "no-store"}));
    if (capturedEpoch === epoch) render(result);
  } catch (error) {
    notice(`Connection unavailable: ${error.message}`);
  }
}

function clearScene() {
  sceneToken = null;
  dirty = false; fresh = true; audioId = null; renderedAudio = null; renderedRequest = null;
  $("scene-image").hidden = true;
  $("overlay").setAttribute("hidden", "");
  $("overlay").replaceChildren();
  $("empty-scene").hidden = false;
  $("stage").classList.add("empty");
  $("scene-title").textContent = "Your scene starts here";
  $("scene-meta").textContent = "Exact original frames · original pixel coordinates";
  $("proposal-controls").hidden = true;
  $("clarification").hidden = true;
  $("interpret").disabled = true;
  $("ask-question").disabled = true;
  $("new-instruction").disabled = true;
  $("cancel").hidden = true;
  $("original-asr").hidden = true;
  $("typed-mode").hidden = true;
  $("mode-label").textContent = "Typed input";
  $("instruction").value = "";
  $("reply").value = "";
  $("audio-preview").hidden = true;
}

async function invalidateSelection() {
  epoch++;
  clearScene();
  if (recorder?.state === "recording") recorder.stop();
  try { await post("/api/reset", {}); } catch (error) { notice(error.message); }
}

async function recordingChanged() {
  const recording = $("recording").value;
  const invalidation = invalidateSelection();
  frameIds = [];
  if (!recording) return;
  try {
    const result = await readResponse(await fetch(`/api/frames/${encodeURIComponent(recording)}`));
    if ($("recording").value !== recording) return;
    frameIds = result.frame_ids;
    $("frame").value = frameIds[0];
    $("frame").min = frameIds[0];
    $("frame").max = frameIds[frameIds.length - 1];
    $("frame-help").textContent = `${frameIds.length} available frames · ${result.frame_rate} fps`;
    await invalidation;
    await refresh();
  } catch (error) { notice(error.message); }
}

async function openScene() {
  notice(""); epoch++; clearScene();
  $("scene-title").textContent = "Opening exact original frame…";
  try {
    await post("/api/select", {recording_id: $("recording").value, frame_id: Number($("frame").value)});
    await refresh();
  } catch (error) { notice(error.message); }
}

function svg(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function drawOverlay() {
  const overlay = $("overlay");
  overlay.replaceChildren();
  if (!view?.selected || !sceneToken) return;
  const width = view.selected.frame.width;
  const scale = width / Math.max($("stage").clientWidth, 1);
  const box = (coords, color, dashed, label) => {
    const [x, y, right, bottom] = coords;
    overlay.append(svg("rect", {x, y, width: right - x, height: bottom - y,
      fill: dashed ? "none" : "#13a66e22", stroke: color, "stroke-width": scale * 2.5,
      "stroke-dasharray": dashed ? `${scale * 7} ${scale * 4}` : "none"}));
    const labelWidth = Math.min(width, scale * (label?.length * 7.5 + 8));
    const labelX = Math.max(labelWidth / 2, Math.min(width - labelWidth / 2, (x + right) / 2));
    if (label) overlay.append(svg("text", {x: labelX, y: Math.max(scale * 17, y - scale * 7), "text-anchor": "middle",
      fill: color, "font-size": scale * 14, "font-family": "Segoe UI,Arial,sans-serif",
      "font-weight": 700, stroke: "white", "stroke-width": scale * 3, "paint-order": "stroke"}, label));
  };
  if ($("show-tracks").checked) {
    for (const track of view.selected.tracks) box(track.bbox_xyxy, "#276ec4", true, `${track.class_name} · ${track.track_id}`);
  }
  const proposal = !dirty && !fresh ? view.state?.proposal : null;
  if (proposal) {
    for (const anchor of proposal.result.anchor_boxes_xyxy || []) box(anchor, "#178ccb", true, "Landmark");
    box(proposal.result.destination_box_xyxy, "#078654", false,
      view.state.status === "confirmed" ? "Confirmed destination" : "Proposed destination");
    const [cx, cy] = proposal.result.destination_point_xy;
    overlay.append(svg("circle", {cx, cy, r: scale * 6, fill: "#078654", stroke: "white", "stroke-width": scale * 2}));
  }
}

function status(kind, kicker, title, text) {
  const card = $("grounding-status");
  card.className = `status-card ${kind}`;
  card.replaceChildren(element("span", kicker, "status-kicker"), element("h3", title), element("p", text));
}

function renderInteraction(data) {
  const state = data.state;
  const selected = Boolean(data.selected);
  if (data.audio && data.audio.audio_id !== renderedAudio) {
    renderedAudio = audioId = data.audio.audio_id;
    speechPending = false; fresh = true; dirty = true;
    $("instruction").value = data.audio.text;
    $("original-text").textContent = data.audio.text;
    $("original-asr").hidden = false;
    $("typed-mode").hidden = false;
    $("mode-label").textContent = "Voice transcript · editable";
  }
  if (state && state.request.request_id !== renderedRequest && !fresh) {
    renderedRequest = state.request.request_id;
    $("instruction").value = state.request.text;
    dirty = false;
    if (state.request.input_mode === "voice") {
      $("original-asr").hidden = false;
      $("original-text").textContent = state.request.original_transcript;
      $("mode-label").textContent = "Voice transcript · editable";
      $("typed-mode").hidden = false;
    } else {
      audioId = null;
      $("original-asr").hidden = true;
      $("typed-mode").hidden = true;
      $("mode-label").textContent = "Typed input";
    }
  }
  $("interpret").textContent = fresh ? "Interpret instruction" : "Update interpretation";
  $("interpret").disabled = !selected || (state?.status === "pending" && !dirty);
  $("new-instruction").disabled = !selected;
  $("ask-question").disabled = !selected;
  $("clarification").hidden = !state || state.status !== "needs_clarification" || dirty || fresh;
  $("proposal-controls").hidden = !state || state.status !== "proposed" || dirty || fresh;
  $("cancel").hidden = !state || state.status === "cancelled";
  $("conversation").hidden = !state || !state.request.dialogue.length;
  $("turns").replaceChildren();
  for (const turn of state?.request.dialogue || []) {
    const li = element("li");
    li.append(element("strong", turn.role === "user" ? "Passenger: " : "Assistant: "), document.createTextNode(turn.text));
    $("turns").append(li);
  }
  if (dirty) status("neutral", "TEXT CHANGED", "Ready to interpret your text", "Submit this text to request a new interpretation. Any new proposal will need explicit confirmation.");
  else if (!state || fresh) status("neutral", "READY WHEN YOU ARE", selected ? "Describe your destination" : "Choose a scene to begin", "Any proposed destination will need your explicit confirmation.");
  else if (state.status === "pending") status("neutral", "INTERPRETING", "Waiting for the grounding model", "You can cancel or change the scene while this request is running.");
  else if (state.status === "needs_clarification") status("warning", "CLARIFICATION NEEDED", "Help resolve the reference", state.pending_clarification);
  else if (state.status === "proposed") status("success", "DESTINATION PROPOSED", "Review the highlighted region", "Check the destination in the scene, then explicitly confirm or edit your instruction.");
  else if (state.status === "confirmed") status("success", "EXPLICITLY CONFIRMED", "This destination is confirmed", "Your confirmation is bound to this proposal and original frame. Edits require a new confirmation.");
  else if (state.status === "cancelled") status("neutral", "CANCELLED", "No active destination", "You can edit the passenger text or start a new instruction.");
  else if (state.status === "insufficient_visual_evidence") status("warning", "INSUFFICIENT VISUAL EVIDENCE", "The scene cannot support this request", "No destination was proposed. Try another visible landmark or choose a different scene.");
  else status("error", "REQUEST ERROR", "The grounding request failed", data.last_outcome?.error?.message || "An operational error occurred. No proposal is active.");
  if (data.proposal_binding && data.selected) {
    const binding = structuredClone(data.proposal_binding), token = data.selected.token;
    $("confirm").onclick = async () => {
      try { await post("/api/confirm", {scene_token: token, confirmation: binding}); await refresh(); }
      catch (error) { notice(error.message); }
    };
  }
}

function renderQA(calls) {
  const signature = JSON.stringify(calls);
  if (signature === qaSignature) return;
  qaSignature = signature;
  const root = $("qa-results"); root.replaceChildren();
  if (!calls.length) root.append(element("div", "Your answer and cited frames will appear here.", "qa-empty"));
  for (const call of [...calls].reverse()) {
    const section = element("article", undefined, "qa-result");
    section.append(element("h3", call.question.question));
    section.append(element("div", `${call.question.recording_id} · original frame ${call.question.query_frame}`, "helper"));
    if (!call.result) {
      section.append(element("p", call.error?.message || "Retrieving evidence and requesting the answer…", "qa-answer"));
    } else {
      const result = call.result, answer = result.answer;
      section.append(element("span", answer.answer_status.replaceAll("_", " "), "answer-status"));
      const fallback = {insufficient_evidence: "The selected evidence does not establish an answer.",
        not_requested: "Answer generation is disabled for this run. Retrieved evidence is shown below.", error: "The answer provider failed."};
      section.append(element("p", answer.answer || fallback[answer.answer_status] || "", "qa-answer"));
      const citations = element("div", "Cited evidence: ", "citation-links");
      for (const id of answer.cited_evidence_ids) {
        const link = element("a", `${id} `); link.href = `#${call.question.question_id}-${id}`; citations.append(link);
      }
      if (answer.cited_evidence_ids.length) section.append(citations);
      for (const evidence of result.evidence) {
        const block = element("div", undefined, "evidence"); block.id = `${call.question.question_id}-${evidence.evidence_id}`;
        block.append(element("h4", `${evidence.evidence_id} · ${evidence.interval.track_id === null ? "Full scene" : `Track ${evidence.interval.track_id}`}`));
        const images = element("div", undefined, "evidence-images");
        for (const frame of evidence.images) {
          const link = element("a"); link.href = frame.image_url; link.target = "_blank"; link.rel = "noopener";
          const img = element("img"); img.src = frame.image_url; img.alt = `${evidence.evidence_id}, original frame ${frame.frame_id}`;
          link.append(img, element("span", `Original frame ${frame.frame_id}`)); images.append(link);
        }
        block.append(images); section.append(block);
      }
      if (!result.evidence.length) section.append(element("p", "No evidence returned.", "helper"));
    }
    root.append(section);
  }
}

function render(data) {
  view = data;
  if ($("recording").options.length === 1) {
    for (const recording of data.recordings) {
      const option = element("option", recording.recording_id); option.value = recording.recording_id; $("recording").append(option);
    }
  }
  if (!initialized) {
    initialized = true;
    fresh = !data.state;
    if (data.state) renderedAudio = data.audio?.audio_id || null;
    if (data.selected) {
      const frame = data.selected.frame;
      $("recording").value = frame.recording_id;
      $("frame").value = frame.frame_id;
      fetch(`/api/frames/${encodeURIComponent(frame.recording_id)}`).then(readResponse).then(result => {
        if ($("recording").value !== frame.recording_id) return;
        frameIds = result.frame_ids;
        $("frame").min = frameIds[0]; $("frame").max = frameIds[frameIds.length - 1];
        $("frame-help").textContent = `${frameIds.length} available frames · ${result.frame_rate} fps`;
      }).catch(error => notice(error.message));
    }
  }
  const condition = data.condition;
  $("condition-banner").hidden = !condition.development_adapters;
  $("condition-banner").textContent = condition.development_adapters
    ? `DEVELOPMENT WIRING MODE · Tracks, grounding, answers and transcripts may be generated fixtures. This is not trained-model acceptance. Adapter: ${condition.development_adapters}` : "";
  $("execution-details").replaceChildren();
  for (const [label, value] of [["Vision", condition.vision === "saved" ? "Validated saved predictions" : condition.vision === "fresh" ? "Fresh detector inference" : "Development observations"],
    ["Grounding", condition.grounding.replaceAll("_", " ")], ["Model", condition.model_id], ["QA answerer", condition.qa_answerer], ["Speech", condition.speech]]) {
    $("execution-details").append(element("dt", label), element("dd", value));
  }
  $("speech-note").textContent = data.speech_available ? "Audio transcripts stay editable before interpretation."
    : "Task 3 speech adapter is not connected. Typed instructions work; audio attempts retain an explicit error receipt.";
  if (data.selected) {
    const selected = data.selected, frame = selected.frame;
    if (sceneToken !== selected.token) {
      sceneToken = selected.token;
      $("scene-image").src = selected.image_url;
      $("scene-image").alt = `Original frame ${frame.frame_id}, recording ${frame.recording_id}`;
      $("stage").style.aspectRatio = `${frame.width}/${frame.height}`;
      $("overlay").setAttribute("viewBox", `0 0 ${frame.width} ${frame.height}`);
    }
    $("scene-image").hidden = false; $("overlay").removeAttribute("hidden"); $("empty-scene").hidden = true;
    $("stage").classList.remove("empty");
    $("scene-title").textContent = `${frame.recording_id} · frame ${frame.frame_id}`;
    $("scene-meta").textContent = `${frame.width} × ${frame.height} px · ${frame.timestamp_s.toFixed(2)} s · ${selected.tracks.length} current observation${selected.tracks.length === 1 ? "" : "s"}`;
  } else {
    sceneToken = null;
    $("scene-image").hidden = true; $("overlay").setAttribute("hidden", ""); $("empty-scene").hidden = false;
    $("stage").classList.add("empty");
    $("scene-title").textContent = data.selection?.status === "pending" ? "Opening exact original frame…"
      : data.selection?.status === "error" ? "Scene unavailable" : "Your scene starts here";
    $("scene-meta").textContent = "Exact original frames · original pixel coordinates";
  }
  renderInteraction(data); drawOverlay(); renderQA(data.qa);
  for (const job of data.jobs) {
    if (job.error && !seenErrors.has(job.job_id)) {
      seenErrors.add(job.job_id); speechPending = false; notice(`${job.kind}: ${job.error.message}`);
    }
  }
}

$("recording").addEventListener("change", recordingChanged);
$("frame").addEventListener("input", invalidateSelection);
$("load-scene").onclick = openScene;
for (const [id, step] of [["previous", -1], ["next", 1]]) $(id).onclick = () => {
  const index = frameIds.indexOf(Number($("frame").value));
  const next = Math.max(0, Math.min(frameIds.length - 1, index + step));
  if (frameIds.length) { $("frame").value = frameIds[next]; openScene(); }
};
$("show-tracks").onchange = drawOverlay;
window.addEventListener("resize", drawOverlay);
for (const route of ["dropoff", "qa"]) $("tab-" + route).onclick = () => {
  for (const name of ["dropoff", "qa"]) {
    $("tab-" + name).classList.toggle("active", name === route);
    $("tab-" + name).setAttribute("aria-selected", String(name === route));
    $(name + "-panel").hidden = name !== route;
  }
};
$("instruction").addEventListener("input", () => {
  const wasDirty = dirty; dirty = true; drawOverlay();
  if (sceneToken && (speechPending || (!wasDirty && ["proposed", "confirmed", "pending"].includes(view?.state?.status)))) {
    const discardAudio = speechPending;
    speechPending = false;
    post("/api/cancel", {scene_token: sceneToken, discard_audio: discardAudio}).then(refresh).catch(error => notice(error.message));
  }
  if (view) renderInteraction(view);
});
$("interpret").onclick = async () => {
  if (!sceneToken) return;
  notice("");
  const current = view?.state;
  const action = !fresh && current ? "/api/revise" : "/api/instruction";
  const payload = {scene_token: sceneToken, text: $("instruction").value};
  if (action === "/api/revise") payload.request_id = current.request.request_id;
  else if (audioId) payload.audio_id = audioId;
  try { await post(action, payload); fresh = false; dirty = false; await refresh(); }
  catch (error) { notice(error.message); await refresh(); }
};
$("send-reply").onclick = async () => {
  try {
    await post("/api/reply", {scene_token: sceneToken, request_id: view.state.request.request_id, text: $("reply").value});
    $("reply").value = ""; dirty = false; fresh = false; await refresh();
  } catch (error) { notice(error.message); }
};
$("cancel").onclick = async () => {
  dirty = false;
  try { await post("/api/cancel", {scene_token: sceneToken}); await refresh(); }
  catch (error) { notice(error.message); }
};
$("new-instruction").onclick = async () => {
  fresh = true; dirty = false; audioId = null;
  $("instruction").value = ""; $("original-asr").hidden = true; $("typed-mode").hidden = true;
  $("mode-label").textContent = "Typed input";
  drawOverlay(); if (view) renderInteraction(view);
  try { await post("/api/cancel", {scene_token: sceneToken}); await refresh(); }
  catch (error) { notice(error.message); }
};
$("typed-mode").onclick = () => {
  audioId = null; fresh = true; dirty = true;
  $("original-asr").hidden = true; $("typed-mode").hidden = true; $("mode-label").textContent = "Typed input";
  drawOverlay(); if (view) renderInteraction(view);
  if (sceneToken) post("/api/cancel", {scene_token: sceneToken, discard_audio: true}).then(refresh).catch(error => notice(error.message));
};
$("ask-question").onclick = async () => {
  notice("");
  try { await post("/api/qa", {scene_token: sceneToken, text: $("question").value}); await refresh(); }
  catch (error) { notice(error.message); }
};

async function uploadAudio(blob, filename, boundToken = sceneToken) {
  if (boundToken !== sceneToken) return notice("The scene changed while recording. Record again for the current scene.");
  if (!sceneToken) return notice("Open a scene before uploading or recording audio.");
  notice(""); dirty = true; fresh = true; audioId = null; speechPending = true;
  drawOverlay(); $("proposal-controls").hidden = true;
  try { await post("/api/cancel", {scene_token: boundToken, discard_audio: true}); }
  catch (error) { return notice(error.message); }
  if (!blob.size || blob.size > 16 * 1024 * 1024) {
    speechPending = false;
    return notice("Audio must be nonempty and no larger than 16 MiB.");
  }
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(blob); $("audio-preview").src = audioUrl; $("audio-preview").hidden = false;
  try {
    await post("/api/audio", blob, {"Content-Type": blob.type || "application/octet-stream",
      "X-Scene-Token": sceneToken, "X-Filename": encodeURIComponent(filename)});
    await refresh();
  } catch (error) { speechPending = false; notice(error.message); }
}
$("audio-file").onchange = async event => {
  const file = event.target.files[0];
  if (file) await uploadAudio(file, file.name);
  event.target.value = "";
};
$("record-audio").onclick = async () => {
  if (recorder?.state === "recording") { recorder.stop(); return; }
  if (!sceneToken) return notice("Open a scene before recording audio.");
  const recordingScene = sceneToken;
  $("record-audio").disabled = true;
  try {
    dirty = true; fresh = true; audioId = null;
    drawOverlay(); if (view) renderInteraction(view);
    await post("/api/cancel", {scene_token: recordingScene, discard_audio: true});
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error("Recording is unavailable in this browser. Upload an audio file instead.");
    microphone = await navigator.mediaDevices.getUserMedia({audio: true});
    if (recordingScene !== sceneToken) throw new Error("The scene changed. Start recording again for the current scene.");
    const ownedStream = microphone;
    const chunks = [];
    let recordingFailed = false;
    const ownedRecorder = new MediaRecorder(ownedStream);
    recorder = ownedRecorder;
    ownedRecorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    ownedRecorder.onerror = () => {
      recordingFailed = true;
      ownedStream.getTracks().forEach(track => track.stop());
      $("record-audio").textContent = "Record voice"; $("record-audio").classList.remove("recording");
      notice("Audio recording failed. No destination was submitted; try uploading an audio file.");
    };
    ownedRecorder.onstop = async () => {
      ownedStream.getTracks().forEach(track => track.stop());
      $("record-audio").textContent = "Record voice"; $("record-audio").classList.remove("recording");
      if (recordingFailed) return;
      const type = ownedRecorder.mimeType || "audio/webm";
      await uploadAudio(new Blob(chunks, {type}), type.includes("mp4") ? "microphone.mp4" : "microphone.webm", recordingScene);
    };
    ownedRecorder.start(); $("record-audio").disabled = false; $("record-audio").textContent = "Stop recording"; $("record-audio").classList.add("recording");
    setTimeout(() => { if (ownedRecorder.state === "recording") ownedRecorder.stop(); }, 60000);
  } catch (error) { $("record-audio").disabled = false; microphone?.getTracks().forEach(track => track.stop()); notice(`Microphone unavailable: ${error.message}`); }
};
refresh();
setInterval(refresh, 800);
