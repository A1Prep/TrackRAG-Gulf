import {addFrame, openFrame} from "./frames.js";

const get = id => document.getElementById(id);
function node(tag, text) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  return item;
}
async function fetchJson(url, opts) {
  const response = await fetch(url,opts);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Could not search the recording.");
  return body;
}
const scene = get("scene"), question = get("question"), mode = get("answer-mode"), results = get("results");
let busy = false;
function editing(enabled) {
  busy = !enabled;
  for (const control of [scene, question, mode, get("frame-id"), get("open-frame"), get("go")]) control.disabled = !enabled;
}
function clearResults() {
  results.replaceChildren();
  get("progress").textContent = "";
}
function selectScene() {
  clearResults();
  const url = new URL(location.href);
  if (scene.value) url.searchParams.set("scene", scene.value);
  else url.searchParams.delete("scene");
  history.replaceState(null, "", url);
  get("passenger-link").href = scene.value ? "/?scene=" + encodeURIComponent(scene.value) : "/";
}
editing(false);
try {
  const catalog = await fetchJson("/api/scenes");
  if (!catalog.footage_available) throw new Error("Footage search needs the original recordings and saved predictions.");
  const covered = new Set(catalog.footage_scenes);
  for (const f of catalog.scenes) if (covered.has(f.key)) addFrame(scene, f);
  const requested = new URLSearchParams(location.search).get("scene");
  if (requested) scene.value = requested;
  selectScene();
  if (requested && !scene.value) get("progress").textContent = "This frame has no attached footage. Choose a recording to search.";
  get("frame-picker").hidden=false;
  editing(true);
} catch (error) {
  get("progress").textContent = error.message;
}
scene.addEventListener("change", selectScene);
question.addEventListener("input", clearResults);
mode.addEventListener("change", clearResults);
get("open-frame").onclick=async () => {
  if (busy) return;
  editing(false);
  get("progress").textContent="Opening the original frame…";
  try {
    await openFrame(scene,get("frame-id"),true);
    selectScene();
    get("frame-picker").open=false;
  } catch(error) {
    get("progress").textContent=error.message;
  } finally { editing(true); }
};
get("search").addEventListener("submit",async event => {
  event.preventDefault();
  if (busy) return;
  clearResults();
  editing(false);
  get("progress").textContent = "Searching the selected recording…";
  try {
    const body = await fetchJson("/api/footage",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({scene:scene.value,question:question.value,hosted:mode.value==="gemini"})});
    const labels = {answered:"Gemini answer",error:"Answer service error",insufficient_evidence:"Insufficient evidence"};
    if (body.answer.answer_status !== "not_requested") {
      const fallback = body.answer.answer_status === "error" ? "The answer service could not complete this request." : "The available evidence does not establish an answer.";
      results.append(node("h2",labels[body.answer.answer_status]),node("p",body.answer.answer || fallback));
    }
    const count = body.evidence.length;
    get("progress").textContent = scene.selectedOptions[0].textContent + (count ? " · " + count + " evidence interval" + (count === 1 ? "" : "s") : " · No matching evidence");
    for (const item of body.evidence) {
      const article = node("article");
      article.className = "card review-form";
      const cited = body.answer.cited_evidence_ids.includes(item.id) ? " · cited in the answer" : "";
      article.append(node("h3",item.id+cited),node("p","Frames "+item.interval.start_frame+"–"+item.interval.end_frame));
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
        link.append(photo,node("span","Frame "+im.frame_id));
        strip.append(link);
      }
      article.append(strip);
      results.append(article);
    }
  } catch (error) {
    get("progress").textContent = error.message;
  } finally {
    editing(true);
  }
});
