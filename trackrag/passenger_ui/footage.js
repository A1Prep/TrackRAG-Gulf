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
try {
  const catalog = await fetchJson("/api/scenes");
  if (!catalog.footage_available) throw new Error("Footage search needs the original recordings and saved predictions.");
  for (const f of catalog.scenes) addFrame(get("scene"),f);
  get("frame-picker").hidden=false;
} catch (error) {
  get("progress").textContent = error.message;
  get("go").disabled = true;
}
get("open-frame").onclick=async () => {
  get("open-frame").disabled=get("go").disabled=true;
  get("progress").textContent="Opening the original frame…";
  try {
    await openFrame(get("scene"),get("frame-id"));
    get("results").replaceChildren();
    get("progress").textContent="";
    get("frame-picker").open=false;
  } catch(error) {
    get("progress").textContent=error.message;
  } finally { get("open-frame").disabled=get("go").disabled=false; }
};
get("search").addEventListener("submit",async event => {
  event.preventDefault();
  const results = get("results");
  results.replaceChildren();
  get("open-frame").disabled=get("scene").disabled=get("go").disabled=true;
  get("progress").textContent = "Searching the selected recording…";
  try {
    const body = await fetchJson("/api/footage",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({scene:get("scene").value,question:get("question").value,hosted:get("answer-mode").value==="gemini"})});
    const labels = {answered:"Gemini answer",error:"Answer service error",insufficient_evidence:"Insufficient evidence"};
    if (body.answer.answer_status !== "not_requested") {
      results.append(node("h2",labels[body.answer.answer_status]),node("p",body.answer.answer || "The available evidence does not establish an answer."));
    }
    get("progress").textContent = body.evidence.length + " evidence intervals · saved detector predictions.";
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
    get("open-frame").disabled=get("scene").disabled=get("go").disabled=false;
  }
});
