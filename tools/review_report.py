"""Render a completed reviewer run as a local HTML evidence report."""

import argparse
import html
import json
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote


def write_report(run_dir):
    run_dir = Path(run_dir)
    results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sections = []
    models, reported_models = set(), set()
    for receipt in (run_dir / "hosted").glob("*/receipt.json"):
        details = json.loads(receipt.read_text(encoding="utf-8"))
        models.add(details["model"])
        if details.get("reported_model"):
            reported_models.add(details["reported_model"])
    for number, result in enumerate(results, start=1):
        question = html.escape(result["question"]["question"])
        status = html.escape(result["answer"]["answer_status"].replace("_", " "))
        fallback = {
            "not_requested": "Answer generation was disabled for this run.",
            "insufficient_evidence": "The selected frames do not establish an answer.",
            "error": "The answer provider failed.",
        }
        answer = html.escape(result["answer"]["answer"] or fallback.get(result["answer"]["answer_status"], ""))
        anchors = {item["evidence_id"]: f"question-{number}-evidence-{i}" for i, item in enumerate(result["evidence"], 1)}
        cited = []
        for citation in result["answer"]["cited_evidence_ids"]:
            if citation in anchors:
                cited.append(f'<a href="#{anchors[citation]}">{html.escape(citation)}</a>')
        citations = f'<p>Cited evidence {", ".join(cited)}</p>' if cited else ""
        references = []
        for item in result["evidence"]:
            interval = item["interval"]
            images = []
            for frame in item["images"]:
                url = quote(frame["image_path"], safe="/")
                label = f"Frame {frame['frame_id']}"
                description = f"{label}, full scene" if frame["bbox_xyxy"] is None else f"{label} with the selected road user boxed"
                processed = Path("hosted") / sha256(result["question"]["question_id"].encode()).hexdigest() / f"{item['evidence_id']}-{frame['frame_id']}.jpg"
                model_input = f'<a class="processed" href="{quote(processed.as_posix(), safe="/")}" target="_blank">View model input</a>' if (run_dir / processed).exists() else ""
                images.append(f'<div><a href="{url}" target="_blank"><img src="{url}" alt="{description}" loading="lazy"><span>{label}</span></a>{model_input}</div>')
            subject = "Full scene" if interval["track_id"] is None else f"Track {interval['track_id']}"
            title = f"{item['evidence_id']} · {subject} · Frames {interval['start_frame']}–{interval['end_frame']}"
            references.append(f'<div class="evidence" id="{anchors[item["evidence_id"]]}"><h3>{html.escape(title)}</h3><div class="frames">{"".join(images)}</div></div>')
        evidence = "".join(references) or '<p>No evidence was returned.</p>'
        sections.append(f'<section id="question-{number}"><h2>{question}</h2><p class="status">{status}</p><p class="answer">{answer}</p>{citations}{evidence}</section>')
    model = html.escape(", ".join(sorted(models)) or "No successful hosted answer call")
    if reported_models and reported_models != models:
        model += " (provider-reported " + html.escape(", ".join(sorted(reported_models))) + ")"
    sources = ", ".join(sorted({result["source"] for result in results}))
    configuration = ""
    if (run_dir / "run.json").exists():
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        configuration = metadata["configuration"].get("adapter_module", "")
    method = {
        "tools.emt_trial_adapters": "Development baseline using labeled track history",
        "tools.temporal_trial_adapters": "Development trial using semantic evidence selection and labeled track history",
    }.get(configuration, "Recording review")
    sources = sources.replace("annotated", "human-labeled").replace("predicted", "model-predicted")
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>TrackRAG-Gulf evidence review</title><style>
body{{font-family:Inter,"Segoe UI",sans-serif;background:#f4f6f8;color:#17202a;margin:0;line-height:1.5}}
main{{max-width:1260px;margin:auto;padding:32px 24px}}h1{{font-size:32px;margin:0}}h2{{font-size:23px;margin-top:0}}
h3{{font-size:15px;margin-bottom:10px}}section{{background:white;border:1px solid #dde3e8;border-radius:12px;padding:24px;margin:24px 0}}
.meta,.status{{color:#506278}}.status{{font-size:14px;text-transform:uppercase;letter-spacing:.05em}}.answer{{font-size:18px}}
.frames{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}img{{width:100%;border-radius:6px;display:block}}
a{{color:#284f77;text-decoration:none}}a span{{display:block;margin:5px 0}}.processed{{font-size:13px;text-decoration:underline}}.evidence{{margin-top:24px}}
@media(max-width:700px){{.frames{{grid-template-columns:1fr}}main{{padding:20px 12px}}section{{padding:18px}}}}
</style></head><body><main><h1>TrackRAG-Gulf evidence review</h1>
<p class="meta">EMT development footage · {html.escape(sources)} tracks · {model}</p><p>{method}</p>
<p>Read each answer alongside its cited evidence. Click an image to inspect the original-size marked frame.</p>
{''.join(sections)}</main></body></html>'''
    destination = run_dir / "report.html"
    destination.write_text(page, encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    print(write_report(parser.parse_args().run))
