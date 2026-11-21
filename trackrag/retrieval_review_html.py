"""Local, model-free worksheet for reviewing Task 4 evidence candidates.

The rendered page reads only packaged original images and an explicitly selected
local JSONL draft. It sends no data anywhere and never assigns relevance labels.
Human provenance is a reviewer assertion; a static page cannot authenticate it.
"""

from html import escape
import json
import math
import re


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _json(value: object) -> str:
    """Serialize JSON safely even when placed inside a script element."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    for character, replacement in (("<", "\\u003c"), (">", "\\u003e"),
                                   ("&", "\\u0026"), ("\u2028", "\\u2028"),
                                   ("\u2029", "\\u2029")):
        text = text.replace(character, replacement)
    return text


def _image(image: dict) -> str:
    path = image["image_path"]
    if not isinstance(path, str) or not re.fullmatch(
        r"images/[0-9a-f]{64}\.(?:png|jpg|jpeg)", path
    ):
        raise ValueError("Review images must be packaged local images/<sha256> files")
    width, height = image["width"], image["height"]
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Review images require positive original dimensions")
    box = image["bbox_xyxy"]
    rectangle = ""
    if box is not None:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in box)):
            raise ValueError("Review image box must have four finite coordinates")
        left, top, right, bottom = box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Review image box must fit the original image")
        rectangle = (
            f'<rect x="{left}" y="{top}" width="{right - left}" '
            f'height="{bottom - top}" fill="none" stroke="#ff2020" '
            'stroke-width="3" vector-effect="non-scaling-stroke" />'
        )
    frame = _text(image["frame_id"])
    caption = "Full scene; no highlighted target" if box is None else "Target box in original pixels"
    return (
        '<figure><svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Frame {frame}">'
        f'<title>Frame {frame}: {caption}</title>'
        f'<image href="{_text(path)}" width="{width}" height="{height}" />'
        f'{rectangle}</svg><figcaption>Frame {frame} · {caption}</figcaption></figure>'
    )


def _facts(features: dict | None) -> str:
    if features is None:
        return '<p class="facts">Exact scene frame. No target identity or track facts.</p>'
    facts = (
        ("Class counts", features.get("class_counts", {})),
        ("Observed frames", features.get("observed_frame_ids", [])),
        ("Missing observation ranges", features.get("missing_observation_ranges", [])),
        ("Continuity breaks", features.get("continuity_breaks", [])),
    )
    return '<dl class="facts">' + ''.join(
        f'<dt>{label}</dt><dd>{_text(json.dumps(value, ensure_ascii=False))}</dd>'
        for label, value in facts
    ) + '</dl>'


def render_review(pack: dict, batches: dict[str, dict], labels: list[dict]) -> str:
    """Render a standalone worksheet beside its packaged ``images`` folder.

    Pack/batch schema validation belongs to the dataset builder. This renderer
    additionally rejects remote/traversing image references and unsafe geometry.
    Downloads contain all rows; pending candidates are never implicit negatives.
    """
    sections: list[str] = []
    expected: list[dict] = []
    for case in pack["cases"]:
        case_id = case["case_id"]
        batch = batches[case_id]
        question = batch["question"]
        histories = {item["history_id"]: item for item in batch["histories"]}
        cards: list[str] = []
        for candidate in batch["candidates"]:
            position = len(expected)
            expected.append({
                "case_id": case_id,
                "candidate_id": candidate["candidate_id"],
                "frame_ids": sorted({item["frame_id"] for item in candidate["images"]}),
            })
            track = "scene" if candidate["track_id"] is None else str(candidate["track_id"])
            history = histories.get(candidate["history_id"])
            history_context = (
                '<details class="history"><summary>Associated whole-history facts</summary>'
                + _facts(history["features"]) + '</details>' if history else ""
            )
            cards.append(
                f'<details class="candidate" data-position="{position}">'
                f'<summary><span>{_text(candidate["kind"])} · track {_text(track)} · '
                f'frames {_text(candidate["start_frame"])}–{_text(candidate["end_frame"])}</span>'
                '<strong class="badge">pending</strong></summary>'
                f'<p class="identifier">Candidate: {_text(candidate["candidate_id"])}</p>'
                + _facts(candidate["features"]) + history_context
                + '<div class="gallery">' + ''.join(_image(item) for item in candidate["images"])
                + '</div><div class="controls">'
                f'<label for="judgment-{position}">Relevance judgment</label>'
                f'<select id="judgment-{position}" class="judgment">'
                '<option value="pending">Pending — not reviewed</option>'
                '<option value="relevant">Relevant — supports this question</option>'
                '<option value="irrelevant">Irrelevant — does not support this question</option>'
                '<option value="uncertain">Uncertain — needs further review</option></select>'
                f'<label for="reason-{position}">Reason</label>'
                f'<select id="reason-{position}" class="reason">'
                '<option value="">Choose a reason</option>'
                '<option value="supports_question">Supports the question</option>'
                '<option value="wrong_vehicle">Wrong vehicle</option>'
                '<option value="wrong_time">Wrong time</option>'
                '<option value="unrelated_scene">Unrelated scene</option>'
                '<option value="insufficient_evidence">Insufficient evidence</option>'
                '<option value="ambiguous">Ambiguous</option>'
                '<option value="other">Other (explain below)</option></select>'
                f'<label for="note-{position}">Rationale (required for reviewed rows)</label>'
                f'<textarea id="note-{position}" class="note" rows="3" '
                'placeholder="Explain what these frames establish, or why they do not support the question."></textarea>'
                '<label class="check"><input type="checkbox" class="inspected"> '
                'I inspected all shown frames for this candidate.</label>'
                '<p class="provenance"></p></div></details>'
            )
        sections.append(
            '<section class="case">'
            f'<h2>{_text(question["question"])}</h2>'
            f'<p>Question {_text(question["question_id"])} · recording {_text(case["recording_id"])} '
            f'· source {_text(case["source"])} · split <strong>{_text(case["split"])}</strong> '
            f'· query frame {_text(question["query_frame"])} · {len(cards)} candidates</p>'
            + ''.join(cards) + ('<p>No candidates in this case.</p>' if not cards else '')
            + '</section>'
        )
    payload = {"pack_sha256": pack["pack_sha256"], "expected": expected, "labels": labels}
    return (_HTML_TOP + ''.join(sections) + '<script id="review-data" type="application/json">'
            + _json(payload) + '</script>' + _HTML_BOTTOM)


_HTML_TOP = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Task 4 evidence relevance worksheet</title>
<style>
* {box-sizing: border-box} body {font: 16px/1.5 system-ui, sans-serif; color: #183044;
 background: #f2f5f7; margin: 0} main {max-width: 1300px; margin: auto; padding: 24px}
h1 {font-size: 1.8rem; margin-top: 0} h2 {font-size: 1.35rem} p {overflow-wrap: anywhere}
.intro,.case {background: white; border: 1px solid #c8d6df; border-radius: 10px; padding: 22px; margin-bottom: 24px}
.notice {background: #fff5da; border-left: 4px solid #ad7300; padding: 12px}
.toolbar {display: flex; flex-wrap: wrap; gap: 16px; align-items: end; margin: 16px 0}
.toolbar label {display: block} input[type=text] {min-width: 260px} input,select,textarea,button {font: inherit}
input[type=text],select,textarea {padding: 8px; border: 1px solid #8195a4; border-radius: 4px}
button {background: #104f7b; color: white; padding: 10px 16px; border: 0; border-radius: 5px; cursor: pointer}
button:focus-visible,summary:focus-visible {outline: 3px solid #cf7b00; outline-offset: 3px}
.candidate {border: 1px solid #bccbd6; border-radius: 6px; margin: 12px 0; padding: 12px}
.candidate>summary {display: flex; justify-content: space-between; gap: 12px; cursor: pointer; font-weight: 600}
.badge {font-size: .85rem; background: #e8eef2; border-radius: 4px; padding: 2px 8px}
.identifier {font-size: .8rem; color: #496272}.facts {font-size: .9rem; overflow-wrap: anywhere}
.facts dt {font-weight: 600}.facts dd {margin: 0 0 6px}.history {margin-bottom: 12px}
.gallery {display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,320px),1fr)); gap: 12px}
figure {margin: 0; background: #edf1f4; padding: 6px} svg {display: block; width: 100%; height: auto}
figcaption {font-size: .85rem; padding: 6px}.controls {display: grid; gap: 8px; margin-top: 12px}
.controls>label {font-weight: 600}.check {font-weight: 400!important}.provenance {font-size: .85rem; color: #496272}
#status {white-space: pre-wrap; padding: 10px; background: #edf1f4} .error {color: #8c1d18}
.small {font-size: .9rem} textarea {width: 100%; resize: vertical} @media(max-width:600px){main{padding:10px}.case,.intro{padding:14px}}
</style></head><body><main><div class="intro">
<h1>Task 4 evidence relevance worksheet</h1>
<p>For each question, inspect candidate evidence and record whether it supports answering that question.
Judge both the vehicle and the time. A correct vehicle at the wrong time can be irrelevant.
Unreviewed candidates stay pending; they are never treated as negative examples.</p>
<p class="notice">Images are local original, unmasked frames. Red rectangles identify a candidate vehicle;
scene candidates have no rectangle. Missing track observations do not prove visual absence.
Sampled images may be insufficient to establish an event: use uncertain and explain the limitation.</p>
<p>This worksheet contains no model judgments, gold answers or suggested relevance grades.
It makes no uploads or network requests. Keep the HTML file beside its packaged images folder.</p>
<div class="toolbar"><div><label for="reviewer">Human reviewer name</label>
<input type="text" id="reviewer" autocomplete="off" placeholder="Your name"></div>
<div><label for="resume">Resume from labels JSONL</label><input type="file" id="resume" accept=".jsonl,application/json"></div>
<button type="button" id="download">Download all labels</button></div>
<label class="check"><input type="checkbox" id="human-reviewer"> I am the named human reviewer and personally
inspected the evidence for every new or changed judgment in this export.</label>
<p class="small">This assertion records human provenance but cannot authenticate your identity.
Unchanged imported judgments retain their existing reviewer and provenance. Downloads include all rows,
including pending ones. Save a download to keep your work; closing this page loses unsaved edits.</p>
<p id="progress" aria-live="polite"></p><p id="status" role="status">Ready. No candidates have been automatically labeled.</p>
</div>
'''


_HTML_BOTTOM = r'''
<script>
"use strict";
(() => {
  const data = JSON.parse(document.getElementById("review-data").textContent);
  const reasons = {
    relevant: ["supports_question", "other"],
    irrelevant: ["wrong_vehicle", "wrong_time", "unrelated_scene", "insufficient_evidence", "other"],
    uncertain: ["ambiguous", "insufficient_evidence", "other"], pending: []
  };
  const keys = ["pack_sha256", "case_id", "candidate_id", "judgment", "reviewer", "reviewer_type",
    "reason", "note", "checked_frame_ids", "reviewed_at"].sort();
  const keyOf = row => JSON.stringify([row.case_id, row.candidate_id]);
  const expected = new Map(data.expected.map(row => [keyOf(row), row]));
  const status = document.getElementById("status");
  const reviewer = document.getElementById("reviewer");
  const assertion = document.getElementById("human-reviewer");
  const cards = Array.from(document.querySelectorAll(".candidate")).map(element => ({
    element, judgment: element.querySelector(".judgment"), reason: element.querySelector(".reason"),
    note: element.querySelector(".note"), inspected: element.querySelector(".inspected"),
    baseline: null, dirty: false
  }));
  function message(text, error = false) {
    status.textContent = text;
    status.classList.toggle("error", error);
  }
  function validateRows(rows) {
    if (!Array.isArray(rows) || rows.length !== expected.size) {
      throw new Error("The draft must contain exactly one row for every candidate in this pack.");
    }
    const seen = new Set();
    for (const row of rows) {
      if (!row || typeof row !== "object" || Array.isArray(row)
          || JSON.stringify(Object.keys(row).sort()) !== JSON.stringify(keys)) {
        throw new Error("Each label must contain exactly the supported label fields.");
      }
      const key = keyOf(row), candidate = expected.get(key);
      if (!candidate || seen.has(key) || row.pack_sha256 !== data.pack_sha256) {
        throw new Error("Draft has a foreign pack, unknown candidate or duplicate row.");
      }
      seen.add(key);
      if (!Object.hasOwn(reasons, row.judgment) || typeof row.reviewer !== "string"
          || typeof row.note !== "string" || !Array.isArray(row.checked_frame_ids)
          || row.checked_frame_ids.some(value => !Number.isSafeInteger(value) || value < 0)
          || new Set(row.checked_frame_ids).size !== row.checked_frame_ids.length) {
        throw new Error("A label contains invalid judgment, text or checked frame IDs.");
      }
      if (row.judgment === "pending") {
        if (row.reviewer !== "" || row.reviewer_type !== null || row.reason !== null
            || row.note !== "" || row.checked_frame_ids.length !== 0 || row.reviewed_at !== null) {
          throw new Error("Pending rows must have empty review fields.");
        }
      } else {
        if (!row.reviewer.trim() || !row.note.trim() || !reasons[row.judgment].includes(row.reason)
            || !["human", "ai_assisted"].includes(row.reviewer_type)
            || typeof row.reviewed_at !== "string"
            || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(row.reviewed_at)
            || !Number.isFinite(Date.parse(row.reviewed_at))
            || row.checked_frame_ids.length !== candidate.frame_ids.length
            || candidate.frame_ids.some(frame => !row.checked_frame_ids.includes(frame))) {
          throw new Error("Reviewed rows need valid provenance, rationale, reason, date and all shown frame IDs.");
        }
      }
    }
    return new Map(rows.map(row => [keyOf(row), row]));
  }
  function update() {
    const counts = {pending: 0, relevant: 0, irrelevant: 0, uncertain: 0};
    for (const card of cards) {
      const judgment = card.judgment.value;
      counts[judgment] += 1;
      card.element.querySelector(".badge").textContent = judgment;
      for (const option of card.reason.options) {
        option.disabled = option.value !== "" && !reasons[judgment].includes(option.value);
      }
      if (card.reason.value && !reasons[judgment].includes(card.reason.value)) card.reason.value = "";
    }
    document.getElementById("progress").textContent = `${cards.length} candidates: `
      + Object.entries(counts).map(([name, count]) => `${count} ${name}`).join(" · ");
  }
  function applyRows(rows) {
    const byKey = validateRows(rows);
    cards.forEach((card, position) => {
      const row = byKey.get(keyOf(data.expected[position]));
      card.baseline = row;
      card.judgment.value = row.judgment;
      card.reason.value = row.reason || "";
      card.note.value = row.note;
      card.inspected.checked = row.judgment !== "pending";
      card.dirty = false;
      card.element.querySelector(".provenance").textContent = row.judgment === "pending" ? "Not reviewed."
        : `Saved reviewer: ${row.reviewer} (${row.reviewer_type}); ${row.reviewed_at}`;
    });
    assertion.checked = false;
    update();
  }
  for (const card of cards) {
    for (const field of [card.judgment, card.reason, card.note, card.inspected]) {
      field.addEventListener("input", () => {card.dirty = true; update();});
      field.addEventListener("change", () => {card.dirty = true; update();});
    }
  }
  document.getElementById("resume").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const rows = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(line => line.trim()).map(JSON.parse);
      validateRows(rows);
      if (cards.some(card => card.dirty) && !confirm("Replace unsaved worksheet edits with this validated draft?")) return;
      applyRows(rows);
      message("Imported complete draft. Existing reviewed rows retain their original provenance.");
    } catch (error) { message(`Draft not loaded: ${error.message}`, true); }
    finally {event.target.value = "";}
  });
  document.getElementById("download").addEventListener("click", () => {
    try {
      const rows = cards.map((card, position) => {
        const baseline = card.baseline;
        if (card.judgment.value === "pending") {
          return {...baseline, judgment: "pending", reviewer: "", reviewer_type: null,
            reason: null, note: "", checked_frame_ids: [], reviewed_at: null};
        }
        if (!card.dirty) return baseline;
        if (!reviewer.value.trim() || !assertion.checked || !card.inspected.checked
            || !card.note.value.trim() || !reasons[card.judgment.value].includes(card.reason.value)) {
          card.element.open = true;
          card.element.scrollIntoView({block: "center"});
          throw new Error("Each new or changed judgment requires your name, human-review assertion, compatible reason, rationale and frame-inspection checkbox.");
        }
        return {...baseline, judgment: card.judgment.value, reviewer: reviewer.value.trim(),
          reviewer_type: "human", reason: card.reason.value, note: card.note.value.trim(),
          checked_frame_ids: data.expected[position].frame_ids.slice(), reviewed_at: new Date().toISOString()};
      });
      validateRows(rows);
      const blob = new Blob([rows.map(row => JSON.stringify(row)).join("\n") + (rows.length ? "\n" : "")],
        {type: "application/x-ndjson;charset=utf-8"});
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = "labels-reviewed.jsonl";
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      message("Download requested. Save labels-reviewed.jsonl, then use the dataset validator before training. Pending rows remain unreviewed.");
    } catch (error) {message(error.message, true);}
  });
  window.addEventListener("beforeunload", event => {
    if (cards.some(card => card.dirty)) {event.preventDefault(); event.returnValue = "";}
  });
  try {applyRows(data.labels);} catch (error) {
    document.getElementById("download").disabled = true;
    message(`Worksheet data invalid: ${error.message}`, true);
  }
})();
</script></main></body></html>
'''
