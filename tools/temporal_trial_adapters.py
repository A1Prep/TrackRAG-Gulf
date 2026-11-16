"""Semantic selection over observed track histories and exact scene snapshots.

This development adapter uses the configured Gemini model for selection and
grounded answering. It never receives benchmark answers or future observations.
"""

import base64
import io
import json
import os
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageDraw

from tools.emt_trial_adapters import (
    _hosted_image, build_index, load_annotated_tracks, load_questions, open_recording,
)
from tools.trial_gemini import request_json


def history_facts(rows, first_recording_frame):
    first, last = rows[0], rows[-1]
    a, b = first["bbox_xyxy"], last["bbox_xyxy"]
    gaps = [[left["frame_id"] + 1, right["frame_id"] - 1] for left, right in zip(rows, rows[1:]) if right["frame_id"] > left["frame_id"] + 1]
    return {
        "class_name": first["class_name"], "first_observed_frame": first["frame_id"],
        "last_observed_frame": last["frame_id"],
        "annotated_at_first_available_frame": first["frame_id"] == first_recording_frame,
        "first_visibility_or_entry": "not established by annotation timing",
        "first_box": first["bbox_xyxy"], "last_box": last["bbox_xyxy"],
        "first_size_wh_pixels": [a[2]-a[0], a[3]-a[1]],
        "last_size_wh_pixels": [b[2]-b[0], b[3]-b[1]],
        "center_displacement_xy_pixels": [(b[0]+b[2]-a[0]-a[2])/2, (b[1]+b[3]-a[1]-a[3])/2],
        "observed_frames": len(rows), "annotation_gaps": gaps,
    }


def candidates_at_query(question, index):
    recording = index["recording"]
    if question["recording_id"] != recording.recording_id:
        raise ValueError("Question and index belong to different recordings")
    first_frame = next(recording.iter_frames())["frame_id"]
    candidates = []
    for frame in sorted({first_frame, question["query_frame"]}):
        candidates.append({"track_id": None, "start_frame": frame, "end_frame": frame, "kind": "full scene"})
    for track_id, all_rows in sorted(index["tracks"].items()):
        rows = [row for row in all_rows if row["frame_id"] <= question["query_frame"]]
        if not rows:
            continue
        history = history_facts(rows, first_frame)
        intervals = {(rows[0]["frame_id"], rows[-1]["frame_id"])}
        for start in range(rows[0]["frame_id"], rows[-1]["frame_id"] + 1, 15):
            window = [row for row in rows if start <= row["frame_id"] < start + 30]
            if window:
                intervals.add((window[0]["frame_id"], window[-1]["frame_id"]))
        for start, end in sorted(intervals):
            window = [row for row in rows if start <= row["frame_id"] <= end]
            candidates.append({
                "track_id": track_id, "start_frame": start, "end_frame": end,
                "history": history, "window": history_facts(window, first_frame),
            })
    # A sequential C6 was unrelated to gallery label T6. Bind the selection token
    # directly to its identity and observed window so those numbers cannot alias.
    return [{"candidate_id": (
        f"scene-F{candidate['start_frame']}" if candidate["track_id"] is None else
        f"T{candidate['track_id']}-F{candidate['start_frame']}-F{candidate['end_frame']}"
    ), **candidate} for candidate in candidates]


def selection_context(candidates):
    """Share track history while retaining each window's distinct observations."""
    scenes, tracks = [], {}
    for candidate in candidates:
        track_id = candidate["track_id"]
        window = {key: candidate[key] for key in ("candidate_id", "start_frame", "end_frame")}
        if track_id is None:
            scenes.append(window)
            continue
        # Only the selector's textual summaries are rounded. Stored observations
        # and exported evidence retain the original subpixel coordinates.
        history = dict(candidate["history"])
        for key in ("first_box", "last_box"):
            history[key] = [round(value, 1) for value in history[key]]
            window[key] = [round(value, 1) for value in candidate["window"][key]]
        window["observed_frames"] = candidate["window"]["observed_frames"]
        if candidate["window"]["annotation_gaps"]:
            window["annotation_gaps"] = candidate["window"]["annotation_gaps"]
        track = tracks.setdefault(track_id, {"track_id": track_id, "history": history, "windows": []})
        track["windows"].append(window)
    return {"full_scenes": scenes, "tracks": list(tracks.values())}


def selector_views(question, index):
    """Initial/current masked scenes make appearance-based selection possible."""
    recording = index["recording"]
    first_frame = next(recording.iter_frames())["frame_id"]
    parts, sent = [], []
    directory = Path(os.environ.get("TRACKRAG_TRIAL_CACHE", "runs/temporal-request-cache")) / "selector-images"
    directory.mkdir(parents=True, exist_ok=True)
    for frame_id in sorted({first_frame, question["query_frame"]}):
        frame = recording.get_frame(frame_id)
        rows = [row for observations in index["tracks"].values() for row in observations if row["frame_id"] == frame_id]
        payload = _hosted_image(frame["image_path"], rows)
        with Image.open(io.BytesIO(payload)) as processed:
            marked = processed.convert("RGB")
        draw = ImageDraw.Draw(marked)
        sx, sy = marked.width / frame["width"], marked.height / frame["height"]
        prominent = sorted(rows, key=lambda row: (row["bbox_xyxy"][2]-row["bbox_xyxy"][0]) * (row["bbox_xyxy"][3]-row["bbox_xyxy"][1]), reverse=True)[:24]
        for row in prominent:
            left, top, right, bottom = row["bbox_xyxy"]
            draw.rectangle((left*sx, top*sy, right*sx, bottom*sy), outline=(0, 255, 255), width=1)
            label = f"T{row['track_id']}"
            x, y = left*sx, max(0, top*sy-12)
            draw.rectangle((x, y, x+7*len(label), y+12), fill="black")
            draw.text((x, y), label, fill="white")
        output = io.BytesIO()
        marked.save(output, format="JPEG", quality=90)
        payload = output.getvalue()
        name = sha256(payload).hexdigest() + ".jpg"
        (directory / name).write_bytes(payload)
        parts.extend([
            {"text": f"Masked full scene at frame {frame_id}; T labels denote annotated track IDs, only the 24 largest labeled objects are marked."},
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(payload).decode("ascii")}},
        ])
        sent.append({"frame_id": frame_id, "image": name, "width": marked.width, "height": marked.height})
    return parts, sent


def identity_views(question, index):
    """Show each observed identity at its largest available, query-bounded view."""
    recording = index["recording"]
    representatives = []
    by_frame = {}
    for track_id, observations in sorted(index["tracks"].items()):
        rows = [row for row in observations if row["frame_id"] <= question["query_frame"]]
        if not rows:
            continue
        for row in rows:
            by_frame.setdefault(row["frame_id"], []).append(row)
        representative = max(rows, key=lambda row: (
            (row["bbox_xyxy"][2] - row["bbox_xyxy"][0]) * (row["bbox_xyxy"][3] - row["bbox_xyxy"][1]),
            row["frame_id"],
        ))
        representatives.append((track_id, representative))
    parts, sent, processed = [], [], {}
    directory = Path(os.environ.get("TRACKRAG_TRIAL_CACHE", "runs/temporal-request-cache")) / "selector-images"
    directory.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(representatives), 16):
        entries = representatives[offset:offset + 16]
        sheet = Image.new("RGB", (896, 160 * ((len(entries) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        identities = []
        for i, (track_id, row) in enumerate(entries):
            frame_id = row["frame_id"]
            if frame_id not in processed:
                frame = recording.get_frame(frame_id)
                payload = _hosted_image(frame["image_path"], by_frame[frame_id], max_size=None)
                with Image.open(io.BytesIO(payload)) as image:
                    processed[frame_id] = image.convert("RGB")
            image = processed[frame_id]
            left, top, right, bottom = row["bbox_xyxy"]
            pad = max(3, (bottom - top) * 0.08)
            crop = image.crop((max(0, int(left - pad)), max(0, int(top - pad)),
                               min(image.width, int(right + pad)), min(image.height, int(bottom + pad))))
            crop.thumbnail((208, 124))
            x, y = (i % 4) * 224, (i // 4) * 160
            sheet.paste(crop, (x + 8, y + 29))
            draw.text((x + 6, y + 3), f"T{track_id} @ frame {frame_id}", fill="black")
            draw.text((x + 6, y + 15), f"Label: {row['class_name']}", fill="black")
            identities.append({"track_id": track_id, "frame_id": frame_id})
        output = io.BytesIO()
        sheet.save(output, format="JPEG", quality=92)
        payload = output.getvalue()
        name = sha256(payload).hexdigest() + ".jpg"
        (directory / name).write_bytes(payload)
        parts.extend([
            {"text": "Identity reference gallery. Each tile is a privacy-masked crop at the stated observed frame, not necessarily the query frame. Annotation categories are broader than everyday body-shape descriptions: an SUV can be labeled Car. Preserve annotation labels and distinguish visible subtypes from the pixels. " + json.dumps(identities)},
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(payload).decode("ascii")}},
        ])
        sent.append({"image": name, "identities": identities})
    return parts, sent


def rank_evidence(question, index, k=3):
    candidates = candidates_at_query(question, index)
    # Track history is shared by its windows. Send it once, rather than repeating
    # the same history for every overlapping candidate and spending the quota.
    prompt = (
        "Select up to three evidence candidates for a driving-footage question. Do not answer it. "
        "Use meaning, object class and time constraints, including paraphrases. "
        "The boxes are left/top/right/bottom original-image pixels. Compare observed changes when relevant. "
        "Use the supplied initial/current scene images to identify appearance and distinguish objects sharing a class. "
        "The identity gallery provides a separate labeled crop for every observed track. Verify the requested body's shape against those crops before selecting a track; do not substitute a similarly coloured SUV for a pickup or an open truck for a closed truck. "
        "Annotation classes are broad: Car does not exclude a visually supported SUV subtype. Keep the annotation category separate from visible body shape. Gallery times are explicit; use scene snapshots for current visibility. "
        "T labels map visible objects to track IDs. Other objects remain in the scene; missing labels do not imply visual absence. "
        "Candidate IDs contain that same T identity and their observed F frame endpoints. Match a gallery label to the identical T prefix before selecting its window; different windows of one track are not different objects. Scene candidates have no track identity. "
        "Choose full-scene snapshots for current presence/absence, signs or lighting. "
        "For temporal questions prefer sufficient history with a clear later view, and context distinguishing earlier objects. "
        "Also extract requested_frame_ids: exact observation frames requested by the question, with at most two comparison endpoints. "
        "For a comparison against the beginning, use the first available frame as its opening endpoint. "
        "For general history questions without specific endpoints, return []. Do not confuse track IDs with frame IDs. "
        "Never request a frame after the query. These frame IDs will refine tracked intervals when those observations exist. "
        "A first annotation is not proof of physical entry; gaps may be occlusion or missing labels. "
        "Candidate histories contain only observations available by query time. Never invent candidate IDs. "
        f"Query frame {question['query_frame']}. Question: {question['question']}\n"
        f"Candidates: {json.dumps(selection_context(candidates))}"
    )
    schema = {"type": "OBJECT", "required": ["selected", "requested_frame_ids"], "properties": {
        "selected": {"type": "ARRAY", "maxItems": k, "items": {"type": "STRING", "enum": [row["candidate_id"] for row in candidates]}},
        "requested_frame_ids": {"type": "ARRAY", "maxItems": 2, "items": {"type": "INTEGER"}},
    }}
    lookup = {row["candidate_id"]: row for row in candidates}
    def validate(response):
        chosen = response.get("selected")
        if (not isinstance(chosen, list) or len(chosen) > k
                or any(not isinstance(value, str) or value not in lookup for value in chosen)
                or len(set(chosen)) != len(chosen)):
            raise ValueError("Selector returned invalid or duplicate evidence candidates")
        frames = response.get("requested_frame_ids")
        first = next(index["recording"].iter_frames())["frame_id"]
        if (not isinstance(frames, list) or len(frames) > 2
                or any(type(frame) is not int or not first <= frame <= question["query_frame"] for frame in frames)
                or len(set(frames)) != len(frames)):
            raise ValueError("Selector returned invalid requested frame IDs or future frames")
        for frame in frames:
            index["recording"].get_frame(frame)
    views, sent = selector_views(question, index)
    gallery, gallery_sent = identity_views(question, index)
    selected = request_json([{"text": prompt}, *views, *gallery], schema, "evidence selection", metadata={
        "question_id": question["question_id"], "recording_id": question["recording_id"],
        "query_frame": question["query_frame"], "selection_images": sent,
        "identity_images": gallery_sent,
    }, validate=validate, repair_once=True)
    validate(selected)
    chosen = selected["selected"]
    ranked = []
    seen = set()
    requested = sorted(selected["requested_frame_ids"])
    for candidate_id in chosen:
        candidate = lookup[candidate_id]
        start, end = candidate["start_frame"], candidate["end_frame"]
        if candidate["track_id"] is not None and requested:
            observed = {row["frame_id"] for row in index["tracks"][candidate["track_id"]] if row["frame_id"] <= question["query_frame"]}
            if all(frame in observed for frame in requested):
                start, end = requested[0], requested[-1]
        identity = (candidate["track_id"], start, end)
        if identity in seen:
            continue
        seen.add(identity)
        rank = len(ranked) + 1
        ranked.append({
            "recording_id": question["recording_id"], "source": "annotated",
            "track_id": candidate["track_id"], "start_frame": start,
            "end_frame": end, "rank": rank, "score": 1 / rank,
        })
    return ranked


def _attention_image(path, observations, box):
    # Apply the same masks at original resolution before either view is resized.
    # A crop taken after global downscaling had discarded small-object detail.
    payload = _hosted_image(path, observations, max_size=None)
    if box is None:
        return _hosted_image(path, observations)
    with Image.open(io.BytesIO(payload)) as processed:
        left, top, right, bottom = box
        padding = max(8, (bottom - top) * 0.2)
        region = (
            max(0, int(left - padding)), max(0, int(top - padding)),
            min(processed.width, int(right + padding)), min(processed.height, int(bottom + padding)),
        )
        crop = processed.crop(region)
        crop.thumbnail((360, 220))
        if crop.width < 220 and crop.height < 160:
            factor = min(220 / crop.width, 160 / crop.height)
            crop = crop.resize((int(crop.width * factor), int(crop.height * factor)))
        full = processed.copy()
        full.thumbnail((1024, 576))
        canvas = Image.new("RGB", (max(360, full.width), full.height + 250), "white")
        canvas.paste(full, (0, 0))
        ImageDraw.Draw(canvas).text((8, full.height + 8), "Selected road user from the same privacy-masked source", fill="black")
        canvas.paste(crop, (8, full.height + 28))
        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=85)
        return output.getvalue()


def answer_question(question, evidence, run_dir):
    if not evidence:
        return {"answer_status": "insufficient_evidence", "answer": "No evidence was selected.", "cited_evidence_ids": []}
    observations = [row for row in load_questions(run_dir / "tracks.jsonl") if row["frame_id"] <= question["query_frame"]]
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    first_frame = metadata["first_frame_id"]
    facts = []
    for item in evidence:
        track_id = item["interval"]["track_id"]
        start, end = item["interval"]["start_frame"], item["interval"]["end_frame"]
        rows = [row for row in observations if row["track_id"] == track_id and start <= row["frame_id"] <= end]
        history = history_facts(rows, first_frame) if rows else None
        if history is not None and len(rows) > 1:
            dx, dy = history["center_displacement_xy_pixels"]
            history["net_image_motion"] = {
                "horizontal": "left" if dx < 0 else "right" if dx > 0 else "unchanged",
                "vertical": "up" if dy < 0 else "down" if dy > 0 else "unchanged",
                "dominant_axis": "horizontal" if abs(dx) > abs(dy) else "vertical" if abs(dy) > abs(dx) else "equal",
            }
        facts.append({"evidence_id": item["evidence_id"], "interval": item["interval"], "history": history})
    prompt = (
        "Answer the question from these cited scene images, selected object views and observed track facts. "
        "Images include a full scene and, for tracked objects, an enlarged view of the same processed image. "
        "Never turn image-relative movement into physical speed, intent or road maneuvers. "
        "Image growth alone does not prove physical approach or acceleration. Use the supplied size fields for width/height; box coordinates are not dimensions. "
        "For direction questions report the computed net_image_motion, including the dominant image axis and any relevant secondary movement. "
        "The query frame is the latest allowed time, not a requirement that every object be observed at that exact frame. Earlier history remains usable. "
        "Preserve identity across time. Similar-looking objects with different track IDs remain distinct; do not assign an ID to an incidental scene object without visual support. "
        "Annotation classes are broad: Car can include an SUV. Keep annotation category and visible subtype separate. Match the requested body shape, not just colour. "
        "First observed means first annotation, not proof of first visibility or entry. Annotation gaps do not prove absence. "
        "An object first annotated inside the frame does not establish a boundary crossing. "
        "A false annotated_at_first_available_frame value does not establish that the object was visually absent. "
        "A scene snapshot is the evidence for what is visibly present at that moment. "
        "A clear full-scene snapshot can support an answered statement that no requested object is clearly visible, citing that scene. This differs from an unseen interior, hidden action, or absence inferred only from missing annotations. "
        "Visible colour or clarity changes can be described at the resolution actually shown. A broad visible colour need not establish an exact paint shade. Privacy masks do not make unmasked body panels unknowable. "
        "Compare lit and shaded unmasked body panels before naming a colour; bright reflections alone do not establish paint colour. When glare or processing makes neighbouring colour categories ambiguous, give the supported broad appearance and briefly state that uncertainty rather than choosing an overconfident label or withholding all visible colour. This can still be an answered observation. "
        "Comparing clarity or prominence in supplied early/late views does not claim an object's first-ever appearance or a boundary crossing. "
        "Keep the answer limited to the requested information and language. For sign questions give the readable name needed to answer; do not add an unrequested exact transcription or translation in another script. "
        "Use one or two concise sentences with only the requested observation and necessary qualification. Relative clarity does not require a claim of complete visibility; masked or clipped portions remain unavailable. "
        "An explicit answer format in the question takes priority over that default sentence style: follow requests for bare copied text, a name, a list or a single word without introductory wording. Structured citations remain separate. "
        "Give the supported identification or observation with a clear qualification when the exact event is uncertain. "
        "If a necessary fact remains unknowable, return insufficient_evidence and explain what is missing. "
        "Even in an insufficient_evidence answer, include the useful supported observation or identification and visible appearance when available. "
        "Failure to see a hidden action does not establish that it is not happening. Distinguish scene absence from an unobservable state. "
        "Always write a meaningful explanation. Separately list any parts of the question that the evidence cannot establish in unverified_parts. "
        "When identification is supported but the exact event is not, answer the identification and put the event uncertainty in unverified_parts. "
        "Before finalizing, check that event verbs such as entering or exiting are established by visual evidence rather than annotation timing alone. "
        "Cite only supplied evidence IDs for answered results; use [] for insufficient_evidence. "
        "Some image regions are masked for privacy. Do not guess masked details. "
        f"Question at frame {question['query_frame']}: {question['question']}\nObserved facts: {json.dumps(facts)}"
    )
    parts = [{"text": prompt}]
    hosted = run_dir / "hosted" / sha256(question["question_id"].encode()).hexdigest()
    hosted.mkdir(parents=True, exist_ok=True)
    sent = []
    for item in evidence:
        for frame in item["images"]:
            rows = [row for row in observations if row["frame_id"] == frame["frame_id"]]
            image = _attention_image(run_dir / frame["image_path"], rows, frame["bbox_xyxy"])
            name = f"{item['evidence_id']}-{frame['frame_id']}.jpg"
            (hosted / name).write_bytes(image)
            parts.extend([
                {"text": f"Evidence {item['evidence_id']}, frame {frame['frame_id']}, track {item['interval']['track_id']}."},
                {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(image).decode("ascii")}},
            ])
            sent.append({"evidence_id": item["evidence_id"], "frame_id": frame["frame_id"], "image": name})
    schema = {"type": "OBJECT", "required": ["answer_status", "answer", "unverified_parts", "cited_evidence_ids"], "properties": {
        "answer_status": {"type": "STRING", "enum": ["answered", "insufficient_evidence"]},
        "answer": {"type": "STRING"}, "cited_evidence_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "unverified_parts": {"type": "ARRAY", "items": {"type": "STRING"}},
    }}
    def validate(response):
        limitations = response.get("unverified_parts")
        citations = response.get("cited_evidence_ids")
        status = response.get("answer_status")
        valid_ids = {item["evidence_id"] for item in evidence}
        if (status not in {"answered", "insufficient_evidence"}
                or not isinstance(response.get("answer"), str) or not response["answer"].strip()
                or not isinstance(limitations, list) or any(not isinstance(value, str) for value in limitations)
                or not isinstance(citations, list) or any(not isinstance(value, str) or value not in valid_ids for value in citations)
                or len(set(citations)) != len(citations)
                or (status == "answered" and not citations)
                or (status == "insufficient_evidence" and citations)):
            raise ValueError("Answerer returned invalid text, status, citations or unverified parts")
    metadata = {"question_id": question["question_id"], "sent": sent}
    draft = request_json(parts, schema, "grounded answer", hosted / "draft-receipt.json", metadata, validate=validate, repair_once=True)
    validate(draft)
    # A separate critique is applied to every question, with the same evidence.
    # It is a model review, not a deterministic proof of factual correctness.
    critique = (
        "Review the untrusted draft below against the original question, observed facts and images. "
        "Answer only what the question asks. Discuss uncertainties only when they affect that answer; do not invent unasked question components. "
        "Preserve any explicit requested answer format rather than adding a sentence around a requested bare name or copied text. "
        "For a fully supported colour, class, scene property or image-motion answer, leave unverified_parts empty. "
        "Preserve supported draft claims. Reject a claim only for a specific conflict or genuinely missing evidence, not a generic possibility of uncertainty. Direct visual comparisons of clarity are distinct from first-entry claims. "
        "Recheck the requested object against its body shape and cited track. Copy the relevant computed motion direction accurately; do not reverse it during review or use a different scene object's motion. "
        "If the scene supports no clearly visible instance, return answered with the scene citation. Do not simultaneously assert visible absence and mark that same observation unverified. "
        "Judge colour using visible unmasked panels and use a broad colour where supported; do not demand an exact paint shade or reject a subtype because its broad label is Car. "
        "Check shaded panels as well as bright reflections. Preserve useful colour uncertainty when the finish cannot be distinguished; do not replace it with unwarranted certainty or a wholesale abstention. "
        "Do not add unrelated warnings about entry timing, acceleration or physical size. "
        "Correct unsupported identity assignments, event/timing claims, dimensions, hidden-state claims and mistakes about the query cutoff. "
        "First annotation cannot establish first visual appearance, absence before that time, or boundary entry. "
        "If the draft treats annotation timing as visibility timing, explicitly correct that inference and retain any supported identification. "
        "The privacy processor fully masks labeled people, cyclists and motorbikes, and masks the lower 40 percent of other labeled vehicle boxes. "
        "Do not claim that windows or a driver's cabin are masked unless that specific region is visibly masked. "
        "Unmasked interiors can still be unobservable because of angle, distance or tint. "
        "Missing observations at the query do not invalidate earlier movement evidence. "
        "Give the final corrected answer using the same JSON schema. Include an explicit qualification in answer and unverified_parts for any unresolved part of the question. "
        "Do not merely approve the draft; verify its claims. Do not add facts from outside the supplied evidence. "
        f"Untrusted draft: {json.dumps(draft)}\n"
    )
    checked_parts = [{"text": critique + parts[0]["text"]}, *parts[1:]]
    response = request_json(checked_parts, schema, "grounding review", hosted / "receipt.json", metadata, validate=validate, repair_once=True)
    validate(response)
    limitations = response["unverified_parts"]
    return {
        "answer_status": response["answer_status"],
        "answer": response["answer"].strip() + (" Unverified details: " + "; ".join(limitations) if limitations else ""),
        "cited_evidence_ids": response["cited_evidence_ids"],
    }
