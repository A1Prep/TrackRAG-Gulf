"""Export composite passenger-run evidence without creating a native Task 1 model condition."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools.dropoff_predict import read_requests
from trackrag.dropoff.contracts import FULL_FIELDS, validate_request, validate_target
from trackrag.dropoff.data import file_sha256, inference_input_sha256
from trackrag.dropoff.passenger import digest


def need(ok, message):
    if not ok:
        raise ValueError(message)


def records(path):
    """Keep the original line hash as well as its decoded contents."""
    return [(sha256(line).hexdigest(), json.loads(line)) for line in path.read_bytes().splitlines() if line.strip()]


def bound_outcome(outcome, request, *, passenger_selected=False):
    need(isinstance(outcome, dict) and outcome.get("request_id") == request["request_id"], "Mismatched outcome ID")
    source = outcome.get("provenance", {})
    if outcome.get("ok") is False:
        return {"ok": False, "error": outcome.get("error"), "method": source.get("method"),
                "saved_provenance": source}
    need(outcome.get("ok") is True, "Unrecognized outcome type")
    frame = request["scene"]["frame"]
    image_sha = request["scene"]["provenance"]["image_sha256"]
    need(source.get("image", {}).get("sha256") == image_sha and file_sha256(frame["image_path"]) == image_sha,
         "Outcome or original image does not match the requested scene")
    need(source.get("request_sha256") == digest(request), "Outcome does not match the exact request")
    portable = inference_input_sha256(request, image_sha)
    if "inference_input_sha256" in source:
        need(source["inference_input_sha256"] == portable, "Outcome has a mismatched portable input binding")
    target = outcome["result"]
    need(FULL_FIELDS.issubset(target), "Incomplete passenger interpretation")
    target = validate_target(target, list(target), frame)
    method = source.get("method")
    record = {"ok": True, "method": method, "result": target, "request_id": request["request_id"],
              "inference_input_sha256": portable, "image_sha256": image_sha, "source_scene": request["scene"]["provenance"]}
    if passenger_selected:
        need(method == "explicit_passenger_region" and source.get("independent_model_proposal") is False,
             "Passenger marking lacks its manual-intervention declaration")
    else:
        need(method == "qwen_landmark_region" and source.get("independent_model_proposal") is not False,
             "An assisted or unknown method was supplied as a model attempt")
        resolver = source.get("resolver", {})
        raw = resolver.get("raw_text")
        need(isinstance(raw, str) and resolver.get("request_sha256") == source["request_sha256"],
             "Missing or mismatched raw landmark response")
        need((resolver.get("image_sha256") or resolver.get("image", {}).get("sha256")) == image_sha,
             "Raw landmark response belongs to another image")
        model = resolver.get("backend_details", {})
        need(isinstance(resolver.get("model_id"), str) and isinstance(resolver.get("revision"), str)
             and model.get("model_id") == resolver["model_id"] and model.get("revision") == resolver["revision"]
             and model.get("runtime") == "streamed", "Incomplete local resolver runtime identity")
        messages_sha = sha256(json.dumps(resolver["messages"], sort_keys=True).encode()).hexdigest()
        need(resolver.get("messages_sha256") == messages_sha, "Resolver prompt differs from its hash")
        record["landmark_raw_text_sha256"] = sha256(raw.encode()).hexdigest()
        if "visual_verification" in source:
            inspection_raw = source["visual_verification"].get("raw_text")
            need(isinstance(inspection_raw, str), "Missing visual-verifier response")
            record["verification_raw_text_sha256"] = sha256(inspection_raw.encode()).hexdigest()
    # Full original provenance includes weights, prompts, raw response, optional visual check and road evidence.
    record["saved_provenance"] = source
    return record


def evidence_head():
    return {"schema_version": 1, "kind": "passenger_composite_evidence",
            "native_task1_condition": False, "scoring_references_read": False}


def batch(run_dir, requests_path, replay_path=None):
    settings_path, manifest_path, outcomes_path = (run_dir / name for name in ("settings.json", "run.json", "outcomes.jsonl"))
    settings = json.loads(settings_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    need(settings == {k: v for k, v in manifest.items() if k not in ("errors", "outcomes_sha256")},
         "Saved run settings and completion manifest disagree")
    input_hash = file_sha256(requests_path)
    need(manifest.get("requests_sha256") == input_hash and manifest.get("outcomes_sha256") == file_sha256(outcomes_path)
         and manifest.get("reference_access") is False, "Saved inputs, outcomes or reference-access record do not match")
    requests = {r["request_id"]: r for r in read_requests(requests_path)}
    raws = records(outcomes_path)
    need(len(raws) == len(requests) == manifest["requested"] and
         {v["request_id"] for _, v in raws} == requests.keys(), "Missing, extra or duplicate run outcomes")
    kind = settings["resolver_execution"]
    saved = {}
    if kind == "saved_decisions":
        need(replay_path is not None and file_sha256(replay_path) == manifest["saved_decisions_sha256"],
             "Supply the exact saved raw landmark batch named by this run")
        rows = records(replay_path)
        saved = {row["request_id"]: (h, row) for h, row in rows}
        need(len(saved) == len(rows) and saved.keys() == requests.keys(), "Saved raw landmark cohort differs")
    else:
        need(kind == "live" and replay_path is None and manifest["saved_decisions_sha256"] is None,
             "Invalid declaration of live or saved resolver execution")
    events = []
    for raw_sha, outcome in raws:
        key = outcome["request_id"]
        request = requests[key]
        item = bound_outcome(outcome, request)
        need(outcome["provenance"].get("resolver_execution") == kind, "Outcome and batch disagree on resolver execution")
        if outcome["ok"]:
            resolver = outcome["provenance"]["resolver"]
            cfg = settings["config"]
            need(resolver["model_id"] == cfg["model_id"] and resolver["revision"] == cfg["revision"]
                 and bool(resolver.get("adapter")) == bool(cfg.get("adapter_dir"))
                 and outcome["provenance"]["geometry_sha256"] == settings["geometry_sha256"],
                 "Model or geometry in the recorded answer differs from the batch")
            if saved:
                need(saved[key][1] == resolver, "Embedded landmark receipt differs from the original raw decision")
        item.update(request_id=key, phase="model_after_prior_dialogue" if request["dialogue"] else "initial_model",
                    input_mode=request["input_mode"], raw_outcome_line_sha256=raw_sha,
                    raw_saved_decision_line_sha256=saved[key][0] if saved else None)
        events.append(item)
    need(sum(not e["ok"] for e in events) == manifest["errors"], "Completed run error count disagrees with outcomes")
    return dict(evidence_head(), origin="passenger_batch", settings=settings, run_sha256=file_sha256(manifest_path),
                settings_sha256=file_sha256(settings_path), requests_sha256=input_hash,
                outcomes_sha256=manifest["outcomes_sha256"], saved_decisions_sha256=manifest["saved_decisions_sha256"],
                events=events)


def session(journal_path):
    raw = records(journal_path)
    need(raw and raw[0][1]["action"] == "opened", "Session lacks its opening record")
    scene = raw[0][1]["detail"]["scene"]
    request, previous = None, {}
    events = []
    for line_sha, event in raw:
        action, detail, view = event["action"], event["detail"], event["view"]
        frame = scene["frame"]
        need((view["recording_id"], view["frame_id"], view["image_sha256"]) ==
             (frame["recording_id"], frame["frame_id"], scene["provenance"]["image_sha256"]), "Session changed scene")
        item = {"action": action, "timestamp": event["timestamp"], "request_id": view["request_id"],
                "proposal_id": view["proposal_id"], "selected_by": view["selected_by"],
                "state": view["state"], "raw_event_line_sha256": line_sha}
        if action in ("submitted", "passenger_marked"):
            manual = action == "passenger_marked"
            if not manual:
                # The application can journal a rejected submission before it reaches the model.
                # Keep that error in the export even if the submitted text failed validation.
                request = detail["request"] if detail["outcome"] is None else validate_request(detail["request"])
                need(isinstance(request, dict), "Submitted request is missing")
                need(request["scene"] == scene, "Submitted request belongs to another scene")
            need(request is not None and request["request_id"] == view["request_id"], "Mismatched interaction request")
            item["phase"] = ("passenger_marked" if manual else
                             "model_after_prior_dialogue" if request["dialogue"] else "initial_model")
            item["submitted_text"] = request["text"]
            item["input_mode"] = request["input_mode"]
            item["original_transcript"] = request.get("original_transcript")
            if manual:
                item["requested_box"] = detail["requested_box"]
            else:
                item["speech"] = detail.get("speech")
                if item["speech"] and detail["outcome"] is not None:
                    audio_sha = item["speech"].get("audio_sha256")
                    need(isinstance(audio_sha, str) and len(audio_sha) == 64 and
                         all(c in "0123456789abcdef" for c in audio_sha), "Invalid original audio digest")
                    wav = journal_path.parent / "audio" / (audio_sha + ".wav")
                    need(wav.is_file() and file_sha256(wav) == audio_sha and
                         item["speech"].get("transcript") == request.get("original_transcript"),
                         "Saved speech receipt differs from the original WAV or recognized transcript")
                    item["original_wav_verified"] = True
            if detail["outcome"] is None:
                need(view["state"] == "error", "Missing outcome for a successful passenger turn")
                item["outcome"] = {"ok": False, "error": view["error"], "adapter_outcome_missing": True}
            else:
                item["outcome"] = bound_outcome(detail["outcome"], request, passenger_selected=manual)
            if item["outcome"]["ok"]:
                need(view["selected_by"] == ("passenger" if manual else "model") and
                     view["result"] == item["outcome"]["result"] and view["state"] == view["result"]["status"],
                     "Saved view changed the grounding result or who selected it")
            else:
                need(view["state"] == "error", "Failed model call appears as successful interaction")
        elif action == "confirmed":
            need(previous.get("state") == "proposed" and view["state"] == "confirmed" and
                 (previous["request_id"], previous["proposal_id"], previous["selected_by"], previous["result"]) ==
                 (view["request_id"], detail["proposal_id"], detail["selected_by"], view["result"]) and
                 (view["proposal_id"], view["selected_by"]) == (detail["proposal_id"], detail["selected_by"]),
                 "Confirmation differs from the currently offered proposal")
            item["phase"] = "confirmed_previous_selection"
        else:
            need(action in ("opened", "cancelled", "confirmation_blocked"), "Unsupported interaction action")
            item["phase"] = action
        events.append(item)
        previous = view
    return dict(evidence_head(), origin="passenger_interaction_journal", journal_sha256=file_sha256(journal_path),
                application_source_commit_recorded_by_journal=False,
                source_scene=scene["provenance"], events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    b = modes.add_parser("batch", help="Export a completed live or saved-landmark batch")
    b.add_argument("--run", required=True, type=Path)
    b.add_argument("--requests", required=True, type=Path)
    b.add_argument("--replay", type=Path)
    j = modes.add_parser("session", help="Export submitted, manually marked, and confirmed turns separately")
    j.add_argument("--journal", required=True, type=Path)
    for sub in (b, j):
        sub.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = batch(args.run, args.requests, args.replay) if args.mode == "batch" else session(args.journal)
    with args.out.open("x", encoding="utf8", newline="\n") as target:
        json.dump(data, target, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
