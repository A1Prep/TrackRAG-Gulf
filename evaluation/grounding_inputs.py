r"""Turn the team's files, in their own formats, into inputs for ``python -m evaluation.grounding``, and freeze the
evaluation protocol. Each command reads the given files and writes new files only.

    python -m evaluation.grounding_inputs task5-cases --cases test-cases.jsonl --intended intended.jsonl \
        --scenes scenes/ --out-dir refs/
    python -m evaluation.grounding_inputs task5-run --run <tools.dropoff_predict output folder> \
        --references refs/references.jsonl --scenes scenes/ --transcript typed --out outputs-trained-direct.jsonl
    python -m evaluation.grounding_inputs protocol --references refs/references.jsonl [--replies replies.jsonl] \
        [--voice voice.jsonl] --evaluator-commit <commit> --responder-rules "<text>" --out protocol.json

task5-cases reads Task 2's final-test cases in the case schema of trackrag/dropoff/data.py (partition "test"),
validated by that module's own loader, and Task 3's record of each needs_clarification case:
    {"case_id", "category": "ambiguous" | "missing_reference", "intended_mask": path | null,
     "intended_mask_sha256": str | null}
An ambiguous case needs its intended mask, drawn before any proposal was seen; paths are relative to that file.

task5-run reads one run folder of tools/dropoff_predict.py (run.json, resolved-requests.jsonl, model-outcomes.jsonl,
receipts.jsonl). One run is one model and one transcript kind, and every request_id must be a case_id. Each request's
frame, image and predicted observations are checked against the case's Task 1 scene. The Ollama route is a separate
baseline, not the pretrained condition, and is refused.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluation.grounding import (PROTOCOL_DEFAULTS, GroundingError, _require, _sha256_bytes, _text,
                                  from_task5_outcome, instruction_sha256, load_references, load_scenes, read_json,
                                  read_jsonl, sha256_file, text_sha256)

TASK5_CONDITIONS = {"pretrained": "pretrained", "fine_tuned": "trained"}
TASK5_RUN_FILES = (("resolved-requests.jsonl", "resolved_requests_sha256"), ("model-outcomes.jsonl", "outcomes_sha256"),
                   ("receipts.jsonl", "receipts_sha256"))
# Fields of GroundingConfig (trackrag/dropoff/model.py) that change what the model does; paths and devices do not.
TASK5_SETTINGS = ("model_id", "revision", "dtype", "quantization", "allow_annotated", "max_new_tokens",
                  "max_input_tokens", "max_time_s", "min_pixels", "max_pixels", "max_image_bytes",
                  "max_original_pixels")
TASK5_CONTEXT_POLICY = "task5_interpret: current frame image, its predicted observations and same-frame dialogue"
CATEGORY_OF_STATUS = {"proposed": "clear", "insufficient_visual_evidence": "missing_reference"}


def _track_keys(tracks) -> list:
    return sorted((t["track_id"], t["frame_id"], t["class_name"], tuple(round(float(v), 6) for v in t["bbox_xyxy"]))
                  for t in tracks)


def scene_problems(scene: dict, frame: dict, tracks: list, image_sha256: str | None) -> list[str]:
    """What differs between a Task 1 scene and the frame, image and observations a Task 5 request carried."""
    model_input, problems = scene["model_input"], []
    if (model_input["recording_id"], model_input["frame_id"]) != (frame["recording_id"], frame["frame_id"]):
        problems.append("recording or frame differs")
    if (model_input["width"], model_input["height"]) != (frame["width"], frame["height"]):
        problems.append("image size differs")
    if image_sha256 is not None and image_sha256 != model_input["image_sha256"]:
        problems.append("image bytes differ")
    if _track_keys(tracks) != _track_keys(model_input["observations"]):
        problems.append("predicted observations differ (another prediction run or profile)")
    return problems


def find_scene(scenes: dict, frame: dict, tracks: list, image_sha256: str | None) -> tuple[dict | None, list[str]]:
    candidates = [s for s in scenes.values() if (s["model_input"]["recording_id"], s["model_input"]["frame_id"])
                  == (frame["recording_id"], frame["frame_id"])]
    if not candidates:
        return None, [f"no scene file for {frame['recording_id']} frame {frame['frame_id']}"]
    problems = []
    for scene in candidates:
        found = scene_problems(scene, frame, tracks, image_sha256)
        if not found:
            return scene, []
        problems += found
    return None, sorted(set(problems))


def _copy_pinned(source: Path, expected_sha256: str, target: Path, where: str) -> None:
    _require(sha256_file(source) == expected_sha256, f"{where}: {source.name} does not match its sha256")
    shutil.copyfile(source, target)
    _require(sha256_file(target) == expected_sha256, f"{where}: copying {source.name} changed its bytes")


def convert_cases(cases_path, scenes_dir, out_dir, *, intended_path=None, allow_stub_scenes=False) -> dict:
    """Task 2's final-test cases (Task 5's case schema) and Task 3's intended destinations -> references.jsonl.
    Nothing is left behind if a case is refused."""
    out_dir = Path(out_dir)
    _require(not out_dir.exists(), f"{out_dir} already exists; choose a new folder")
    try:
        return _convert_cases(cases_path, scenes_dir, out_dir, intended_path, allow_stub_scenes)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _convert_cases(cases_path, scenes_dir, out_dir: Path, intended_path, allow_stub_scenes: bool) -> dict:
    try:
        from trackrag.dropoff.data import load_cases
    except ImportError as error:
        raise GroundingError("task5-cases needs Task 5's trackrag/dropoff/data.py in this checkout") from error
    from PIL import Image

    try:
        cases = load_cases(cases_path, allowed_partitions=("test",))
    except (ValueError, OSError, RecursionError, SyntaxError, EOFError, Image.DecompressionBombError) as error:
        raise GroundingError(f"Task 5's case loader refused {Path(cases_path).name}: {error}") from error
    scenes = load_scenes(scenes_dir, allow_stub=allow_stub_scenes)
    intended = {}
    if intended_path is not None:
        for number, row in enumerate(read_jsonl(intended_path), 1):
            where = f"{Path(intended_path).name} line {number}"
            _require(_text(row.get("case_id")) and row["case_id"] not in intended, f"{where}: missing or repeated case_id")
            _require(row.get("category") in ("ambiguous", "missing_reference"),
                     f"{where}: category must be ambiguous or missing_reference")
            intended[row["case_id"]] = row
    (out_dir / "masks").mkdir(parents=True)
    references, used = [], set()
    for number, case in enumerate(cases, 1):
        where = f"case {case['case_id']}"
        _require("status" in case["supervised_fields"],
                 f"{where}: a final-evaluation case needs a full interpretation target (status included)")
        request, target, source = case["request"], case["target"], case["source"]
        frame = request["scene"]["frame"]
        scene, problems = find_scene(scenes, frame, request["scene"]["tracks"], source["image_sha256"])
        _require(scene is not None, f"{where}: no matching Task 1 scene: {'; '.join(problems)}")
        status = target["status"]
        if status == "needs_clarification":
            record = intended.get(case["case_id"])
            _require(record is not None, f"{where}: needs_clarification cases must be listed in the intended file "
                                         "as ambiguous or missing_reference")
            category = record["category"]
            used.add(case["case_id"])
        else:
            category = CATEGORY_OF_STATUS[status]
            _require(case["case_id"] not in intended, f"{where}: the intended file lists a case whose status is {status}")
        row = {"case_id": case["case_id"], "group_id": case["scene_group"], "partition": "final_evaluation",
               "scene_sha256": scene["scene_sha256"], "recording_id": frame["recording_id"],
               "frame_id": frame["frame_id"], "image_sha256": scene["model_input"]["image_sha256"],
               "width": frame["width"], "height": frame["height"], "category": category, "expected_status": status,
               "instruction": request["text"], "dialogue": request["dialogue"],
               "mask": None, "mask_sha256": None, "target_box": None, "box_rule": None, "reference_point": None,
               "anchors": [], "anchor_roles_ordered": False, "intended_mask": None, "intended_mask_sha256": None,
               "review": {"independent_reviews": len(set(case["review"].get("reviewers") or [])),
                          "disagreements_resolved": case["review"].get("status") == "human_reviewed"}}
        if category == "clear":
            _require(source.get("mask_path") is not None, f"{where}: a clear case needs a destination mask")
            name = f"masks/{number:04d}-destination.png"
            _copy_pinned(Path(source["mask_path"]), source["mask_sha256"], out_dir / name, where)
            row.update(mask=name, mask_sha256=source["mask_sha256"], target_box=target["destination_box_xyxy"],
                       box_rule="pixel_bbox", reference_point=target["destination_point_xy"])
        elif category == "ambiguous":
            path, sha = record.get("intended_mask"), record.get("intended_mask_sha256")
            _require(_text(path) and _text(sha), f"{where}: an ambiguous case needs intended_mask and its sha256")
            name = f"masks/{number:04d}-intended.png"
            _copy_pinned(Path(intended_path).parent / path, sha, out_dir / name, where)
            row.update(intended_mask=name, intended_mask_sha256=sha)
        if target.get("anchor_boxes_xyxy"):
            row["anchors"] = [{"role": f"anchor{i}", "bbox_xyxy": box}
                              for i, box in enumerate(target["anchor_boxes_xyxy"], 1)]
        references.append(row)
    unused = sorted(set(intended) - used)
    _require(not unused, f"The intended file lists cases that are not needs_clarification test cases: {unused[:5]}")
    references_path = out_dir / "references.jsonl"
    references_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in references), encoding="utf-8",
                               newline="\n")
    load_references(references_path, scenes)  # the evaluator's own checks, before anything is reported done
    record = {"version": 1, "cases_file": Path(cases_path).name, "cases_sha256": text_sha256(cases_path),
              "intended_sha256": text_sha256(intended_path) if intended_path else None,
              "references_sha256": text_sha256(references_path),
              "categories": dict(Counter(r["category"] for r in references)),
              "scenes": {r["case_id"]: r["scene_sha256"] for r in references}}
    (out_dir / "conversion.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _run_provenance(configuration: dict, outcomes: list[dict], model: str) -> dict:
    provenances = [_dict(o.get("provenance")) for o in outcomes]
    adapters = {json.dumps(p["adapter"].get("manifest_sha256")) for p in provenances if isinstance(p.get("adapter"), dict)}
    _require(len(adapters) <= 1, "The run names more than one adapter")
    adapters = {json.loads(a) for a in adapters}
    _require(all(_text(a) for a in adapters), "The run's adapter record has no manifest_sha256")
    _require(model == "trained" or not adapters, "A pretrained run names an adapter")
    substitute = any(p.get("injected") is not False or p.get("backend_kind") != "transformers_local"
                     for p in provenances)
    settings = {key: configuration.get(key) for key in TASK5_SETTINGS}
    return {"base_revision": f"{configuration.get('model_id')}@{configuration.get('revision')}",
            "adapter_sha256": adapters.pop() if adapters else None,
            "settings_sha256": _sha256_bytes(json.dumps(settings, sort_keys=True).encode("utf-8")),
            "image_budget": [configuration.get("min_pixels"), configuration.get("max_pixels")],
            "context_policy": TASK5_CONTEXT_POLICY, "coordinate_convention": "original_pixels", "turn_budget": None,
            "substitute": substitute}


def convert_run(run_dir, references_path, scenes_dir, transcript: str, *, allow_stub_scenes=False) -> list[dict]:
    """One tools/dropoff_predict.py run folder -> evaluator output records (direct mode)."""
    try:
        return _convert_run(Path(run_dir), references_path, scenes_dir, transcript, allow_stub_scenes)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise GroundingError(f"The run folder {Path(run_dir).name} has an unexpected shape: {error!r}") from error


def _convert_run(run_dir: Path, references_path, scenes_dir, transcript: str, allow_stub_scenes: bool) -> list[dict]:
    _require(transcript in ("typed", "human", "asr"), "transcript must be typed, human or asr")
    run = read_json(run_dir / "run.json")
    _require(isinstance(run, dict) and isinstance(run.get("configuration"), dict), "run.json has no configuration")
    for name, key in TASK5_RUN_FILES:
        _require(sha256_file(run_dir / name) == run.get(key), f"{name} does not match run.json; the run folder changed")
    _require(run["configuration"].get("allow_annotated") is not True,
             "The run allowed annotated (oracle) scenes; only predicted scenes are scored")
    scenes = load_scenes(scenes_dir, allow_stub=allow_stub_scenes)
    references, _ = load_references(references_path, scenes, allow_unreviewed=True)
    requests = {r.get("request_id"): r for r in read_jsonl(run_dir / "resolved-requests.jsonl")}
    receipts = {r.get("request_id"): r for r in read_jsonl(run_dir / "receipts.jsonl")}
    outcomes = read_jsonl(run_dir / "model-outcomes.jsonl")
    _require(len(outcomes) == run.get("requested_rows") == len(requests), "The run's files disagree on the row count")
    conditions = {json.dumps(_dict(o.get("provenance")).get("requested_condition")) for o in outcomes}
    _require(len(conditions) == 1, f"The run mixes conditions {sorted(conditions)}")
    requested = json.loads(conditions.pop())
    _require(isinstance(requested, str) and requested in TASK5_CONDITIONS,
             f"The run's condition {requested!r} is neither the pretrained nor the fine-tuned Qwen3-VL condition "
             "(the Ollama route is a separate baseline)")
    model = TASK5_CONDITIONS[requested]
    provenance = dict(_run_provenance(run["configuration"], outcomes, model),
                      task5_run={"run_json_sha256": sha256_file(run_dir / "run.json"),
                                 "requests_sha256": run.get("requests_sha256")})
    records, seen = [], set()
    for outcome in outcomes:
        case = outcome.get("request_id")
        _require(_text(case) and case in references, f"Request {case!r} is not a reserved case (request_id must equal case_id)")
        _require(case not in seen, f"Request {case!r} appears twice in the run")
        seen.add(case)
        request = requests.get(case)
        _require(isinstance(request, dict), f"Request {case!r} is missing from resolved-requests.jsonl")
        _require(_dict(request.get("scene")).get("source") == "predicted",
                 f"Request {case!r} does not use a predicted scene")
        reference = references[case]
        provided = _dict(outcome.get("provenance")).get("image")
        image_sha = (provided.get("sha256") if isinstance(provided, dict)
                     else _dict(request["scene"].get("provenance")).get("image_sha256"))
        problems = scene_problems(scenes[reference["scene_sha256"]], request["scene"]["frame"],
                                  request["scene"]["tracks"], image_sha)
        call, result, invalid_reason = from_task5_outcome(outcome)
        record = {"case_id": case, "condition": {"model": model, "mode": "direct", "transcript": transcript},
                  "scene_sha256": None if problems else reference["scene_sha256"],
                  "instruction_sha256": instruction_sha256(request["text"], request.get("dialogue", [])),
                  "call": call, "result": result, "raw_response": (receipts.get(case) or {}).get("raw_text"),
                  "provenance": provenance}
        if invalid_reason:
            record["invalid_reason"] = invalid_reason
        if problems:
            record["scene_problems"] = problems
        records.append(record)
    return records


def freeze_protocol(references, *, evaluator_commit: str, responder_rules: str, replies=None, voice=None,
                    frozen_utc: str | None = None) -> dict:
    _require(_text(evaluator_commit) and _text(responder_rules), "evaluator_commit and responder_rules are required")
    stamp = frozen_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"version": 1, "frozen_utc": stamp, "evaluator_commit": evaluator_commit,
            "references_sha256": text_sha256(references), "replies_sha256": text_sha256(replies) if replies else None,
            "voice_sha256": text_sha256(voice) if voice else None, "responder_rules": responder_rules,
            **json.loads(json.dumps(PROTOCOL_DEFAULTS))}


def _write_new(path, text: str) -> None:
    path = Path(path)
    _require(not path.exists() and path.parent.is_dir(), f"{path} already exists or its folder does not")
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.grounding_inputs",
                                     description="Turn the team's files into evaluator inputs, or freeze the protocol.")
    commands = parser.add_subparsers(dest="command", required=True)
    cases = commands.add_parser("task5-cases", help="Task 2's final-test cases -> references")
    cases.add_argument("--cases", required=True)
    cases.add_argument("--intended", help="Task 3's record of each needs_clarification case")
    cases.add_argument("--scenes", required=True)
    cases.add_argument("--out-dir", required=True, help="A new folder for references.jsonl and the masks")
    run = commands.add_parser("task5-run", help="A tools.dropoff_predict run folder -> evaluator outputs")
    run.add_argument("--run", required=True)
    run.add_argument("--references", required=True)
    run.add_argument("--scenes", required=True)
    run.add_argument("--transcript", required=True, choices=("typed", "human", "asr"))
    run.add_argument("--out", required=True)
    protocol = commands.add_parser("protocol", help="Freeze the protocol before final outputs exist")
    protocol.add_argument("--references", required=True)
    protocol.add_argument("--replies")
    protocol.add_argument("--voice")
    protocol.add_argument("--evaluator-commit", required=True)
    protocol.add_argument("--responder-rules", required=True)
    protocol.add_argument("--out", required=True)
    for sub in (cases, run):
        sub.add_argument("--allow-stub-scenes", action="store_true", help="Tests only")
    args = parser.parse_args(argv)
    try:
        if args.command == "task5-cases":
            result = convert_cases(args.cases, args.scenes, args.out_dir, intended_path=args.intended,
                                   allow_stub_scenes=args.allow_stub_scenes)
            print(f"{sum(result['categories'].values())} references {result['categories']} -> {args.out_dir}")
        elif args.command == "task5-run":
            _require(not Path(args.out).exists(), f"{args.out} already exists")
            records = convert_run(args.run, args.references, args.scenes, args.transcript,
                                  allow_stub_scenes=args.allow_stub_scenes)
            _write_new(args.out, "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
            mismatched = sum("scene_problems" in r for r in records)
            print(f"{len(records)} outputs ({records[0]['condition']['model'] if records else '-'}, {args.transcript}); "
                  f"{mismatched} with a scene mismatch -> {args.out}")
        else:
            body = freeze_protocol(args.references, evaluator_commit=args.evaluator_commit,
                                   responder_rules=args.responder_rules, replies=args.replies, voice=args.voice)
            _write_new(args.out, json.dumps(body, indent=2) + "\n")
            print(f"protocol frozen at {body['frozen_utc']} -> {args.out}")
    except GroundingError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
