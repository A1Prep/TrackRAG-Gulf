"""Evaluate a frozen language contrast bank against actual current-frame referent rankings, using CPU only."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import subprocess
import types

from PIL import Image

from tools.dropoff_passenger_handoff import batch, bound_outcome, records
from tools.dropoff_predict import read_requests
from tools.vision_saved_adapters import load_track_artifact
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.referent import ReferentRanker


PROTOCOL = "e49f92ae599b65955aac17dbb6586485bf5ca74a63c2178bc80667ee641a27b0"
PROTOCOL_FILE = Path("experiments/dropoff/passenger-egocentric-language-protocol-20261122.json")
PRECISE = {"other_landmark", "specific_ordinal", "specific_visual_attribute", "static_destination"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def registered_python_bytes(path, expected_sha256, *, git_blob=None):
    """Check an LF-registered Python source, including a CRLF working-tree checkout of it."""
    if path.suffix != ".py":
        raise ValueError("Only Python source may be checked with normalized line endings")
    original = path.read_bytes() if git_blob is None else git_blob
    registered = original.replace(b"\r\n", b"\n")
    if sha256(registered).hexdigest() != expected_sha256:
        raise ValueError(f"Registered Python source changed: {path}")
    return registered


def load_protocol():
    if file_sha256(PROTOCOL_FILE) != PROTOCOL:
        raise ValueError("The language contrast protocol changed after registration")
    return json.loads(PROTOCOL_FILE.read_bytes())


def prepare(predictions, weights, results):
    protocol = load_protocol()
    registered_python_bytes(Path("trackrag/dropoff/referent.py"), protocol["source_ranker_module_sha256"])
    ranker = ReferentRanker.load(weights, expected_sha256=protocol["ranker_artifact_sha256"])
    frames, track_ids = {}, {}
    for scene, declared in protocol["real_nonfinal_scenes"].items():
        rec, fid = declared["recording_id"], declared["frame_id"]
        meta = predictions / (rec + ".provenance.json")
        trackfile = predictions / (rec + ".tracks.jsonl")
        if (file_sha256(meta) != declared["source_provenance_sha256"]
                or file_sha256(trackfile) != declared["source_trackfile_sha256"]):
            raise ValueError("Task 1 artifact changed")
        if rec not in track_ids:
            track_ids[rec] = load_track_artifact(trackfile, json.loads(meta.read_bytes()), rec)
        all_rows, all_frames, size, _ = track_ids[rec]
        rows = [r for r in all_rows if r["frame_id"] == fid]
        path = Path(declared["relative_image_path"])
        if (fid not in all_frames or list(size) != declared["size"]
                or file_sha256(path) != declared["image_sha256"]
                or sha256(canonical(rows)).hexdigest() != declared["tracks_sha256"]):
            raise ValueError("Frozen current-frame pixels or detections changed")
        frame = {"recording_id": rec, "frame_id": fid, "timestamp_s": (fid - 1) / 10,
                 "width": size[0], "height": size[1], "image_path": str(path.resolve())}
        frames[scene] = (frame, rows)
    measured = []
    for case in protocol["cases"]:
        frame, rows = frames[case["scene"]]
        with Image.open(frame["image_path"]) as image:
            ranking = ranker.rank(image.convert("RGB"), case["utterance"], rows, frame=frame)
        if ranking["candidate_count"] != len(rows) or ranking["checkpoint_sha256"] != ranker.sha256:
            raise ValueError("Ranker did not consume the original observations and checkpoint")
        measured.append({"case_id": case["id"], "scene": case["scene"], "utterance": case["utterance"],
                         "original_track_ids": sorted(row["track_id"] for row in rows),
                         "ranking_sha256": sha256(canonical(ranking)).hexdigest(), "ranking": ranking})
    with results.open("xb") as out:
        for item in measured:
            out.write(canonical(item) + b"\n")


def old_source(protocol):
    reference = protocol["old_policy_reachable_git_ref"]
    blob = subprocess.check_output(["git", "show", reference])
    value = registered_python_bytes(Path(reference.split(":", 1)[1]), protocol["old_policy_sha256"], git_blob=blob)
    module = types.ModuleType("trackrag.dropoff._frozen_egocentric_baseline")
    module.__package__ = "trackrag.dropoff"
    exec(compile(value, "<archived frozen clarification policy>", "exec"), module.__dict__)
    return module.clarify_reference


def evaluate_case(policy, case, row, scene):
    try:
        ranked = deepcopy(row["ranking"])
        answer = policy(case["utterance"], ranked, tuple(scene["size"]))
        if sha256(canonical(ranked)).hexdigest() != row["ranking_sha256"]:
            raise ValueError("Guard mutated the frozen ranking")
        if answer is None:
            return {"action": "defer", "matches": case["expected_guard_action"] == "defer"}
        ids = answer.get("track_ids", [])
        if (len(ids) < 2 or len(set(ids)) != len(ids) or not set(ids) <= set(row["original_track_ids"])
                or not isinstance(answer.get("question"), str) or "?" not in answer["question"]):
            raise ValueError("Question lacks two distinct genuine detector IDs or question text")
        if case["scene"] == "two_buses" and set(ids) != set(scene["observed_bus_ids"]):
            raise ValueError("Bus question does not refer to the two recorded buses")
        return {"action": "ask", "matches": case["expected_guard_action"] == "ask",
                "track_ids": ids, "question": answer["question"], "reason": answer.get("reason"),
                "policy": answer.get("policy")}
    except (ValueError, TypeError, KeyError, RuntimeError) as error:
        return {"action": "error", "matches": False, "error": str(error)}


def compare(rankings, new_sha):
    protocol = load_protocol()
    registered_python_bytes(Path("trackrag/dropoff/perspective.py"), new_sha)
    registered_python_bytes(Path("trackrag/dropoff/referent_policy.py"), protocol["old_policy_sha256"])
    new = importlib.import_module("trackrag.dropoff.perspective").clarify_reference
    old = old_source(protocol)
    rows = [json.loads(line) for line in rankings.read_bytes().splitlines()]
    if [r["case_id"] for r in rows] != [r["id"] for r in protocol["cases"]]:
        raise ValueError("Missing, extra or re-ordered frozen cases")
    cases, counts = [], {}
    for case, row in zip(protocol["cases"], rows, strict=True):
        scene = protocol["real_nonfinal_scenes"][case["scene"]]
        if (row["utterance"] != case["utterance"] or row["scene"] != case["scene"]
                or sha256(canonical(row["ranking"])).hexdigest() != row["ranking_sha256"]
                or row["ranking"]["checkpoint_sha256"] != protocol["ranker_artifact_sha256"]):
            raise ValueError("Ranking belongs to another registered utterance, scene or checkpoint")
        record = dict(case, ranking_sha256=row["ranking_sha256"], actual_original_image_sha256=scene["image_sha256"])
        for name, method in (("original", old), ("additive", new)):
            outcome = evaluate_case(method, case, row, scene)
            record[name] = outcome
            group = counts.setdefault(name, {}).setdefault(case["group"], Counter())
            group["cases"] += 1
            group["matches"] += int(outcome["matches"])
            group["false_asks"] += int(outcome["action"] == "ask" and case["expected_guard_action"] == "defer")
            group["missed_asks"] += int(outcome["action"] == "defer" and case["expected_guard_action"] == "ask")
            group["operational_errors"] += int(outcome["action"] == "error")
        cases.append(record)
    ego_improvement = counts["additive"]["ego_uncertainty"]["matches"] > counts["original"]["ego_uncertainty"]["matches"]
    new_false_asks = sum(r["additive"]["action"] == "ask" and r["original"]["action"] != "ask"
                         for r in cases if r["group"] in PRECISE)
    cardinality = all(r["additive"]["matches"] for r in cases if r["group"] == "real_cardinality_control")
    return {"kind": "paired_CPU_referent_clarification_language_diagnostic", "protocol_sha256": PROTOCOL,
            "old_source_sha256": protocol["old_policy_sha256"], "new_source_sha256": new_sha,
            "identical_rankings_sha256": file_sha256(rankings), "learned_ranker_sha256": protocol["ranker_artifact_sha256"],
            "real_nonfinal_frames": 4, "utterance_scene_cases": len(cases), "human_reviews": 0,
            "selection": {"ego_language_improved": ego_improvement, "new_false_asks_on_precise_controls": new_false_asks,
                          "one_vs_two_real_buses_passed": cardinality,
                          "meets_registered_signal_for_separate_integration": ego_improvement and not new_false_asks and cardinality},
            "counts_by_reference_group": counts, "cases": cases,
            "meaning": "These are AI-authored expectations for one ask-only guard. Defer leaves the decision to Qwen. "
                       "No vehicle control, human intention or final destination accuracy was measured."}


def replay(study, new_sha):
    protocol = load_protocol()
    registered_python_bytes(Path("trackrag/dropoff/perspective.py"), new_sha)
    registered_python_bytes(Path("trackrag/dropoff/referent_policy.py"), protocol["old_policy_sha256"])
    from trackrag.dropoff.perspective import PassengerViewpoint

    cohorts = {"original_six": study / "trial", "changed_context_four": study / "operational-context"}
    all_results = {}
    for label, folder in cohorts.items():
        inputs = folder / "requests.jsonl"
        requests = {r["request_id"]: r for r in read_requests(inputs)}
        registered = json.loads((folder / "protocol.json").read_bytes())
        dimensions = (6, "7d805dc9f478f894e198892c435df00db897586f40465c393b8644b64c8aadc0") if label == "original_six" else (
            4, "cc92d862a0fe739a2f4874cbcb12d7914e25e652e78069dab6db17de98010d43")
        if len(requests) != dimensions[0] or file_sha256(inputs) != dimensions[1]:
            raise ValueError("Unexpected replay cohort")
        conditions = {}
        for condition in ("public_parent", "human_child"):
            original_run = folder / (condition + "-passenger" if label == "original_six" else condition)
            historical = batch(original_run, inputs)
            if historical["settings"]["resolver_execution"] != "live":
                raise ValueError("Expected historically live, independently verified original outputs")
            archive = original_run / "outcomes.jsonl"
            initial_sha = file_sha256(archive)
            changes, examples = Counter(), []
            originals = {r["request_id"]: r for r in historical["events"]}
            for line_sha, old in records(archive):
                identity = old["request_id"]
                req = requests[identity]
                event = originals[identity]
                if line_sha != event["raw_outcome_line_sha256"]:
                    raise ValueError("Archived outcome differs from the audited raw input")
                retained = canonical(old)
                revised = PassengerViewpoint.review(req, deepcopy(old))
                if canonical(old) != retained:
                    raise ValueError("Original production output changed")
                before = old.get("result", {}).get("status", "operational_error")
                after = revised.get("result", {}).get("status", "operational_error")
                advice = revised["provenance"].get("viewpoint_clarification")
                if revised != old:
                    original_provenance = dict(revised["provenance"])
                    original_provenance.pop("viewpoint_clarification")
                    if (not old["ok"] or before != "proposed" or after != "needs_clarification"
                            or original_provenance != old["provenance"] or advice["original_proposal"] != old["result"]
                            or advice["decision"]["question"] != revised["result"]["clarification_question"]):
                        raise ValueError("Policy replay modified more than a qualifying original proposal")
                    ids = advice["decision"]["track_ids"]
                    if len(set(ids)) < 2 or not set(ids) <= {r["track_id"] for r in req["scene"]["tracks"]}:
                        raise ValueError("Replayed question lacks two genuine current-frame candidates")
                bound_outcome(revised, req)
                changes[before + " -> " + after] += 1
                group = (registered["targets_scoring_only"][identity]["group"] if label == "original_six" else
                         next(r["original_input_kind"] for r in registered["bindings"] if r["request_id"] == identity))
                examples.append({"request_id": identity, "original_reference_group": group,
                    "changed_input_has_human_review": False if label == "changed_context_four" else None,
                    "text": req["text"], "image_sha256": req["scene"]["provenance"]["image_sha256"],
                    "current_track_count": len(req["scene"]["tracks"]), "raw_original_outcome_line_sha256": line_sha,
                    "original_raw_resolver_text_sha256": event.get("landmark_raw_text_sha256"),
                    "before": old.get("result"), "after": revised.get("result"), "original_error": old.get("error"),
                    "replay_error": revised.get("error"), "viewpoint_advice": advice,
                    "replayed_envelope_sha256": sha256(canonical(revised)).hexdigest()})
            if file_sha256(archive) != initial_sha:
                raise ValueError("One of the archived production result files was modified")
            conditions[condition] = {"verified_source_gpu_run_sha256": historical["run_sha256"],
                                     "original_outcomes_sha256": initial_sha, "status_transitions": changes,
                                     "rows": examples}
        all_results[label] = conditions
    return {"kind": "CPU_postprocessing_replay_of_saved_genuine_application_outputs", "new_policy_sha256": new_sha,
            "new_GPU_calls": 0, "source_live_attempts": 20, "human_confirmed_original_commands_per_condition": 2,
            "human_reviewed_changed_context_inputs": 0, "status_transition_results": all_results,
            "interpretation": "The original Qwen, detector, learned referent ranking and predictions were held fixed. "
                              "Only the new clarification check was recomputed. Exact original human confirmations "
                              "apply to the original image-only commands. No changed-context or new human accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="mode", required=True)
    prep = actions.add_parser("prepare")
    for arg in ("predictions", "weights", "rankings"):
        prep.add_argument("--" + arg, required=True, type=Path)
    trial = actions.add_parser("compare")
    trial.add_argument("--rankings", type=Path, required=True)
    trial.add_argument("--new-sha", required=True)
    trial.add_argument("--out", type=Path, required=True)
    replay_parser = actions.add_parser("replay")
    replay_parser.add_argument("--study", type=Path, required=True)
    replay_parser.add_argument("--new-sha", required=True)
    replay_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.predictions, args.weights, args.rankings)
    else:
        result = replay(args.study, args.new_sha) if args.mode == "replay" else compare(args.rankings, args.new_sha)
        with args.out.open("x", encoding="utf8", newline="\n") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
