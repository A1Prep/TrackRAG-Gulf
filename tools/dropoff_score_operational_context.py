"""Compare four exact nonfinal inputs before and after adding their original Task 1 detector tracks."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from tools.dropoff_passenger_handoff import batch
from tools.dropoff_predict import read_requests
from tools.dropoff_reviewed_trial import normalized_question
from tools.vision_saved_adapters import load_track_artifact
from trackrag.dropoff.data import file_sha256, inference_input_sha256, load_cases


PROTOCOL = "9017add2c987628541d3fb11f63e8994928bece931f9a37c3279d97c1dcdc503"
ORIGINAL_PROTOCOL = "702c5f5575431df621b0e78847ddadb67d102e487a4be44016050773e4af8b81"
CONDITIONS = ("public_parent", "human_child")
MEANINGS = {
    "proposed": "The application offers a location for the passenger to confirm.",
    "needs_clarification": "The application asks a question to resolve the reference.",
    "insufficient_visual_evidence": "The application withholds a location without asking a distinguishing question.",
    "operational_error": "The application did not return a valid interpretation. The attempt remains in its denominator.",
}


def read(path):
    return json.loads(path.read_bytes())


def verify_inputs(study, protocol, predictions):
    previous = study / "trial/requests.jsonl"
    current = study / "operational-context/requests.jsonl"
    if (file_sha256(previous) != protocol["original_trial_requests_sha256"]
            or file_sha256(current) != protocol["changed_requests_sha256"]):
        raise ValueError("Frozen model inputs changed")
    old = {r["request_id"]: r for r in read_requests(previous)}
    new = {r["request_id"]: r for r in read_requests(current)}
    bound = {r["request_id"]: r for r in protocol["bindings"]}
    if len(old) != 6 or len(new) != 4 or set(new) != set(bound) or not new.keys() <= old.keys():
        raise ValueError("Expected exactly the original two development commands and their two paraphrases")
    cached = {}
    for identity, changed in new.items():
        original = old[identity]
        binding = bound[identity]
        frame = original["scene"]["frame"]
        rec, fid = frame["recording_id"], frame["frame_id"]
        if rec not in ("video_054907", "video_174340") or fid != 781 or original["scene"]["tracks"]:
            raise ValueError("Unexpected recording, frame or original source")
        if rec not in cached:
            track = predictions / (rec + ".tracks.jsonl")
            provenance = predictions / (rec + ".provenance.json")
            meta = read(provenance)
            if (file_sha256(provenance) != binding["saved_provenance_sha256"]
                    or meta["identity_hash"] != binding["vision_identity_sha256"]):
                raise ValueError("Task 1 detector provenance differs from the frozen inputs")
            observations, frames, size, digest = load_track_artifact(track, meta, rec)
            if (fid not in frames or list(size) != [frame["width"], frame["height"]]
                    or digest != binding["saved_tracks_sha256"]):
                raise ValueError("Wrong Task 1 artifact or frame")
            cached[rec] = json.loads(json.dumps([r for r in observations if r["frame_id"] == fid]))
        expected = cached[rec]
        if len(expected) != binding["saved_track_count"] or changed["scene"]["tracks"] != expected:
            raise ValueError("Counterfactual observations differ from the complete original current-frame rows")
        restore = deepcopy(changed)
        restore["scene"]["tracks"] = []
        restore["scene"]["provenance"] = original["scene"]["provenance"]
        image_sha = file_sha256(frame["image_path"])
        if (restore != original or image_sha != binding["image_sha256"]
                or inference_input_sha256(original, image_sha) != binding["image_only_inference_input_sha256"]
                or inference_input_sha256(changed, image_sha) != binding["changed_trackful_inference_input_sha256"]
                or binding["changed_input_has_human_review"] is not False):
            raise ValueError("The pair changed an additional input or its original review provenance")
    return bound


def summarize(event, input_sha, image_sha, weight):
    summary = {"operational_ok": event["ok"], "raw_outcome_line_sha256": event["raw_outcome_line_sha256"]}
    if not event["ok"]:
        return dict(summary, status="operational_error", error=event["error"])
    source = event["saved_provenance"]
    resolver = source["resolver"]
    if (event["inference_input_sha256"] != input_sha or event["image_sha256"] != image_sha
            or resolver["adapter"]["files"]["adapter_model.safetensors"]["sha256"] != weight
            or source.get("resolver_execution") != "live" or event["phase"] != "initial_model"):
        raise ValueError("Model, phase, pixels, or portable input changed in the raw result")
    return dict(summary, status=event["result"]["status"], four_field_result=event["result"],
                resolver_decision=resolver["decision"], resolver_raw_text=resolver["raw_text"],
                resolver_raw_text_sha256=event["landmark_raw_text_sha256"],
                resolver_messages_sha256=resolver["messages_sha256"], resolver_generation_contract=resolver["contract"],
                referent_clarification=source.get("referent_clarification"), constraint=source.get("constraint"))


def score(study, predictions):
    changed = study / "operational-context"
    if file_sha256(changed / "protocol.json") != PROTOCOL:
        raise ValueError("The study protocol is not the frozen version")
    protocol = read(changed / "protocol.json")
    execution = read(changed / "execution.json")
    if (execution.get("study_protocol_sha256") != PROTOCOL or not execution.get("end")
            or tuple(row["condition"] for row in execution["runs"]) != CONDITIONS):
        raise ValueError("Seal both prespecified production model conditions before comparison")
    bindings = verify_inputs(study, protocol, predictions)
    condition_results = {}
    for cond in CONDITIONS:
        prior = batch(study / "trial" / (cond + "-passenger"), study / "trial/requests.jsonl")
        current = batch(changed / cond, changed / "requests.jsonl")
        old_settings, new_settings = prior["settings"], current["settings"]
        for key in ("config", "source", "geometry_sha256", "referent_sha256", "resolver_execution",
                    "verify_landmarks", "saved_decisions_sha256", "reference_access"):
            if old_settings[key] != new_settings[key]:
                raise ValueError(f"Production {key} changed between input conditions")
        if (new_settings["resolver_execution"] != "live" or new_settings["reference_access"]
                or new_settings["verify_landmarks"] or new_settings["saved_decisions_sha256"] is not None
                or new_settings["geometry_sha256"] != protocol["geometry_sha256"]
                or new_settings["referent_sha256"] != protocol["referent_sha256"]):
            raise ValueError("Expected the frozen live production pipeline")
        old = {v["request_id"]: v for v in prior["events"]}
        new = {v["request_id"]: v for v in current["events"]}
        weight = protocol["conditions"][cond]["local_weights_sha256"]
        cases = []
        strata = {}
        for identity, binding in bindings.items():
            before = summarize(old[identity], binding["image_only_inference_input_sha256"], binding["image_sha256"], weight)
            after = summarize(new[identity], binding["changed_trackful_inference_input_sha256"], binding["image_sha256"], weight)
            if (before["operational_ok"] and after["operational_ok"]
                    and before["resolver_messages_sha256"] == after["resolver_messages_sha256"]):
                raise ValueError("Stored resolver prompts fail to reflect the new tracked scene")
            group = binding["original_input_kind"]
            transitions = strata.setdefault(group, Counter())
            transitions[before["status"] + " -> " + after["status"]] += 1
            cases.append(dict(request_id=identity, prior_input_group=group, image_sha256=binding["image_sha256"],
                              current_frame_track_count=binding["saved_track_count"], text=binding["text"],
                              before=before, after=after))
        condition_results[cond] = {
            "weights_sha256": weight, "previous_production_run_sha256": prior["run_sha256"],
            "tracked_production_run_sha256": current["run_sha256"],
            "previous_raw_outcomes_sha256": prior["outcomes_sha256"], "tracked_raw_outcomes_sha256": current["outcomes_sha256"],
            "paired_cases": cases, "transitions_by_original_input_group": strata,
            "changed_status_count": sum(r["before"]["status"] != r["after"]["status"] for r in cases),
        }
    return {"kind": "observed_operational_track_context_sensitivity", "protocol_sha256": PROTOCOL,
            "requests_per_adapter": 4, "fresh_changed_context_model_attempts": 8,
            "geometry_sha256": protocol["geometry_sha256"], "referent_sha256": protocol["referent_sha256"],
            "changed_requests_sha256": protocol["changed_requests_sha256"],
            "status_meanings": MEANINGS,
            "interpretation": "Original human confirmations apply only to the two image-only requests. "
                              "The four track-bearing requests have no new human reference. The production "
                              "landmark guide and allowed generation forms switch automatically when tracks "
                              "are available. These pairs measure the operational effect of detector context.",
            "human_reviews_of_changed_inputs": 0, "accuracy_estimate": None, "conditions": condition_results}


def score_original(study, native_result):
    trial = study / "trial"
    if file_sha256(trial / "protocol.json") != ORIGINAL_PROTOCOL:
        raise ValueError("Original trial protocol changed")
    protocol = read(trial / "protocol.json")
    native = read(native_result)
    if native["protocol_sha256"] != ORIGINAL_PROTOCOL:
        raise ValueError("Native comparator evaluated a different cohort")
    req = {r["request_id"]: r for r in read_requests(trial / "requests.jsonl")}
    reviewed = {r["case_id"]: r for r in load_cases(study / "data/development.jsonl",
                                                     allowed_partitions={"development"})}
    targets = protocol["targets_scoring_only"]
    if set(req) != set(targets) or len(req) != 6 or len(reviewed) != 2:
        raise ValueError("Unexpected comparator cohort")
    bank = Path(__file__).resolve().parents[1] / "annotations/dropoff/passenger-development"
    previous_ai = {r["request_id"]: r for line in (bank / "development-references.jsonl").read_bytes().splitlines()
                   for r in [json.loads(line)] if r["request_id"] in req}
    settings = None
    conditions = {}
    for cond in CONDITIONS:
        truth = native["fresh_generation"][cond]
        native_rows = {r["request_id"]: r for r in truth["rows"]}
        live = batch(trial / (cond + "-passenger"), trial / "requests.jsonl")
        live_settings = live["settings"]
        if (live["requests_sha256"] != protocol["requests_sha256"] or live_settings["resolver_execution"] != "live"
                or live_settings["verify_landmarks"] or live_settings["reference_access"]
                or set(native_rows) != set(req)):
            raise ValueError("Expected matched full-application calls without references or a verifier")
        comparable = {k: v for k, v in live_settings.items() if k != "config"}
        comparable["config"] = {k: v for k, v in live_settings["config"].items() if k != "adapter_dir"}
        if settings is None:
            settings = comparable
        if comparable != settings:
            raise ValueError("The matched full-application conditions differ beyond their adapter")
        weight_field = "parent_sha256" if cond == "public_parent" else "child_sha256"
        weight = native["tensor_verification"][weight_field]
        strata, cases = {}, []
        for event in live["events"]:
            identity = event["request_id"]
            r, target = req[identity], targets[identity]
            image = file_sha256(r["scene"]["frame"]["image_path"])
            binding = inference_input_sha256(r, image)
            expected = target["target"]["status"]
            group = target["group"]
            expected_question = None
            if group == "two_reviewer_confirmations_same_wording":
                original = reviewed[identity]
                if original["review"]["input_sha256"] != binding or original["target"] != target["target"]:
                    raise ValueError("A purported human datum differs from its confirmed input or target")
                expected_question = original["target"]["clarification_question"]
            elif group == "existing_ai_development_control":
                if previous_ai[identity]["target"]["status"] != expected:
                    raise ValueError("Earlier provisional AI control label changed")
            elif group != "unreviewed_paraphrase_same_scene":
                raise ValueError("Unknown reference type")
            application = summarize(event, binding, image, weight)
            status = application["status"]
            stats = strata.setdefault(group, {"attempts": 0, "matches_registered_status": 0,
                                               "proposals_when_registered_status_unresolved": 0,
                                               "operational_errors": 0})
            stats["attempts"] += 1
            stats["operational_errors"] += int(not event["ok"])
            stats["matches_registered_status"] += int(status == expected)
            stats["proposals_when_registered_status_unresolved"] += int(status == "proposed" and expected != "proposed")
            question = application.get("four_field_result", {}).get("clarification_question")
            cases.append({"request_id": identity, "reference_type": group, "original_text": r["text"],
                          "recording_id": r["scene"]["frame"]["recording_id"], "frame_id": r["scene"]["frame"]["frame_id"],
                          "image_sha256": image, "portable_input_sha256": binding, "registered_status": expected,
                          "registered_human_question": expected_question,
                          "matches_human_question": (normalized_question(question) == normalized_question(expected_question))
                              if expected_question else None,
                          "native_direct_status_on_identical_input": native_rows[identity]["actual"],
                          "full_application": application})
        conditions[cond] = {"weights_sha256": weight, "full_application_run_sha256": live["run_sha256"],
                            "raw_outcomes_sha256": live["outcomes_sha256"],
                            "native_direct_run_sha256": truth["run_sha256"], "reference_groups": strata,
                            "requests": cases}
    return {"kind": "normal_application_on_reviewed_clarification_and_registered_controls",
            "protocol_sha256": ORIGINAL_PROTOCOL, "requests_sha256": protocol["requests_sha256"],
            "actual_full_application_attempts": 12, "native_direct_companion_sha256": file_sha256(native_result),
            "status_meanings": MEANINGS, "conditions": conditions,
            "reference_scope": "Two exact image-only inputs have two recorded confirmations of AI-assisted labels. "
                               "The two paraphrases are unreviewed tests of the registered expectation. Two other "
                               "controls use provisional AI-authored references. The original two human cases share "
                               "training wording and were also used in selecting an epoch. No final test accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--predictions", type=Path, help="Score the changed Task 1 context sensitivity")
    selection.add_argument("--original-with-native", type=Path, help="Score the original six application rows against their own references")
    args = parser.parse_args()
    result = score_original(args.study, args.original_with_native) if args.original_with_native else score(args.study, args.predictions)
    with args.out.open("x", encoding="utf8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
