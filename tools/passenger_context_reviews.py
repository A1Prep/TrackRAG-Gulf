"""Build a fresh, private two-reviewer packet for the installed saved-detector context."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from evaluation.grounding import load_scenes
from evaluation.grounding_inputs import find_scene
from tools.dropoff_final_input_inventory import PACKAGE, ROOT, TASK2, git
from tools.dropoff_predict import read_requests
from tools.dropoff_restore_evaluator import sources
from tools.passenger_reviewed_baseline import BINDING_HASH, REQUEST_HASH, indexed, json_rows, reviewed_bank
from trackrag.dropoff.data import file_sha256, inference_input_sha256, supervision_sha256
from trackrag.dropoff.inputs import _inspect_image


SAVED = "e6006e632dd75c7d08866f860ee6fa4c6d806f519c1569aba683cb58aa96a0ef"
SIGNED = "37f21c01f44f1a35fffc8092285d997713965dbad22086ac3a3de2fcb0557373"
RESOLVER = "3b903df45662d4f8442ab59396b28d0d6c6021a94442ee2db449c6128dda30cc"
HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def put(path, text):
    path.write_bytes(text.encode("utf-8"))


def private_cases(requests, originals, signed_rows, bindings, scenes, inputs, packet):
    """Join independently ordered sources by ID and bind each exact image, request and candidate."""
    requests = indexed(requests, "request_id")
    signed = indexed(signed_rows, "case_id")
    bindings = indexed(bindings, "case_id")
    require(requests.keys() == originals.keys() == signed.keys() == bindings.keys(), "Cohort identities differ")
    cases, scene_ids, photo_paths, input_hashes, old_hashes = [], set(), set(), set(), set()
    for identity, request in requests.items():
        previous, original, binding = signed[identity], originals[identity], bindings[identity]
        scene = request["scene"]
        frame = scene["frame"]
        image_path, image_hash = _inspect_image(frame)
        relative = sources(binding)
        require(image_path == (inputs / "source" / relative).resolve()
                and image_hash == binding["image_sha256"] == original.image_sha256
                and scene["provenance"]["image_sha256"] == image_hash, "Incorrect original photograph")
        original_hash = inference_input_sha256(original.request, image_hash)
        current_hash = inference_input_sha256(request, image_hash)
        require(original_hash == binding["reviewed_input_sha256"]
                and current_hash == binding["saved_detector_input_sha256"] and current_hash != original_hash,
                "Changed or previously matched detector binding")
        original_frame = original.request["scene"]["frame"]
        require(request["request_id"] == identity and request["input_mode"] == original.request["input_mode"]
                and request["text"] == original.request["text"] and request["dialogue"] == original.request["dialogue"]
                and all(frame[k] == original_frame[k] == binding[k] for k in
                        ("recording_id", "frame_id", "width", "height"))
                and frame["timestamp_s"] == binding["reconstructed_timestamp_s"]
                and original_frame["timestamp_s"] == binding["reviewed_timestamp_s"],
                "Instruction, dialogue, frame or claimed timestamps differ")
        tracks = scene["tracks"]
        canonical, issues = find_scene(scenes, frame, tracks, image_hash)
        condition = binding["saved_detector_conditions"]["finetuned"]
        require(canonical is not None and not issues
                and canonical["scene_sha256"] == condition["scene_sha256_from_original_hashes"]
                and canonical["model_input"]["observations"] == tracks
                and canonical["model_input"]["timestamp_s"] == frame["timestamp_s"]
                and len(tracks) == condition["current_observations"] and bool(tracks)
                and all(t["recording_id"] == frame["recording_id"] and t["frame_id"] == frame["frame_id"]
                        and t["source"] == "predicted" for t in tracks), "Saved observations or scene differ")
        candidate = deepcopy(previous)
        candidate["target"] = deepcopy(original.target)
        fields = candidate["supervised_fields"]
        label_hash = supervision_sha256(candidate["target"], fields, frame)
        require(label_hash == previous["review"]["supervision_sha256"], "Source candidate differs")
        candidate["request"] = deepcopy(request)
        candidate["request"]["scene"]["frame"]["image_path"] = relative.as_posix()
        candidate["source"] = {k: previous["source"][k] for k in
                               ("dataset", "source_split", "recording_id", "frame_id", "image_sha256",
                                "scene_group_verified", "grouping_level")}
        candidate["source"]["annotation_basis"] = "Candidate from independently reviewed image-only context."
        candidate["review"] = {"status": "pending_reconfirmation", "new_human_review": False,
                               "independent_reviewers_required": 2}
        destination = packet / relative
        require(relative not in photo_paths and current_hash not in input_hashes
                and original_hash not in old_hashes and canonical["scene_sha256"] not in scene_ids,
                "Repeated frame, input or canonical scene")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, destination)
        require(file_sha256(destination) == image_hash, "Copied photo differs")
        photo_paths.add(relative)
        old_hashes.add(original_hash)
        input_hashes.add(current_hash)
        scene_ids.add(canonical["scene_sha256"])
        cases.append(candidate)
    require(scene_ids == set(scenes) and not input_hashes.intersection(old_hashes), "Unmatched original or saved scene")
    return cases


def blank_row(case, resolver):
    target, request, source = case["target"], case["request"], case["source"]
    frame = request["scene"]["frame"]
    row = dict.fromkeys(resolver["HEADER"], "")
    row.update(case_id=case["case_id"], partition=case["partition"], recording_id=frame["recording_id"],
               frame_id=str(frame["frame_id"]), image_available="true", image_path=frame["image_path"],
               image_sha256=source["image_sha256"], instruction=request["text"], status=target["status"],
               clarification_question=target["clarification_question"] or "",
               input_sha256=resolver["input_sha256"](case), supervision_sha256=resolver["supervision_sha256"](case))
    for key in ("destination_point_xy", "destination_box_xyxy"):
        row[key] = compact(target[key]) if target[key] is not None else ""
    require(row["input_sha256"] == inference_input_sha256(request, source["image_sha256"])
            and row["supervision_sha256"] == supervision_sha256(target, case["supervised_fields"], frame),
            "Original resolver bindings differ from the project contracts")
    return row


def write_forms(packet, cases, bindings):
    """Generate one identical blank form per independent person and a self-contained offline page."""
    resolver = runpy.run_path(str(packet / "tools/dropoff_resolve_reviews.py"))
    rows = [blank_row(case, resolver) for case in cases]
    out = StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=resolver["HEADER"])
    writer.writeheader()
    writer.writerows(rows)
    for role in (1, 2):
        put(packet / f"reviewer_{role}.csv", out.getvalue())
    bank_hash = sha256(compact(rows).encode()).hexdigest()
    data = {"headers": resolver["HEADER"], "bank": bank_hash, "rows": rows,
            "scenes": [{"request": c["request"], "target": c["target"],
                        "old_time": bindings[c["case_id"]]["reviewed_timestamp_s"],
                        "old_input": bindings[c["case_id"]]["reviewed_input_sha256"],
                        "canonical_scene": bindings[c["case_id"]]["saved_detector_conditions"]["finetuned"][
                            "scene_sha256_from_original_hashes"]} for c in cases]}
    escaped = compact(data)
    for char in ("&", "<", ">", "\u2028", "\u2029"):
        escaped = escaped.replace(char, f"\\u{ord(char):04x}")
    html = (HERE / "passenger_context_reviews.html").read_text(encoding="utf-8")
    require(html.count("@@PACKET@@") == 1, "Invalid offline HTML template")
    put(packet / "review.html", html.replace("@@PACKET@@", escaped))
    return bank_hash


def archive(packet, path):
    with ZipFile(path, "x") as z:
        for item in sorted(p for p in packet.rglob("*") if p.is_file()):
            name = item.relative_to(packet).as_posix()
            info = ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = ZIP_STORED if item.suffix == ".jpg" else ZIP_DEFLATED
            z.writestr(info, item.read_bytes())


def build(installed, output):
    installed, output = Path(installed).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output folder. Existing review packets and returns are preserved.")
    if output.is_relative_to(ROOT):
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(output.relative_to(ROOT))], cwd=ROOT)
        require(ignored.returncode == 0, "Private review packets inside this checkout require a Git-ignored folder")
    inputs = installed / "inputs"
    saved = inputs / "requests-with-saved-tracks.jsonl"
    require(file_sha256(saved) == SAVED, "Use the canonical saved-detector input file")
    originals, audit = reviewed_bank(inputs / "requests-as-human-reviewed.jsonl", datetime.now(timezone.utc))
    signed_bytes, resolver_bytes = (git(TASK2, f"{PACKAGE}/{path}") for path in
                                   ("review_cases.jsonl", "tools/dropoff_resolve_reviews.py"))
    require(sha256(signed_bytes).hexdigest() == SIGNED and sha256(resolver_bytes).hexdigest() == RESOLVER,
            "Pinned Task 2 source or resolver differs")
    signed = [r for r in json_rows(signed_bytes) if r["partition"] == "test"]
    bindings = json_rows((inputs / "input-bindings.jsonl").read_bytes())
    scenes = load_scenes(installed / "canonical/scenes")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "complete"
        packet = stage / "packet"
        (packet / "tools").mkdir(parents=True)
        (packet / "tools/dropoff_resolve_reviews.py").write_bytes(resolver_bytes)
        cases = private_cases(read_requests(saved), originals, signed, bindings, scenes, inputs, packet)
        total = sum(len(c["request"]["scene"]["tracks"]) for c in cases)
        require(len(cases) == 44 and total == 494 and audit["withheld_pending_test_cases"] == 6,
                "Expected 44 original signed cases and 494 current observations")
        put(packet / "review_cases.jsonl", "")
        put(packet / "reconfirmation_cases.jsonl", "".join(compact(c) + "\n" for c in cases))
        bank_hash = write_forms(packet, cases, indexed(bindings, "case_id"))
        put(packet / "resolve_context.py", (HERE / "passenger_context_resolve.py").read_text(encoding="utf-8"))
        guide = (ROOT / "docs/passenger-context-review.md").read_text(encoding="utf-8")
        put(packet / "READ-ME.txt", guide.split("## Rebuild", 1)[0])
        inventory = {f.relative_to(packet).as_posix(): file_sha256(f) for f in sorted(packet.rglob("*")) if f.is_file()}
        put(packet / "packet-files.json", json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        dry = stage / "blank-resolver-check"
        subprocess.run([sys.executable, "-B", str(packet / "resolve_context.py"), "--reviewer-1",
                        str(packet / "reviewer_1.csv"), "--reviewer-2", str(packet / "reviewer_2.csv"),
                        "--output", str(dry)], check=True, capture_output=True)
        status = json.loads((dry / "review_status.json").read_bytes())
        require(status["current_cases"] == status["pending_cases"] == len(cases)
                and status["approved_cases_written"] == status["reviewed_cases"] == status["unresolved_disagreements"] == 0
                and not (dry / "approved_cases.jsonl").exists(), "Blank forms produced a review")
        bundle = stage / "passenger-context-review-private.zip"
        archive(packet, bundle)
        receipt = {"schema_version": 1, "purpose": "Offline context re-review readiness. Zero new human decisions.",
                   "task2_revision": TASK2, "original_task2_resolver_sha256": RESOLVER,
                   "signed_source_cases_sha256": SIGNED, "original_input_file_sha256": REQUEST_HASH,
                   "changed_input_file_sha256": SAVED, "release_bindings_sha256": BINDING_HASH,
                   "candidate_bank_sha256": bank_hash, "packet_inventory_sha256": file_sha256(packet / "packet-files.json"),
                   "private_zip_sha256": file_sha256(bundle), "private_zip_bytes": bundle.stat().st_size,
                   "recordings": len({c["request"]["scene"]["frame"]["recording_id"] for c in cases}),
                   "verified_original_photographs": len(cases), "original_photo_bytes": sum(
                       (packet / c["request"]["scene"]["frame"]["image_path"]).stat().st_size for c in cases),
                   "saved_current_frame_detections": total, "old_observations": 0, "distinct_new_input_bindings": len(cases),
                   "blank_forms": 2, "new_human_approvals": 0, "masks_supplied": 0,
                   "previously_unresolved_original_cases_excluded": 6, "blank_resolver_outcome": status}
        put(stage / "readiness.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stage.rename(output)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.installed, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
