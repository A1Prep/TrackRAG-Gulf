"""Audit the optional overlapping-vehicle passenger grounder or replay a saved application turn."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from tools.dropoff_passenger_handoff import session as verify_interaction
from tools.dropoff_predict import read_requests
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.landmarks import landmark_messages, resolve_image
from trackrag.dropoff.passenger import digest, load_passenger_model
from trackrag.dropoff.projected_pair import GEOMETRY, MOTOR, Ineligible, neighbor_geometry, run, supported_patch
from trackrag.dropoff.road_cache import read_mask


REGISTRATION = Path("experiments/dropoff/passenger-occluded-pair-protocol-20261122.json")
REGISTERED_SHA256 = "440b8ac92a800b27ab94a5ed16fc3d6c9197eda9b750d39d994b201d54c407f7"


def write(path, payload):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


class CachedRoad:
    def __init__(self, path):
        self.path = path

    def mask(self, image, sha):
        mask = read_mask(self.path, sha, image.size)
        if mask is None:
            raise Ineligible("no_cached_current_road_mask")
        return mask


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def controlled_sensitivity(request, model, road):
    scene = request["scene"]
    frame = scene["frame"]
    im, info = resolve_image(request, model.config)
    with im:
        size = im.size
    _, index = landmark_messages(request)
    current = list(index.values())
    mask = read_mask(road, info["sha256"], size)
    counts = Counter()
    eligible = []
    for first, second in combinations((t for t in current if t["class_name"] in MOTOR), 2):
        try:
            group = neighbor_geometry((first, second), current, size)
        except Ineligible as exc:
            counts[str(exc)] += 1
            continue
        row = {"ids": [first["track_id"], second["track_id"]], "geometry": group}
        eligible.append(row)
        if mask is None:
            counts["no_cached_current_road_mask"] += 1
            row["status"] = "no_cached_current_road_mask"
            continue
        try:
            patch = supported_patch(mask, size, model.geometry.pair_regression, group, current)
        except Ineligible as exc:
            counts[str(exc)] += 1
            row["status"] = str(exc)
            continue
        counts["supported_geometric_probe"] += 1
        row.update(status="supported_geometric_probe", patch=patch)
    return {
        "recording": frame["recording_id"],
        "frame": frame["frame_id"],
        "source_image_sha256": info["sha256"],
        "current_tracks_sha256": digest(current),
        "current_observations": len(current),
        "motor_pairs": sum(counts.values()),
        "counts": dict(counts),
        "geometrically_eligible_pairs": eligible,
    }


def controls(model, road):
    if file_sha256(REGISTRATION) != REGISTERED_SHA256:
        raise ValueError("The development protocol changed after registration")
    registry = json.loads(REGISTRATION.read_bytes())
    attempts, scenes = [], {}
    for cohort in registry["nonfinal_controls"]:
        inp, out = Path(cohort["inputs"]), Path(cohort["outputs"])
        if file_sha256(inp) != cohort["inputs_sha256"] or file_sha256(out) != cohort["outputs_sha256"]:
            raise ValueError("The preregistered nonfinal inputs or original outcomes changed")
        inputs, outputs = read_requests(inp), read_rows(out)
        if len(inputs) != len(outputs) or len(inputs) != cohort["attempts"]:
            raise ValueError("Nonfinal controls have a different denominator")
        for q, o in zip(inputs, outputs, strict=True):
            if q["request_id"] != o["request_id"]:
                raise ValueError("Control outcomes changed order or identity")
            new, audit = run(q, o, model)
            attempts.append(
                {
                    "input_bank": inp.name,
                    "request_id": q["request_id"],
                    "request_sha256": digest(q),
                    "original_outcome_sha256": digest(o),
                    "checked_outcome_sha256": digest(new),
                    "original_status": (o.get("result") or {}).get("status", "operational_error"),
                    "audit": audit,
                }
            )
            f = q["scene"]["frame"]
            frame_key = (f["recording_id"], f["frame_id"])
            if frame_key[0] not in ("video_054604", "video_054907") or frame_key in {
                ("video_054907", 781),
                ("video_054604", 42),
            }:
                continue
            _, index = landmark_messages(q)
            if index:
                key = (q["scene"]["provenance"]["image_sha256"], digest(list(index.values())))
                scenes.setdefault(key, q)
    geometry = [controlled_sensitivity(q, model, road) for q in scenes.values()]
    totals = Counter()
    for row in geometry:
        totals.update(row["counts"])
    return {
        "registration_sha256": file_sha256(REGISTRATION),
        "source_files_sha256": {
            str(p): file_sha256(p)
            for p in (Path("trackrag/dropoff/projected_pair.py"), Path("tools/dropoff_projected_pair.py"))
        },
        "control_attempts": len(attempts),
        "controls_canonical_json_unchanged": sum(
            r["original_outcome_sha256"] == r["checked_outcome_sha256"] for r in attempts
        ),
        "control_dispositions": dict(Counter(r["audit"].get("reason", r["audit"]["status"]) for r in attempts)),
        "current_geometry_sha256": model.geometry.sha256,
        "cpu_geometry_counts": dict(totals),
        "diagnostic_sources": len(scenes),
        "cached_mask_use_only": True,
        "new_qwen_calls": 0,
        "genuine_passenger_confirmations": 0,
        "formal_task1_final_cases": 0,
        "meaning": "A geometry probe enumerates actual current predicted vehicle boxes without passenger identity evidence. "
        "A road-supported numerical region neither identifies a passenger's intention nor verifies vehicle paint or a legal stop.",
        "attempts": attempts,
        "geometry": geometry,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--geometry", required=True, type=Path)
    p.add_argument("--road-cache", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    ops = p.add_subparsers(dest="mode", required=True)
    ops.add_parser("controls")
    r = ops.add_parser("turn")
    r.add_argument("--journal", required=True, type=Path)
    r.add_argument("--request-id", required=True)
    r.add_argument("--verify-with-local-model", action="store_true")
    args = p.parse_args()
    if file_sha256(REGISTRATION) != REGISTERED_SHA256:
        raise ValueError("The development protocol changed after registration")
    if file_sha256(args.geometry) != GEOMETRY:
        raise ValueError("Original pretrained pair regression and complete geometry file are required")
    model = load_passenger_model(args.config, geometry_path=args.geometry, vision_cache=args.road_cache)
    model.road = CachedRoad(args.road_cache)
    try:
        if args.mode == "controls":
            evidence = controls(model, args.road_cache)
        else:
            original_session = verify_interaction(args.journal)
            events = [
                row
                for row in read_rows(args.journal)
                if row["action"] == "submitted" and row["detail"]["request"]["request_id"] == args.request_id
            ]
            if len(events) != 1:
                raise ValueError("A unique, original saved passenger submission is required")
            (event,) = events
            req, original = event["detail"]["request"], event["detail"]["outcome"]
            if original is None or event["view"]["result"] != original.get("result"):
                raise ValueError("The original passenger interaction did not accept this native outcome")
            outcome, audit = run(req, original, model, verify=args.verify_with_local_model)
            evidence = {
                "registration_sha256": file_sha256(REGISTRATION),
                "source_journal_sha256": original_session["journal_sha256"],
                "original_outcome_sha256": digest(original),
                "result": outcome,
                "audit": audit,
                "actual_passenger_confirmation": False,
            }
        write(args.out, evidence)
    finally:
        model.close()


if __name__ == "__main__":
    main()
