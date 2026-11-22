"""Replay selected-vehicle geometry on two nonfinal recordings without new native VLM calls."""

import argparse
from collections import Counter
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

from tools.dropoff_passenger_handoff import session as check_journal
from tools.dropoff_projected_pair import CachedRoad, read_rows
from trackrag.dropoff.anchor_geometry import AnchorGeometry
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.model import GroundingConfig
from trackrag.dropoff.passenger import digest
from trackrag.dropoff.passenger_pair_choice import (
    GEOMETRY,
    InvalidSelection,
    choose,
    current_tracks,
    options,
    replay_choice_evidence,
    validate_choice_evidence,
)
from trackrag.dropoff.road_cache import mask_sha256, read_mask


REGISTERED = "bbdd0b98496ef72062a286ecc100ef3ffc7eb06c1ab595ead840b4a766b7aa0c"
DAYTIME = "2eb97a38141c55f9f60aa820ecd48c8e4d2b3c7aaafb54131c40599b5128fc2c"


def independent_rectangle(request, outcome, cache):
    scene = request["scene"]
    size = scene["frame"]["width"], scene["frame"]["height"]
    road = read_mask(cache, scene["provenance"]["image_sha256"], size)
    if road is None or mask_sha256(road) != outcome["provenance"]["road_geometry"]["road_mask_sha256"]:
        raise ValueError("Actual original mask differs from the predicted patch evidence")
    h, w = road.shape
    x, y, r, b = outcome["result"]["destination_box_xyxy"]
    cells = [v * den / dim for v, den, dim in zip((x, y, r, b), (w, h, w, h), (*size, *size))]
    if any(abs(v - round(v)) > 1e-7 for v in cells):
        raise ValueError("Reported rectangle is not aligned with the road raster")
    left, top, right, bottom = map(round, cells)
    visited = road[top:bottom, left:right]
    overlaps = []
    for row in current_tracks(request):
        obj_left, obj_top, obj_right, obj_bottom = row["bbox_xyxy"]
        if max(x, obj_left) < min(r, obj_right) and max(y, obj_top) < min(b, obj_bottom):
            overlaps.append(row["track_id"])
    if visited.size < 9 or not visited.all() or overlaps:
        raise ValueError("Reported patch includes unsupported road or a current road user")
    return {
        "original_road_cells_checked": int(visited.size),
        "all_cells_on_segformer_road": True,
        "current_road_user_intersections": overlaps,
    }


def study(journal, bank, geometry, cache):
    if (
        file_sha256(Path("experiments/dropoff/passenger-two-black-cars-protocol-20261122.json")) != REGISTERED
        or file_sha256(bank) != DAYTIME
    ):
        raise ValueError("Original nonfinal development sources or reference protocol changed")
    checked = check_journal(journal)
    submissions = [event["detail"] for event in read_rows(journal) if event["action"] == "submitted"]
    if len(submissions) != 2:
        raise ValueError("Use the original genuine two-utterance nonfinal F781 journal")
    original = submissions[-1]
    q = original["request"]
    frame = q["scene"]["frame"]
    if (frame["recording_id"], frame["frame_id"], q["scene"]["provenance"]["image_sha256"]) != (
        "video_054907",
        781,
        "c411cf6760695905334f32e5590083cf144543df1ca98e68aae55fb345eb583e",
    ):
        raise ValueError("Different source frame for F781")
    selected_wrong = original["outcome"]["provenance"]["selected_track_ids"]
    targets = [("previously_registered_f781_landmarks", q, [61108, 59268], p) for p in ("between", "behind")]
    targets.append(("replay_incorrect_original_selected_ids", q, selected_wrong, "behind"))
    other = next(
        row
        for row in read_rows(bank)
        if row["scene"]["frame"]["recording_id"] == "video_054604" and row["scene"]["frame"]["frame_id"] == 1
    )
    other = deepcopy(other)
    other.update(
        request_id="nonfinal-second-recording-pair-choice",
        text="Please let me out between those two vehicles.",
        dialogue=[],
        input_mode="typed",
    )
    other.pop("original_transcript", None)
    vehicles = options(other)
    # Fixed by observable input only. Do not search for successful geometry or inspect intended destinations.
    separated = [
        (b["bbox_xyxy"][0] - a["bbox_xyxy"][2], a["track_id"], b["track_id"])
        for a, b in combinations(vehicles, 2)
        if a["bbox_xyxy"][2] < b["bbox_xyxy"][0]
    ]
    if not separated:
        raise ValueError("Expected a separated detector pair in the second control recording")
    _, first, second = min(separated)
    for name, ids in [
        ("smallest_positive_horizontal_gap", [first, second]),
        ("leftmost_and_rightmost", [vehicles[0]["track_id"], vehicles[-1]["track_id"]]),
    ]:
        for relation in ("between", "behind"):
            targets.append((name, other, ids, relation))
    targets.append(("foreign_track_ids_from_f781", other, [61108, 59268], "behind"))
    model = SimpleNamespace(
        config=GroundingConfig(),
        geometry=AnchorGeometry.load(geometry, expected_sha=GEOMETRY),
        road=CachedRoad(cache),
    )
    rows = []
    for name, req, ids, placement in targets:
        f = req["scene"]["frame"]
        row = {
            "case": name,
            "recording_id": f["recording_id"],
            "frame_id": f["frame_id"],
            "request_sha256": digest(req),
            "image_sha256": req["scene"]["provenance"]["image_sha256"],
            "track_ids": ids,
            "placement": placement,
        }
        try:
            outcome = choose(model, req, ids, placement)
        except InvalidSelection as error:
            row.update(status="invalid_choice", reason=str(error))
            rows.append(row)
            continue
        validate_choice_evidence(req, outcome, ids, placement)
        verify = replay_choice_evidence(req, outcome, geometry, cache)
        src = outcome["provenance"]
        result = outcome["result"]
        row.update(
            status=result["status"],
            reason=src.get("withheld_reason"),
            geometry_mode=src.get("geometry_mode"),
            original_selected_boxes=[
                r["bbox_xyxy"] for r in src["passenger_selection"]["selected_observations"]
            ],
            observed_tracks_sha256=src["current_observations_sha256"],
            selection_sha256=src["selection_sha256"],
            geometry_replayed=verify,
            point=result["destination_point_xy"],
            area=result["destination_box_xyxy"],
        )
        if result["status"] == "proposed":
            row["independent_pixel_check"] = independent_rectangle(req, outcome, cache)
        rows.append(row)
    return {
        "kind": "scripted_passenger_landmark_cpu_controls",
        "selection_of_f781_was_informed_by_original_model_failure": True,
        "registered_original_case_sha256": REGISTERED,
        "f781_native_journal_sha256": checked["journal_sha256"],
        "f781_original_automatic_result": original["outcome"]["result"],
        "f781_original_automatic_sha256": digest(original["outcome"]),
        "other_recording_source_bank_sha256": DAYTIME,
        "trained_pair_geometry_sha256": GEOMETRY,
        "new_qwen_calls": 0,
        "passenger_confirmations_in_this_cpu_control": 0,
        "human_reference_authors": 0,
        "formal_final_cases": 0,
        "reference_intention_for_second_recording": None,
        "recordings": 2,
        "cases": len(rows),
        "dispositions": dict(Counter(row["status"] for row in rows)),
        "interpretation": "Analyst-selected labels exercise the same production method. Road-cell agreement checks the saved segmentation mask and observations. It does not measure rider intent, actual traffic safety, or successful autonomous recognition.",
        "records": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=Path("runs/passenger-landmarks/pair-empty-live-requests.jsonl"))
    p.add_argument("--geometry", type=Path, default=Path("models/passenger-feature/geometry.json"))
    p.add_argument("--road-cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = study(args.journal, args.bank, args.geometry, args.road_cache)
    with args.out.open("x", encoding="utf8", newline="\n") as dest:
        json.dump(result, dest, indent=2, allow_nan=False)
        dest.write("\n")
    print(
        json.dumps(
            {
                "sha256": file_sha256(args.out),
                "dispositions": result["dispositions"],
                "records": [
                    (r["case"], r["placement"], r["status"], r.get("point")) for r in result["records"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
