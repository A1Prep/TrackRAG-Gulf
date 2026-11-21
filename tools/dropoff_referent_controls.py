"""Score a frozen referring-object head through its normal runtime interface, including contrastive controls."""
import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from tools.dropoff_anchor_fit import DEV_SHA
from tools.dropoff_bootstrap_regseg import SCENE_METADATA_SHA256
from tools.dropoff_fit_referent import checked, corners, write
from tools.dropoff_public_candidates import read_candidates
from trackrag.dropoff.data import load_cases
from trackrag.dropoff.referent import ReferentRanker, overlap


def proposals(frame, items, size):
    tracks = []
    for i, item in enumerate(items):
        box = corners(item["bbox"], size)
        if box is not None:
            tracks.append({"recording_id": frame["recording_id"], "frame_id": frame["frame_id"], "track_id":i,
                           "source":"predicted", "class_name":item["class"], "bbox_xyxy":box})
    return tracks


def best_iou(result, target):
    objects = result["alternatives"]
    return float(overlap([objects[0]["bbox_xyxy"]], target)[0]) if objects else 0.


def emt_smoke(model, requests):
    # Only the predeclared two familiar training/development recordings may be read.
    permitted = {("video_054604",150), ("video_054907",75)}
    outputs = []
    for line in requests.read_bytes().splitlines():
        req = json.loads(line)
        frame = req["scene"]["frame"]
        if (frame["recording_id"], frame["frame_id"]) not in permitted:
            continue
        if sha256(Path(frame["image_path"]).read_bytes()).hexdigest() != req["scene"]["provenance"]["image_sha256"]:
            raise ValueError("EMT frame bytes changed")
        with Image.open(frame["image_path"]) as image:
            start = time.perf_counter()
            r = model.rank(image,req["text"],req["scene"]["tracks"],frame=frame)
            elapsed = time.perf_counter() - start
        outputs.append({"id":req["request_id"],"input":req["text"],"recording_id":frame["recording_id"],
                        "frame":frame["frame_id"],"image_sha256":req["scene"]["provenance"]["image_sha256"],
                        "current_candidates":r["candidate_count"],"distinct":r["distinct_candidate_count"],
                        "null_score":r["null_score"],"alternatives":r["alternatives"][:4],
                        "seconds_including_image_decode":elapsed})
    if len(outputs) != 4:
        raise ValueError("Existing EMT smoke cohort changed")
    return outputs


def run(args):
    if args.out.exists():
        raise ValueError("Choose a fresh output filename")
    model = ReferentRanker.load(args.model, expected_sha256=args.expected_sha)
    cases = load_cases(args.development, allowed_partitions={"development"}, payload=checked(args.development, DEV_SHA))
    bank = read_candidates(args.proposals)
    commands = {r["command_token"]:r for r in json.loads(checked(args.metadata,SCENE_METADATA_SHA256))["commands"]}
    cases.sort(key=lambda r:r["case_id"])
    output, by_frame = [], defaultdict(list)
    for i, case in enumerate(cases):
        s, req = case["source"], case["request"]
        frame = req["scene"]["frame"]
        original = commands[s["command_token"]]
        scene_bank = bank[str(s["source_index"])]
        if (original["command"].strip() != req["text"].strip() or scene_bank["command_token"] != s["command_token"]
                or original["sample_token"] != s["sample_token"] or original["scene_token"] != s["scene_token"]):
            raise ValueError("Official source differs from the fixed case")
        size = frame["width"], frame["height"]
        if sha256(Path(frame["image_path"]).read_bytes()).hexdigest() != s["image_sha256"]:
            raise ValueError("Public original image no longer matches manifest")
        tracks = proposals(frame,scene_bank["predictions"],size)
        anchor = corners(original["2d_box"],size)
        # After deterministic sort, use the circular next instruction from a different image.
        j = (i+1) % len(cases)
        while cases[j]["source"]["image_sha256"] == s["image_sha256"]:
            j = (j+1) % len(cases)
        with Image.open(frame["image_path"]) as image:
            factual = model.rank(image,req["text"],tracks,frame=frame)
            shuffled = model.rank(image,cases[j]["request"]["text"],tracks,frame=frame)
        best = factual["alternatives"][:1]
        row = {"id":case["case_id"],"image_sha256":s["image_sha256"],"text":req["text"],
               "annotated_reference_for_scoring":anchor,"top_box": best[0]["bbox_xyxy"] if best else None,
               "top_track":best[0]["track_id"] if best else None,
               "top1_iou":best_iou(factual,anchor),"wrong_instruction_id":cases[j]["case_id"],
               "wrong_text_top1_iou":best_iou(shuffled,anchor)}
        by_frame[s["image_sha256"]].append(row)
        output.append(row)
    pairs = []
    for group in by_frame.values():
        for a,b in combinations(group,2):
            if overlap([a["annotated_reference_for_scoring"]],b["annotated_reference_for_scoring"])[0] >= .5:
                continue
            distinct = a["top_box"] and b["top_box"] and overlap([a["top_box"]],b["top_box"])[0] < .5
            pairs.append({"case_a":a["id"],"case_b":b["id"],
                          "different_top_boxes":bool(distinct),"both_objects_correct":a["top1_iou"]>=.5 and b["top1_iou"]>=.5})
    total = {"rows":len(output),"unique_images":len(by_frame),"normal_runtime_top1_correct":sum(r["top1_iou"]>=.5 for r in output),
             "wrong_instruction_same_pixels_top1_correct":sum(r["wrong_text_top1_iou"]>=.5 for r in output),
             "natural_same_image_different_referent_pairs":len(pairs),
             "changed_top_box_given_different_natural_commands":sum(v["different_top_boxes"] for v in pairs),
             "both_natural_referents_correct":sum(v["both_objects_correct"] for v in pairs)}
    result = {"checkpoint_sha256": model.sha256, "public_development_sha256":DEV_SHA,
              "postfit_diagnostic": True, "weights_frozen_before_diagnostics":True,
              "protocol":"All fixed public development rows; circular adjacent command from a different image in sorted case-ID order. All same-image natural command pairs with annotated reference IoU <.5.",
              "counts":total,"natural_pairs":pairs,"case_predictions":output,
              "emt_known_recordings_functionality_only":emt_smoke(model,args.emt)}
    write(args.out,result)
    return total


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for label in ("model","development","proposals","metadata","emt","out"):
        parser.add_argument("--"+label,type=Path,required=True)
    parser.add_argument("--expected-sha",required=True)
    print(json.dumps(run(parser.parse_args()),indent=2))
