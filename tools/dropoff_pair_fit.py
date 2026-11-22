"""Fit a weakly supervised two-landmark region head on the frozen public training scenes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.dropoff_anchor_fit import DEV_SHA, TRAIN_SHA
from tools.dropoff_public_candidates import read_candidates
from trackrag.dropoff.data import load_cases, validate_disjoint_groups
from trackrag.dropoff.pair_geometry import fit, predict

def candidates(row,size):
    w,h = size
    boxes = []
    for item in row:
        if not item["class"].startswith("vehicle.") or item["score"] < .2:
            continue
        x,y,bw,bh = item["bbox"]
        box = [max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)]
        if box[2]-box[0] >= .012*w and box[3]-box[1] >= .015*h:
            boxes.append(box)
    return boxes


def weak_examples(cases, bank):
    examples = []
    for case in cases:
        r,src = case["request"],case["source"]
        candidate = bank[str(src["source_index"])]
        if candidate["command_token"] != src["command_token"]:
            raise ValueError("Public detector bank differs from the source command")
        f = r["scene"]["frame"]
        size = [f["width"],f["height"]]
        px,py = case["target"]["destination_point_xy"]
        eligible = [b for b in candidates(candidate["predictions"],size)
                    if abs(b[3]-py) <= .13*size[1] and py >= b[1]-.01*size[1]]
        left = [b for b in eligible if b[2]+.005*size[0] < px]
        right = [b for b in eligible if b[0]-.005*size[0] > px]
        if not left or not right:
            continue
        def distance(box,edge):
            return abs(box[edge]-px)/size[0]+1.5*abs(box[3]-py)/size[1]
        pair = [min(left,key=lambda b:distance(b,2)),min(right,key=lambda b:distance(b,0))]
        if pair[1][0]-pair[0][2] < .03*size[0]:
            continue
        examples.append({"id":case["case_id"],"anchors":pair,"size":size,"point":[px,py],
                         "box":case["target"]["destination_box_xyxy"],"mask":src["mask_path"]})
    return examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("splits","predictions","single-head","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args = p.parse_args()
    raw = args.predictions.read_bytes()
    bank = read_candidates(args.predictions)
    subsets = {}
    for name, sha in (("train",TRAIN_SHA),("development",DEV_SHA)):
        path = args.splits/f"{name}.jsonl"
        b = path.read_bytes()
        if sha256(b).hexdigest() != sha:
            raise ValueError("Changed frozen partition")
        subsets[name] = load_cases(path,allowed_partitions={name},payload=b)
    validate_disjoint_groups(subsets["train"],subsets["development"])
    train = weak_examples(subsets["train"],bank)
    dev = weak_examples(subsets["development"],bank)
    model = fit(train)
    hits = 0
    for row in dev:
        point,_ = predict(model,row["anchors"],row["size"])
        with Image.open(row["mask"]) as im:
            mask = np.asarray(im.convert("L"))
            hits += bool(mask[int(point[1]*im.height/row["size"][1]),int(point[0]*im.width/row["size"][0])])
    single = args.single_head.read_bytes()
    combined = json.loads(single)
    combined["pair_regression"] = model
    combined["single_head_sha256"] = sha256(single).hexdigest()
    args.out.mkdir(parents=True,exist_ok=False)
    final = args.out/"geometry.json"
    final.write_text(json.dumps(combined,sort_keys=True,allow_nan=False),encoding="utf8")
    receipt = {"kind":"weak_geometric_pair_training","train_manifest_sha256":TRAIN_SHA,
               "public_detector_bank_sha256":sha256(raw).hexdigest(),"train_ids":[r["id"] for r in train],
               "train_count":len(train),"dev_oracle_point_in_mask":hits,"dev_oracle_count":len(dev),
               "pair_selection_uses_reference_point_for_training_and_oracle_diagnostic_only":True,
               "human_paired_intents":False,"single_head_sha256":combined["single_head_sha256"],
               "geometry_sha256":sha256(final.read_bytes()).hexdigest()}
    (args.out/"fit.json").write_text(json.dumps(receipt,indent=2),encoding="utf8")
    print({k:v for k,v in receipt.items() if k != "train_ids"})


if __name__ == "__main__":
    main()
