"""Fit a portable destination-region head on the frozen public training partition."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

from trackrag.dropoff.anchor_geometry import AnchorGeometry
from trackrag.dropoff.data import load_cases, validate_disjoint_groups
from tools.dropoff_bootstrap_regseg import SCENE_METADATA_URL, SCENE_METADATA_SHA256


TRAIN_SHA = "79dfd904b155b1c537170aaba443254da37445c2e8a2f4380baaddd8520bc42b"
DEV_SHA = "533cbcf9a2059b2890a606e896d9a216d5fc663606693c7dc8121b9ddf222556"


def joined(cases, commands):
    examples = []
    for case in cases:
        req, src = case["request"], case["source"]
        row = commands[src["command_token"]]
        if (row["scene_token"] != src["scene_token"] or row["sample_token"] != src["sample_token"]
                or row["command"].strip() != req["text"].strip()
                or row["t2c_img"] != Path(req["scene"]["frame"]["image_path"]).name):
            raise ValueError("Public referring-object annotation does not match this source command")
        w, h = (req["scene"]["frame"][k] for k in ("width", "height"))
        x,y,bw,bh = row["2d_box"]
        anchor = [max(0,x), max(0,y), min(w,x+bw), min(h,y+bh)]
        if anchor[0] >= anchor[2] or anchor[1] >= anchor[3]:
            raise ValueError("Source landmark has no visible intersection")
        examples.append({"id":case["case_id"],"text": req["text"], "size": [w, h], "anchor": anchor,
                         "point": case["target"]["destination_point_xy"],
                         "box": case["target"]["destination_box_xyxy"]})
    return examples


def fit(split_dir, metadata, out):
    metadata = (Path(metadata).read_bytes() if metadata else
                urllib.request.urlopen(SCENE_METADATA_URL, timeout=30).read(16*1024*1024))
    if sha256(metadata).hexdigest() != SCENE_METADATA_SHA256:
        raise ValueError("Referring-landmark metadata digest differs from pinned public source")
    commands = {r["command_token"]:r for r in json.loads(metadata)["commands"]}
    cases = {}
    for name, expected in (("train", TRAIN_SHA), ("development", DEV_SHA)):
        path = Path(split_dir)/f"{name}.jsonl"
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != expected:
            raise ValueError(f"{name} split differs from the frozen public corpus")
        cases[name] = load_cases(path, allowed_partitions={name}, payload=payload)
    validate_disjoint_groups(cases["train"], cases["development"])
    train = joined(cases["train"], commands)
    development = joined(cases["development"], commands)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    totals = {}
    for name, use_text in (("conditioned", True), ("geometry_only", False), ("retrieved", True)):
        fitted = AnchorGeometry.fit(train, use_text)
        if name == "retrieved":
            fitted = AnchorGeometry.fit_retrieval(train,fitted)
        path = out/f"{name}.json"
        path.write_text(json.dumps(fitted, sort_keys=True, allow_nan=False), encoding="utf8")
        model = AnchorGeometry.load(path)
        hits, ious = [], []
        for case, e in zip(cases["development"], development):
            point, box = model.predict([e["anchor"]], e["size"], e["text"])
            with Image.open(case["source"]["mask_path"]) as m:
                px = int(point[0]*m.width/e["size"][0]), int(point[1]*m.height/e["size"][1])
                hits.append(bool(np.asarray(m.convert("L"))[px[1],px[0]]))
            gold = np.array(e["box"])
            overlap = np.maximum(0, np.minimum(box[2:],gold[2:])-np.maximum(box[:2],gold[:2])).prod()
            area = np.prod(np.array(box[2:])-box[:2])+np.prod(gold[2:]-gold[:2])
            ious.append(float(overlap/(area-overlap)))
        totals[name] = {"checkpoint_sha256": model.sha256, "rows":len(hits),
                        "hits":sum(hits), "mean_iou":float(np.mean(ious))}
    receipt = {"kind":"development_with_reference_landmark_oracle_not_application_accuracy",
               "public_train_cases":len(train), "public_train_groups":len({c["scene_group"] for c in cases["train"]}),
               "train_sha256":TRAIN_SHA, "development_sha256":DEV_SHA,
               "landmark_metadata_sha256":SCENE_METADATA_SHA256, "source":SCENE_METADATA_URL,
               "single_landmark_training_only":True, "conditions":totals}
    (out/"fit.json").write_text(json.dumps(receipt,indent=2),encoding="utf8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(fit(a.splits, a.metadata, a.out), indent=2))
