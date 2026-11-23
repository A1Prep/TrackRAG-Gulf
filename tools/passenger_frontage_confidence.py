"""Score an existing Molmo2 frontage pin from its frozen image-model evidence."""
import argparse
from hashlib import sha256
import json
from math import exp
from pathlib import Path
import re

import numpy as np

from trackrag.dropoff.parking_evidence import _authenticate, _decode_map


FEATURES = ("pull_in_margin", "pull_in_percentile", "curb_relative", "landmark_above",
            "qwen_scene_open", "qwen_crop_bay", "parking_fraction", "road_fraction")
POINT_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
SURFACE_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
QUERIES = ("an empty parking bay for a car", "a driveway, vehicle entrance or paved pull-in",
           "the lane of traffic directly ahead of the camera", "a raised pedestrian sidewalk",
           "a parked car or truck", "a curb or edge of the road", "a gate, fence, chain or barrier")


def standardized(value, background):
    return float(np.clip((value - background.mean()) / max(float(background.std()), 0.25), -4, 4))


def surface_fallbacks(path):
    if path is None:
        return {}
    records = (json.loads(line) for line in path.read_bytes().splitlines())
    return {r["case"]: r["signed_packet"] for r in records if r["arm"] == "release" and r.get("signed_packet")}


def evidence(request, point_record, maps, check, surface_packet=None):
    frame = request["scene"]["frame"]
    source = request["scene"]["provenance"]["image_sha256"]
    words = request["text"]
    if (point_record["image_sha256"] != source or str(maps["source"]) != source
            or point_record["words"] != words or str(maps["words"]) != words
            or point_record["revision"] != POINT_REVISION or str(maps["revision"]) != SURFACE_REVISION):
        raise ValueError("Point or surface maps belong to another image, request or checkpoint")
    queries = maps["queries"].tolist()
    planes = maps["logits"].astype(np.float64)
    if (len(queries) != 8 or tuple(queries[:7]) != QUERIES or not queries[7]
            or planes.shape != (8,96,144) or not np.isfinite(planes).all()):
        raise ValueError("Unexpected CLIPSeg output")
    point = point_record["point_1000"]
    if point is None:
        return {"point_1000": None, "veto": "no_point", "features": None}
    if len(point) != 2 or any(type(v) is not int or not 0 <= v <= 1000 for v in point):
        raise ValueError("Invalid specialist point")
    _authenticate(check)
    body = check["body"]
    if (body["image_sha256"] != source or body["words"] != words or body["point_1000"] != point
            or body["model"] != "trained_qwen3_vl_8b" or body["recipe"] != "frontage_point_crop_check_v1"
            or [call["stage"] for call in body["calls"]] != ["scene", "surface"]):
        raise ValueError("Qwen scene/crop classifications belong to different evidence")
    scene, surface = (call["response"] for call in body["calls"])
    if scene not in {"OPEN","BLOCKED","UNCLEAR"} or surface not in {"BAY","LANE","WALK","BLOCKED","UNKNOWN"}:
        raise ValueError("Unknown scene or surface classification")
    x, y = (min(n - 1, v * n / 1000) for v, n in zip(point, (frame["width"], frame["height"])))
    for track in request["scene"]["tracks"]:
        if track["frame_id"] == frame["frame_id"]:
            left, top, right, bottom = track["bbox_xyxy"]
            if left <= x <= right and top <= y <= bottom:
                return {"point_1000": point, "veto": "road_user_box", "features": None}
    map_record = body.get("class_map")
    if map_record is None and surface_packet is not None:
        _authenticate(surface_packet)
        archived = surface_packet["body"]
        if (archived["image_sha256"] != source or archived["recipe"] != "trained_qwen_place_with_parking_surface_v3"
                or archived["models"]["parking_segmenter"] != "4772b6bf101d91f2534c106dc524d906aeb3c68a"):
            raise ValueError("Archived parking map belongs to another image or model")
        map_record = archived["class_map"]
    if map_record is None:
        return {"point_1000": point, "veto": "no_surface_map", "features": None}
    classes = _decode_map(map_record, frame)
    ch, cw = classes.shape
    cx, cy = min(cw-1,point[0]*cw//1000), min(ch-1,point[1]*ch//1000)
    center = int(classes[cy,cx])
    if center not in (10,13):
        return {"point_1000": point, "veto": "non_drivable_surface", "features": None}
    rx, ry = max(2,round(cw*.015)), max(2,round(ch*.015))
    class_patch = classes[max(0,cy-ry):cy+ry+1,max(0,cx-rx):cx+rx+1]
    _, h, w = planes.shape
    col, row = min(w-1,int(point[0]*w/1000)), min(h-1,int(point[1]*h/1000))
    local = planes[:,max(0,row-2):row+3,max(0,col-3):col+4].mean(axis=(1,2))
    pull_in = max(local[:2])
    lower = planes[:,h//3:,:]
    landmark = planes[-1,max(0,row-h//3):row+1,max(0,col-w//8):min(w,col+w//8+1)]
    values = [float(pull_in-max(local[[2,3,4,6]])), float((np.max(lower[:2],axis=0)<=pull_in).mean()),
              standardized(local[5],lower[5]), standardized(np.quantile(landmark,0.85),planes[-1]),
              float(scene=="OPEN"), float(surface=="BAY"), float((class_patch==10).mean()), float((class_patch==13).mean())]
    return {"point_1000": point, "veto": "uncorroborated_road" if center == 13 and surface != "BAY" else None,
            "features": values}


def predict(features, model):
    if model["features"] != list(FEATURES):
        raise ValueError("Unexpected confidence model feature order")
    if features["veto"]:
        return {"accept":False, "score":None, "reason":features["veto"]}
    values = (np.asarray(features["features"])-model["mean"])/model["scale"]
    z = float(model["intercept"] + values @ model["coefficients"])
    score = 1/(1+exp(-max(-60.,min(60.,z))))
    return {"accept":score >= model["threshold"], "score":score, "reason":"learned_confidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model","requests","points","maps","checks","out"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fallback-surfaces", type=Path, help="Optional signed surface receipts for older request banks")
    args = parser.parse_args()
    model = json.loads(args.model.read_bytes())
    digest = sha256(args.model.read_bytes()).hexdigest()
    fallbacks = surface_fallbacks(args.fallback_surfaces)
    outputs = []
    for line in args.requests.read_bytes().splitlines():
        request = json.loads(line)
        case = request["request_id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case):
            raise ValueError("Invalid request identifier")
        point = json.loads((args.points/(case+".json")).read_bytes())
        check = json.loads((args.checks/(case+".json")).read_bytes())
        with np.load(args.maps/(case+".npz"),allow_pickle=False) as maps:
            observed = evidence(request,point,maps,check,fallbacks.get(case))
        decision = predict(observed,model)
        outputs.append({"id":case,"image_sha256":point["image_sha256"],"model_sha256":digest,**decision,
                        "point_1000":observed["point_1000"] if decision["accept"] else None})
    with args.out.open("x",encoding="utf8",newline="\n") as file:
        for output in outputs:
            file.write(json.dumps(output,sort_keys=True,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
