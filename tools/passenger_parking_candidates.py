"""Build a bounded list of research points from current vehicle observations and street segmentation."""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
from itertools import combinations
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from tools.dropoff_predict import _reject_constant, _unique_object, read_requests
from tools.passenger_road_points import render_choices
from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.projected_pair import MOTOR

MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"
MAX_OPTIONS = 8


def _current_vehicles(request):
    scene = request["scene"]
    if any(t["frame_id"] != scene["frame"]["frame_id"] for t in scene["tracks"]):
        raise ValueError("Parking candidates require current-frame observations")
    return sorted(
        (t for t in scene["tracks"] if t["class_name"] in MOTOR),
        key=lambda t: -(t["bbox_xyxy"][2] - t["bbox_xyxy"][0])
                      * (t["bbox_xyxy"][3] - t["bbox_xyxy"][1]),
    )


def _contact(box):
    left, top, right, bottom = box
    return ((left + right) / 2, bottom - .08 * (bottom - top), right - left, bottom - top)


def _between_candidates(surface, vehicles, pair_ids=None):
    # Boxes are in segmentation pixels. Each crop is local to this pair. A large
    # semantic parking strip is clipped before finding connected regions.
    rows, columns = surface.shape
    anchors = vehicles[:14] if pair_ids is None else [v for v in vehicles if v["track"]["track_id"] in pair_ids]
    for first, second in combinations(anchors, 2):
        one, two = first["bbox"], second["bbox"]
        ax, ay, aw, ah = _contact(one)
        bx, by, bw, bh = _contact(two)
        mean_w, mean_h = (aw + bw) / 2, (ah + bh) / 2
        if max(ah, bh) > 3.5 * min(ah, bh) or abs(ax-bx) > 3.2 * max(aw, bw) or abs(ay-by) > 1.6 * max(ah, bh):
            continue
        vx, vy = (bx-ax)/mean_w, (by-ay)/mean_h
        length_squared = vx*vx + vy*vy
        if length_squared < .12:
            continue

        # An intervening car in the same perspective row breaks adjacency.
        blocked = False
        for other in vehicles:
            if other is first or other is second:
                continue
            cx, cy, _, ch = _contact(other["bbox"])
            ox, oy = (cx-ax)/mean_w, (cy-ay)/mean_h
            fraction = (ox*vx + oy*vy) / length_squared
            if (.12 < fraction < .88 and .58*min(ah,bh) < ch < 1.55*max(ah,bh)
                    and math.hypot(ox-fraction*vx,oy-fraction*vy)<.38):
                blocked = True
                break
        if blocked:
            continue

        x0 = max(0, math.floor(min(ax,bx) - .38*mean_w))
        x1 = min(columns, math.ceil(max(ax,bx) + .38*mean_w))
        y0 = max(0, math.floor(min(ay,by) - .68*max(ah,bh)))
        y1 = min(rows, math.ceil(max(ay,by) + .28*max(ah,bh)))
        if x0 >= x1 or y0 >= y1:
            continue
        y, x = np.mgrid[y0:y1, x0:x1]
        u, v = (x+.5-ax)/mean_w, (y+.5-ay)/mean_h
        along = (u*vx + v*vy)/length_squared
        offset = np.hypot(u-along*vx, v-along*vy)
        # A visible ground patch can extend sideways around the more distant
        # car in perspective. The semantic mask excludes car bodies and curbs.
        between = ((along >= .10) & (along <= .90) & (offset <= .60)
                   & (y+.5 <= ay+along*(by-ay)+.26*mean_h))
        region = (surface[y0:y1,x0:x1] & between).astype("uint8")
        count, components, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        pair_size = math.sqrt(aw*ah*bw*bh)
        for comp in range(1, count):
            area = int(stats[comp,cv2.CC_STAT_AREA])
            if area < max(9, .003*mean_w*mean_h):
                continue
            active = components == comp
            inside = cv2.distanceTransform(np.pad(active.astype("uint8"),1),cv2.DIST_L2,5)[1:-1,1:-1]
            if float(inside.max()) < 2:
                continue
            # A point in the supported interior. The center of two vehicle
            # boxes by itself can land on a vehicle or in a traffic lane.
            centrality = np.exp(-2*(along-.5)**2 - 2*offset**2)
            score = inside * centrality
            iy, ix = np.unravel_index(score.argmax(),score.shape)
            rank = pair_size/(1 + math.sqrt(length_squared)) * min(1,area/(.08*mean_w*mean_h))
            rank *= float(centrality[iy,ix])
            yield (rank, (x0+ix+.5, y0+iy+.5), first, second, area)


def options_for(request, classes, surface_name, pair_ids=None):
    request = validate_request(request)
    frame = request["scene"]["frame"]
    if (not isinstance(classes, np.ndarray) or classes.ndim != 2 or classes.dtype != np.uint8
            or not classes.size or surface_name not in ("parking","all-road")):
        raise ValueError("Invalid class map or surface name")
    if pair_ids is not None and (len(pair_ids) != 2 or len(set(pair_ids)) != 2):
        raise ValueError("An identified pair requires two distinct current vehicle IDs")
    tracks = _current_vehicles(request)
    if not re.search(r"\bbetween\b", request["text"], re.IGNORECASE):
        return []
    mw, mh = classes.shape[1], classes.shape[0]
    sx, sy = frame["width"]/mw, frame["height"]/mh
    axes = (sx,sy,sx,sy)
    vehicles = [
        {"track":t,"bbox":[v/a for v,a in zip(t["bbox_xyxy"],axes)]}
        for t in tracks
    ]
    surface = (classes==10) if surface_name=="parking" else np.isin(classes,(10,13))
    result = []
    for _, (x,y), a,b,area in sorted(_between_candidates(surface, vehicles, pair_ids),key=lambda x:-x[0]):
        point = [x*sx, y*sy]
        if any(math.hypot((x-old["point"][0]/sx)/mw,(y-old["point"][1]/sy)/mh)<.018 for old in result):
            continue
        result.append({"point":point,"choice":chr(ord("A")+len(result)),"tracks":[a["track"]["track_id"],b["track"]["track_id"]],
                       "anchor_boxes":[a["track"]["bbox_xyxy"],b["track"]["bbox_xyxy"]],"region_area_map_pixels":area})
        if len(result)==MAX_OPTIONS:
            break
    return result


def export(requests_path, maps, output, surface, font_file=None, *, anchor_records_path=None):
    requests_path, maps, output = Path(requests_path), Path(maps), Path(output)
    if requests_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Grounding requests exceed the snapshot byte budget")
    snapshot = requests_path.read_bytes()
    requests = read_requests(requests_path,payload=snapshot)
    raw_receipts = (maps/"receipt.json").read_bytes()
    receipts = json.loads(raw_receipts, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    by_sha = {r["sha256"]:r for r in receipts}
    if any(r["mask_sha"] != by_sha[r["sha256"]]["mask_sha"] for r in receipts):
        raise ValueError("The same photograph has conflicting class maps")
    anchor_bytes = None
    anchor_by_id = {}
    if anchor_records_path is not None:
        from tools.passenger_parking_anchors import read_anchor_records
        anchor_bytes = Path(anchor_records_path).read_bytes()
        anchor_by_id = read_anchor_records(requests, anchor_bytes)
    checked = []
    for req in requests:
        frame = req["scene"]["frame"]
        photo = Path(frame["image_path"])
        sha = sha256(photo.read_bytes()).hexdigest()
        r = by_sha.get(sha)
        size = (frame["width"],frame["height"])
        if (not r or r["sha256"]!=req["scene"].get("provenance",{}).get("image_sha256")
                or r["revision"]!=REVISION or r["model"]!=MODEL or tuple(r["size"])!=size
                or r["labels"].get("10","").lower()!="parking" or r["labels"].get("13","").lower()!="road"
                or r.get("file")!=sha+".png"):
            raise ValueError("Parking map does not match the selected model and original photo")
        mask = maps/(sha+".png")
        mask_bytes = mask.read_bytes()
        if sha256(mask_bytes).hexdigest()!=r["mask_sha"]:
            raise ValueError("Parking class map changed")
        with Image.open(BytesIO(mask_bytes)) as im:
            classes=np.array(im)
        with Image.open(photo) as im:
            if im.size!=size or list(classes.shape)!=r["mask_shape"]:
                raise ValueError("Class map or photograph dimensions changed")
        decision = anchor_by_id.get(req["request_id"])
        if decision is None:
            options = options_for(req,classes,surface)
        elif decision["state"] == "matched":
            options = options_for(req,classes,surface,decision["track_ids"])
        else:
            options = []
        checked.append((req,photo,size,sha,r,options,decision))
    output.mkdir(parents=True,exist_ok=False)
    result=[]
    for i,(req,photo,size,sha,r,options,decision) in enumerate(checked,1):
        marked = output/f"{i:04}-{sha[:12]}.jpg"
        source_bytes = photo.read_bytes()
        if sha256(source_bytes).hexdigest() != sha:
            raise ValueError("Original photo changed during export")
        render_choices(BytesIO(source_bytes),options,size,marked,font_file=font_file)
        record = {"id":req["request_id"],"request":req["text"],"original":str(photo),"marked":str(marked.resolve()),
                       "marked_sha256":sha256(marked.read_bytes()).hexdigest(),"sha256":sha,"original_size":list(size),
                       "options":options,"surface":surface,"is_pair":bool(re.search(r"\bbetween\b",req["text"],re.IGNORECASE)),"mask_sha256":r["mask_sha"]}
        if decision is not None:
            record.update(anchor_state=decision["state"],pair_ids=decision["track_ids"],
                          anchor_iou=decision["ious"],resolved_boxes_xyxy=decision["boxes"])
        result.append(record)
    choices = output/"choices.json"
    choices.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    manifest = {"model":MODEL,"revision":REVISION,"surface":surface,"max_candidates":MAX_OPTIONS,
                "requests_sha256":sha256(snapshot).hexdigest(),"class_receipts_sha256":sha256(raw_receipts).hexdigest(),
                "source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),"choices_sha256":sha256(choices.read_bytes()).hexdigest(),
                "font_sha256":sha256(Path(font_file).read_bytes()).hexdigest() if font_file else None,
                "automatic_vehicle_observations":True,"destination_labels_used":False}
    if anchor_bytes is not None:
        (output/"anchor-records.json").write_bytes(anchor_bytes)
        manifest.update(geometry_mode="model-anchors",anchor_receipts_sha256=sha256(anchor_bytes).hexdigest())
    else:
        manifest["geometry_mode"]="all-pairs"
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests",type=Path,required=True,help="Annotation-free grounding requests JSONL")
    parser.add_argument("--class-maps",type=Path,required=True,help="Directory containing pinned Mask2Former class maps and receipt.json")
    parser.add_argument("--out",type=Path,required=True,help="New output directory")
    parser.add_argument("--surface",choices=("parking","all-road"),default="parking")
    parser.add_argument("--font-file",type=Path)
    args=parser.parse_args()
    rows=export(args.requests,args.class_maps,args.out,args.surface,args.font_file)
    print(f"Exported {sum(len(r['options']) for r in rows)} points across {len(rows)} photographs")


if __name__=="__main__":
    main()
