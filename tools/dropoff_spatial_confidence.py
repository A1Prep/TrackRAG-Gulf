"""Fit, predict and score a source-mask calibrated experimental dense-head selector."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from tools.dropoff_predict import read_requests
from tools.dropoff_spatial_fit import _iou
from trackrag.dropoff.data import file_sha256, inference_input_sha256, load_cases
from trackrag.dropoff.spatial_confidence import HEADS, choose, features, fit


ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"experiments/dropoff/passenger-spatial-confidence-protocol-20261122.json"
PROTOCOL_HASH="e8e2795df9e92be772a96a61cb3f096f994c6ff4e65529342e13019641dfd6aa"
CONDITIONS=(*HEADS,"selected","selected_confidence_030")


def write(path,value):
    with path.open("x",encoding="utf8",newline="\n") as f:
        f.write(json.dumps(value,indent=2,allow_nan=False,sort_keys=True)+"\n")


def required(condition, message):
    if not condition:
        raise ValueError(message)


def protocol():
    required(file_sha256(PROTOCOL)==PROTOCOL_HASH,"Frozen acquisition or modeling protocol differs")
    return json.loads(PROTOCOL.read_bytes())


def native_mask_result(candidate, case):
    if not candidate or candidate["status"]!="geometry":
        return False,0.
    p=candidate["destination_point_xy"]
    f=case["request"]["scene"]["frame"]
    x,y=p
    required(0<=x<f["width"] and 0<=y<f["height"],"Point outside source image")
    with Image.open(case["source"]["mask_path"]) as image:
        hit=image.convert("L").getpixel((int(x*image.width/f["width"]),int(y*image.height/f["height"])))!=0
    return hit,_iou(candidate["destination_box_xyxy"],case["target"]["destination_box_xyxy"])


def aggregate(outputs, cases):
    required([r["request_id"] for r in outputs]==[r["case_id"] for r in cases],"Case order/membership differs")
    stats={name:Counter() for name in CONDITIONS}
    selected=Counter()
    individual=[]
    for row,case in zip(outputs,cases,strict=True):
        required(row["image_sha256"]==case["source"]["image_sha256"] and
                 inference_input_sha256(case["request"],row["image_sha256"])==row["inference_input_sha256"],
                 "A saved prediction belongs to another image or instruction")
        selected[row["choice"]["head"] if row.get("choice") else "none"]+=1
        result={}
        for name in CONDITIONS:
            head=row.get("choice",{}).get("head") if name.startswith("selected") else name
            if name=="selected_confidence_030" and not row.get("choice",{}).get("meets_selected_threshold"):
                head=None
            cand=row.get("candidates",{}).get(head)
            hit,overlap=native_mask_result(cand,case)
            produced=bool(cand and cand["status"]=="geometry")
            stats[name].update(cases=1,proposals=int(produced),hits=int(hit),iou=overlap,errors=int("error" in row))
            result[name]={"proposed":produced,"hit":hit,"box_iou":overlap}
        individual.append({"request_id":row["request_id"],"result":result})
    scores={}
    for name,v in stats.items():
        scores[name]={"cases":v["cases"],"mask_hits_all":v["hits"],"proposals":v["proposals"],"operational_failures":v["errors"],
                      "mean_coarse_iou_all_missing_zero":v["iou"]/v["cases"],
                      "precision_among_proposals":v["hits"]/v["proposals"] if v["proposals"] else None}
    return {"metrics":scores,"selected_head_counts":dict(selected),"per_case":individual}


def fit_development(args):
    frozen=protocol()
    required(file_sha256(args.cases)==frozen["known_data"]["calibration_input_sha256"],"Unexpected calibration corpus")
    cases=load_cases(args.cases,allowed_partitions={"development"})
    inputs={}
    for name,k in zip(HEADS,("conditioned_predictions_sha256","blind_predictions_sha256"),strict=True):
        path=args.heads/name/"predictions.jsonl"
        required(file_sha256(path)==frozen["known_data"][k],"Development predictions differ from preregistration")
        inputs[name]=[json.loads(line) for line in path.read_bytes().splitlines()]
    examples=[];x=[];y=[];folds=[]
    for idx,case in enumerate(cases):
        candidates={name:rows[idx] for name,rows in inputs.items()}
        required(all(c["case_id"]==case["case_id"] and c["image_sha256"]==case["source"]["image_sha256"]
                     and c["checkpoint_sha256"]==frozen["head_conditions"][name] for name,c in candidates.items()),
                 "Different head, image or case in calibration predictions")
        f=case["request"]["scene"]["frame"]
        size=f["width"],f["height"]
        fold=int(sha256(("spatial-coverage-v1:"+case["scene_group"]).encode()).hexdigest(),16)%5
        for name,values in zip(HEADS,features(candidates,size),strict=True):
            if values is not None:
                x.append(values);y.append(int(native_mask_result(candidates[name],case)[0]));folds.append(fold)
        examples.append((case,fold,candidates,size))
    x,y,folds=np.asarray(x),np.asarray(y),np.asarray(folds)
    required(len(cases)==125 and len({c["scene_group"] for c in cases})==65 and set(folds)==set(range(5)),
             "Unexpected calibration split or missing scene fold")
    models={k:fit(x[folds!=k],y[folds!=k]) for k in range(5)}
    oof=[]
    for case,fold,candidates,size in examples:
        digest=case["source"]["image_sha256"]
        oof.append({"request_id":case["case_id"],"image_sha256":digest,"inference_input_sha256":inference_input_sha256(case["request"],digest),
                    "candidates":candidates,"choice":choose(models[fold],candidates,size),"held_out_fold":fold})
    scored=aggregate(oof,cases)
    output={"protocol_sha256":PROTOCOL_HASH,"condition_hashes":frozen["head_conditions"],"calibration_examples":len(cases),
            "calibration_groups":65,"fit_sha256":file_sha256(args.cases),"model":fit(x,y)}
    args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/"calibrator.json",output)
    write(args.out/"cross-fitted-scores.json",scored)
    write(args.out/"cross-fitted-predictions.json",oof)
    print(json.dumps(scored["metrics"],indent=2))


def predict(args):
    import torch
    from trackrag.dropoff.spatial import NoVisibleGround, SpatialGrounder, geometry
    frozen=protocol()
    required(not args.out.exists(),"Choose a fresh predictions file")
    torch.set_num_threads(2)
    trained=json.loads(args.model.read_bytes())
    required(trained["protocol_sha256"]==PROTOCOL_HASH and trained["condition_hashes"]==frozen["head_conditions"],
             "Calibration model belongs to a different comparison")
    models={name:SpatialGrounder(args.heads/name,vision_cache=args.vision_cache) for name in HEADS}
    required(all(m.metadata["weights_sha256"]==frozen["head_conditions"][name] for name,m in models.items()),
             "Dense head serialization differs from preregistered identity")
    rows=[]
    for req in read_requests(args.requests):
        f=req["scene"]["frame"]
        digest=req["scene"]["provenance"]["image_sha256"]
        row={"request_id":req["request_id"],"image_sha256":digest,"inference_input_sha256":inference_input_sha256(req,digest)}
        required(not req["scene"]["tracks"] and file_sha256(f["image_path"])==digest,"Expected original zero-detector scene")
        try:
            candidates={}
            with Image.open(f["image_path"]) as im:
                size=im.size
                required(size==(f["width"],f["height"]),"Decoded image dimensions differ")
                feature=np.rint(models[HEADS[0]].vision.encode(im)*255).astype(np.float32)/255
            tensor=torch.from_numpy(feature[None])
            for name,m in models.items():
                words=m.vocab.encode(req["text"])
                if name=="language_blind":
                    words.fill(0)
                with torch.inference_mode():
                    logits,extent=m.model(tensor,torch.from_numpy(words[None]))
                try:
                    cand=geometry(logits[0].numpy(),extent[0].numpy(),feature,size)
                    candidates[name]={"status":"geometry",**cand}
                except NoVisibleGround:
                    candidates[name]={"status":"no_visible_road"}
            row.update(candidates=candidates,choice=choose(trained["model"],candidates,size))
        except Exception as error:
            row["error"]={"type":type(error).__name__,"message":str(error)}
        rows.append(row)
    write(args.out,{"source":"original_image_and_instruction_only","protocol_sha256":PROTOCOL_HASH,
                   "requests_sha256":file_sha256(args.requests),"calibrator_sha256":file_sha256(args.model),
                   "conditions":frozen["head_conditions"],"references_read":False,"rows":rows})
    print(f"Saved {len(rows)} predictions, {sum('error' in r for r in rows)} errors")


def score(args):
    p=protocol()
    bank=json.loads(args.bank.read_bytes())
    predictions=json.loads(args.predictions.read_bytes())
    if args.deployed_run is not None:
        required((args.deployed_run/"run.json").is_file(),"Matched, sealed Qwen outcomes are required before public scoring")
    else:
        required(args.cpu_only,"Supply sealed --deployed-run or explicitly declare --cpu-only")
    required(bank["protocol_sha256"]==PROTOCOL_HASH and predictions["protocol_sha256"]==PROTOCOL_HASH
             and predictions["references_read"] is False and predictions["conditions"]==p["head_conditions"]
             and predictions["requests_sha256"]==bank["request_file_sha256"]
             and file_sha256(args.references)==bank["reference_file_sha256"],"Mismatched frozen reference/prediction bank")
    deployed=None
    scope="Actual deployed Qwen was unavailable for a same-input comparison."
    if args.deployed_run is not None:
        from tools.dropoff_audit_passenger_release import audit_batch, json_file
        from tools.dropoff_passenger_references_public import score as score_public
        request_path=args.bank.parent/"inputs"/"requests.jsonl"
        freeze=json_file(ROOT/"experiments/dropoff/passenger-runtime-freeze-20261122.json")
        release=json_file(ROOT/"experiments/dropoff/passenger-release-protocol-20261122.json")
        ids=[row["request_id"] for row in predictions["rows"]]
        freeze["input_banks"]["public"]={"file_sha256":bank["request_file_sha256"],"ids":ids}
        _,_,receipt=audit_batch("trained-public",args.deployed_run.parent,request_path,freeze,release,
                                args.vision_cache,batch_dir=args.deployed_run)
        native=score_public(args.bank.parent,args.deployed_run/"outcomes.jsonl")
        required(native["cases"]==len(ids) and native["outcomes_sha256"]==receipt["outcomes_sha256"]
                 and not native["unexpected_output_ids"],"Different Qwen inputs or primary scorer")
        deployed={"counts":receipt["counts"],"outcomes_sha256":receipt["outcomes_sha256"],
                  "requests_sha256":receipt["requests_sha256"],"primary_rows":native["rows"]}
        deployed["metrics"]={"cases":native["cases"],"mask_hits_all":native["point_hits_all_requests"],
            "proposals":native["destinations_provided"],"operational_failures":sum(native["failures"].values()),
            "mean_coarse_iou_all_missing_zero":native["mean_coarse_box_iou_all_requests_missing_as_zero"],
            "precision_among_proposals":native["point_hits_all_requests"]/native["destinations_provided"] if native["destinations_provided"] else None}
        scope="Actual trained Qwen from the deployed release ran on the same fresh images, instructions and zero-detector inputs."
    cases=load_cases(args.references,allowed_partitions={"development"})
    required([c["source"]["scene_token"] for c in cases]==bank["source_scenes"]
             and len(cases)==len({c["scene_group"] for c in cases})==bank["cases"],"Fresh source scenes differ")
    stats=aggregate(predictions["rows"],cases)
    write(args.out,{"protocol_sha256":PROTOCOL_HASH,"predictions_sha256":file_sha256(args.predictions),
                    "calibrator_sha256":predictions["calibrator_sha256"],"participant_mask_only":True,
                    "deployed_comparison":deployed,"comparison_scope":scope,**stats})
    print(json.dumps(stats["metrics"],indent=2))
    if deployed:
        print("Matched deployed Qwen",json.dumps(deployed["metrics"],indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name,keys in {"fit":("cases","heads","out"),"predict":("requests","heads","vision-cache","model","out"),
                      "score":("predictions","references","bank","out")}.items():
        cmd=sub.add_parser(name)
        for k in keys:
            cmd.add_argument("--"+k,type=Path,required=True)
        if name=="score":
            choices=cmd.add_mutually_exclusive_group(required=True)
            choices.add_argument("--deployed-run",type=Path,help="Sealed normal deployed Qwen run on these same requests")
            choices.add_argument("--cpu-only",action="store_true",help="Declare the normal Qwen comparison unavailable")
            cmd.add_argument("--vision-cache",type=Path,default=Path(os.environ.get("LOCALAPPDATA","."))/"TrackRAG-method-cache"/"segformer-b0")
    args=parser.parse_args()
    {"fit":fit_development,"predict":predict,"score":score}[args.command](args)
