"""Independently audit sealed passenger batches and recompute their predeclared external endpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

from PIL import Image

from tools.dropoff_passenger_handoff import bound_outcome
from tools.dropoff_passenger_references import gold_and_requests
from tools.dropoff_predict import read_requests
from trackrag.dropoff.data import file_sha256, inference_input_sha256, load_cases
from trackrag.dropoff.landmarks import landmark_messages, parse_landmark
from trackrag.dropoff.model import GroundingConfig
from trackrag.dropoff.pair_resolution import replay_pair_decision
from trackrag.dropoff.passenger import instruction_context
from trackrag.dropoff.referent import ReferentRanker
from trackrag.dropoff.referent_policy import clarify_reference
from trackrag.dropoff.road_cache import ROAD_RECIPE_SHA256, mask_sha256, read_mask


LABELS = ("trained-night", "pretrained-night", "trained-public", "pretrained-public")
ROOT = Path(__file__).resolve().parents[1]


def must(condition, message):
    if not condition:
        raise ValueError(message)


def json_file(path):
    return json.loads(path.read_bytes())


def require_completed(external):
    absent = [label for label in LABELS if not (external/label/"run.json").is_file()]
    if absent:
        raise ValueError("All four completed manifests are required before reading references. Missing " + ", ".join(absent))
    sequence = json_file(external/"execution.json")
    must(sequence.get("role") == "blind_inference_only" and sequence.get("end"), "Missing closed inference-only queue")
    must([r["label"] for r in sequence["runs"]] == list(LABELS), "The four saved conditions differ from the freeze")
    must(all(isinstance(r.get("code"),int) and r.get("end") for r in sequence["runs"]), "Unfinished model process")
    return sequence


def intersection_over_union(a,b):
    left,top=max(a[0],b[0]),max(a[1],b[1])
    right,bottom=min(a[2],b[2]),min(a[3],b[3])
    common=max(0.,right-left)*max(0.,bottom-top)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-common
    return common/union


def night_endpoints(refs, requests, predictions):
    counts=Counter()
    coarse=0.
    individual=[]
    for key, annotation in refs.items():
        target=annotation["target"]
        expected=target["status"]
        pred=predictions[key]
        actual=pred["result"]["status"] if pred["ok"] else "operational_failure"
        clear=expected=="proposed"
        row={"id":key,"expected":expected,"actual":actual,"strict_hit":False,"within_5pct_diagonal":False,"coarse_iou":0.}
        counts["status_matches"]+=actual==expected
        counts["clear"]+=clear
        counts["unresolved"]+=not clear
        counts["proposals"]+=actual=="proposed"
        counts["false_proposals_on_unresolved"]+=actual=="proposed" and not clear
        counts["operational_failures"]+=not pred["ok"]
        if clear and actual=="proposed":
            x,y=pred["result"]["destination_point_xy"]
            l,t,r,b=target["destination_box_xyxy"]
            row["strict_hit"]=l<=x<r and t<=y<b
            f=requests[key]["scene"]["frame"]
            threshold=.05*math.hypot(f["width"],f["height"])
            row["within_5pct_diagonal"]=math.dist([x,y],target["destination_point_xy"])<=threshold
            row["coarse_iou"]=intersection_over_union(pred["result"]["destination_box_xyxy"],target["destination_box_xyxy"])
            coarse+=row["coarse_iou"]
        counts["strict_hits_all_clear"]+=row["strict_hit"]
        counts["within_5pct_all_clear"]+=row["within_5pct_diagonal"]
        individual.append(row)
    return {"rows":len(individual),**counts,"mean_coarse_iou_all_clear_missing_zero":coarse/counts["clear"],
            "reference_role":"AI-authored semantic and coarse spatial reference, zero human reviews"},individual


def public_endpoints(cases, predictions):
    hits,proposed,coverage,errors=0,0,0.,0
    individual=[]
    for case in cases:
        key=case["case_id"]
        pred=predictions[key]
        result=pred["result"] if pred["ok"] else {}
        point=result.get("destination_point_xy")
        zone=result.get("destination_box_xyxy")
        ready=point is not None and zone is not None
        hit,iou=False,0.
        if ready:
            f=case["request"]["scene"]["frame"]
            with Image.open(case["source"]["mask_path"]) as mask:
                actual=mask.convert("L")
                px=int(point[0]*actual.width/f["width"])
                py=int(point[1]*actual.height/f["height"])
                hit=actual.getpixel((px,py))>0
            iou=intersection_over_union(zone,case["target"]["destination_box_xyxy"])
        hits+=hit
        proposed+=ready
        coverage+=iou
        errors+=not pred["ok"]
        individual.append({"id":key,"point_in_original_mask":hit,"proposed":ready,"coarse_iou":iou})
    return {"rows":len(cases),"proposals":proposed,"mask_hits_all":hits,"operational_failures":errors,
            "mean_coarse_iou_all_missing_zero":coverage/len(cases),
            "reference_role":"Original participant region mask; no semantic or confirmation labels"},individual


def compare_scorers(label, metrics, audit_rows, primary):
    if label.endswith("-night"):
        comparisons = {
            "rows":primary["requested"], "status_matches":primary["status_matches"],
            "clear":primary["clear_reference_count"], "false_proposals_on_unresolved":primary["proposals_on_nonproposal_cases"],
            "proposals":primary["proposal_count"], "strict_hits_all_clear":primary["point_in_authored_box_count"],
            "within_5pct_all_clear":primary["normalized_tolerance_hits_all_clear"]["0.05"],
            "mean_coarse_iou_all_clear_missing_zero":primary["mean_box_iou_all_clear_missing_as_zero"],
        }
        by_id = {r["request_id"]:r for r in primary["per_case"]}
        for independent in audit_rows:
            original=by_id[independent["id"]]
            must(independent["strict_hit"]==original["point_in_authored_box"]
                 and math.isclose(independent["coarse_iou"],original["box_iou"] or 0.,abs_tol=1e-12),
                 "Scorers disagree on an authored night reference")
    else:
        comparisons = {
            "rows":primary["cases"], "proposals":primary["destinations_provided"], "mask_hits_all":primary["point_hits_all_requests"],
            "mean_coarse_iou_all_missing_zero":primary["mean_coarse_box_iou_all_requests_missing_as_zero"],
        }
        by_id = {r["request_id"]:r for r in primary["rows"]}
        for independent in audit_rows:
            original=by_id[independent["id"]]
            must(independent["point_in_original_mask"]==original["point_in_public_mask"]
                 and math.isclose(independent["coarse_iou"],original["coarse_box_iou"],abs_tol=1e-12),
                 "Scorers disagree on an original public mask")
    comparisons["operational_failures"]=sum(primary["failures"].values())
    must(not primary["unexpected_output_ids"] and all(
             math.isclose(metrics[name],value,rel_tol=1e-10,abs_tol=1e-12) for name,value in comparisons.items()),
         "Primary scorer and independently derived totals differ")
    return {"all_per_case_geometry_and_predeclared_totals_agree":True, "case_count":len(audit_rows)}


def audit_batch(label, external, requests_path, frozen, protocol, vision_cache, *, batch_dir=None):
    condition,cohort=label.split("-")
    bank=frozen["input_banks"][cohort]
    batch_dir=external/label if batch_dir is None else Path(batch_dir)
    settings=json_file(batch_dir/"settings.json")
    manifest=json_file(batch_dir/"run.json")
    raw_path=batch_dir/"outcomes.jsonl"
    must(settings=={k:v for k,v in manifest.items() if k not in ("errors","outcomes_sha256")},"Live settings changed")
    must(file_sha256(raw_path)==manifest["outcomes_sha256"] and file_sha256(requests_path)==bank["file_sha256"]==manifest["requests_sha256"],
         "Request or prediction digest changed")
    must(manifest["requested"]==len(bank["ids"]) and manifest["resolver_execution"]=="live"
         and manifest["reference_access"] is False and manifest["saved_decisions_sha256"] is None
         and manifest["verify_landmarks"] is False,"Mismatch with live, reference-free frozen conditions")
    components=protocol["components"]["files"]
    must(settings["geometry_sha256"]==components["geometry.json"] and settings["referent_sha256"]==components["referent.json"],
         "Unexpected geometry or referent weights")
    cfg=dict(settings["config"])
    for name in ("model_dir","cache_dir","adapter_dir"):
        cfg.pop(name)
    target={k:v for k,v in protocol["runtime"].items() if k not in ("model_dir","cache_dir","adapter_dir")}
    must(cfg==target,"Nonpath inference configuration changed")
    must(bool(settings["config"]["adapter_dir"])==(condition=="trained"),"Incorrect adapter treatment")
    receipt_files = set(frozen["source_sha256_utf8_lf"]) - {
        "trackrag/dropoff/contracts.py", "trackrag/dropoff/data.py", "trackrag/dropoff/spatial.py"}
    must(set(settings["source"])==receipt_files,"Incomplete runtime source inventory in saved batch")
    for name,digest in settings["source"].items():
        p=ROOT/name
        original=p.read_bytes()
        verified=sha256(original.replace(b"\r\n",b"\n")).hexdigest()
        must(verified==frozen["source_sha256_utf8_lf"][name]
             and (sha256(original).hexdigest()==digest or verified==digest),"Model code differs from the saved run or freeze")
    requests={r["request_id"]:r for r in read_requests(requests_path)}
    records=[json.loads(line) for line in raw_path.read_bytes().splitlines()]
    must([r["request_id"] for r in records]==bank["ids"]==list(requests),"Cohort includes missing, duplicated or reordered requests")
    roles=Counter()
    pids=set()
    ranker=ReferentRanker.load(ROOT/"models/passenger-feature/referent.json",expected_sha256=components["referent.json"])
    adapter_cfg=GroundingConfig.from_mapping(settings["config"])
    for row in records:
        key=row["request_id"]
        request=requests[key]
        source=row.get("provenance",{})
        bound=bound_outcome(row,request)
        roles["prior_dialogue" if request["dialogue"] else "initial_instruction"]+=1
        must(source.get("resolver_execution")=="live","A replayed or unidentified outcome entered a live cohort")
        if not bound["ok"]:
            roles["operational_errors"]+=1
            continue
        must(source.get("injected") is not True and source["method"]=="qwen_landmark_region"
             and source.get("independent_model_proposal") is not False,
             "Software-only or passenger-marked outcome entered independent model condition")
        solver=source["resolver"]
        backend=solver["backend_details"]
        expected_adapter=components["adapter/adapter_model.safetensors"]
        if condition=="trained":
            must(solver["adapter"]["files"]["adapter_model.safetensors"]["sha256"]==expected_adapter,
                 "Different LoRA entered trained selector")
        else:
            must(solver["adapter"] is None,"LoRA entered pretrained selector")
        must(backend["weight_identity"]["verified_weights"] is True and
             backend["weight_identity"]["verified_metadata"] is True and
             backend["effective_resident_layers"]==frozen["production_resident_layers"] and
             backend["suffix_layers"]==cfg["suffix_layers"],"Unverified model weights or different GPU layer profile")
        pids.add(backend["worker_pid"])
        if "pair_resolution" in solver:
            pair=replay_pair_decision(request,source["image"],solver["pair_resolution"],adapter_cfg)
            must(pair==source["pair_decision"],"Saved pair answer differs from its independent raw-subquery replay")
            for part in solver["pair_resolution"]["subqueries"]:
                b=part["resolver"]["backend_details"]
                must(b["weight_identity"]["verified_weights"] and b["effective_resident_layers"]==frozen["production_resident_layers"],
                     "Pair subquery omitted actual verified native weights")
            roles["verified_pair_subqueries"]+=len(solver["pair_resolution"]["subqueries"])
        else:
            messages,ids=landmark_messages(request)
            original=solver.get("decision")
            must(messages==solver["messages"] and original==parse_landmark(solver["raw_text"],ids,request["scene"]["frame"]),
                 "Raw local landmark words differ from their semantic receipt")
        current=[t for t in request["scene"]["tracks"] if t["frame_id"]==request["scene"]["frame"]["frame_id"]]
        current_ids={t["track_id"] for t in current}
        ranking=source.get("referent_ranking")
        if ranking:
            must(ranking["checkpoint_sha256"]==components["referent.json"] and ranking["candidate_count"]==len(current),
                 "Different candidate checkpoint or track count")
            must({a["track_id"] for a in ranking["alternatives"]}<=current_ids,"Ranker used a different frame's track")
            text=instruction_context(request)
            f=request["scene"]["frame"]
            with Image.open(f["image_path"]) as image:
                repeat=ranker.rank(image,text,request["scene"]["tracks"],frame=f)
                advice=clarify_reference(text,repeat,image.size)
            must([a["track_id"] for a in ranking["alternatives"]]==[a["track_id"] for a in repeat["alternatives"]]
                 and math.isclose(ranking["null_score"], repeat["null_score"],abs_tol=1e-8)
                 and all(math.isclose(a["score"],b["score"],abs_tol=1e-8) for a,b in zip(ranking["alternatives"],repeat["alternatives"]))
                 and advice==source.get("referent_clarification"),"Clarification evidence differs from a fresh CPU model/policy replay")
            roles["ranker_consulted"]+=1
        if source.get("referent_clarification"):
            must(bound["result"]["status"]=="needs_clarification","Ask-only policy generated a destination")
            roles["ranker_requested_clarification"]+=1
        if bound["result"]["status"]=="proposed":
            frame=request["scene"]["frame"]
            size=frame["width"],frame["height"]
            road=source["road_evidence"]
            mask=read_mask(vision_cache,bound["image_sha256"],size)
            must(mask is not None and road["road_recipe_sha256"]==ROAD_RECIPE_SHA256
                 and mask_sha256(mask)==road["road_mask_sha256"],"Proposed point lacks its exact original-frame road mask")
            x,y=bound["result"]["destination_point_xy"]
            must(bool(mask[int(y*mask.shape[0]/size[1]),int(x*mask.shape[1]/size[0])]),"Proposed point is outside predicted road")
            must(road["current_exclusion_boxes"]==[t["bbox_xyxy"] for t in current]
                 and all(not (l<=x<r and t<=y<b) for l,t,r,b in road["current_exclusion_boxes"]),
                 "Missing current-frame exclusion or point inside an observed object")
            must(set(source["selected_track_ids"])<=current_ids,"Wrong-frame selected landmark")
            roles["original_road_and_current_exclusions_verified"]+=1
    must(roles["operational_errors"]==manifest["errors"],"Published error count differs")
    return requests,{r["request_id"]:r for r in records},{"outcomes_sha256":manifest["outcomes_sha256"],
            "requests_sha256":manifest["requests_sha256"],"counts":dict(roles),"worker_pids":sorted(pids)}


def run(args):
    summary_path = getattr(args,"summary",None)
    if args.out.exists() or (summary_path is not None and (summary_path.exists() or summary_path.resolve()==args.out.resolve())):
        raise ValueError("Choose fresh, separate audit output paths")
    sequence=require_completed(args.external)
    protocol=json_file(args.protocol)
    freeze=json_file(args.runtime_freeze)
    must(file_sha256(args.protocol)==freeze["original_freeze_sha256"],"Original release freeze differs")
    frozen_sources = dict(freeze["source_sha256_utf8_lf"])
    frozen_sources["trackrag/dropoff/model.py"] = freeze["native_model_wrapper_sha256_utf8_lf"]
    for filename, expected in frozen_sources.items():
        actual = sha256((ROOT/filename).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
        must(actual==expected,f"Current checkout differs from external release freeze: {filename}")
    must(datetime.fromisoformat(sequence["start"])>datetime.fromisoformat(freeze["frozen_utc"]),
         "Inference precedes the published runtime freeze")
    paths={"night":args.night_requests,"public":args.public_corpus/"inputs"/"requests.jsonl"}
    live,receipts={},{}
    for label in LABELS:
        cohort=label.split("-")[1]
        source,preds,receipt=audit_batch(label,args.external,paths[cohort],freeze,protocol,args.vision_cache)
        live[label]=(source,preds)
        receipts[label]=receipt
    # This boundary deliberately follows successful verification of all four manifests.
    refs,inputs,_=gold_and_requests(args.night_bank,"night",paths["night"])
    meta=json_file(args.public_corpus/"manifest.json")
    public_ref=args.public_corpus/"corpus"/"references.jsonl"
    must(file_sha256(public_ref)==meta["reference_file_sha256"],"Original public mask-reference manifest differs")
    cases=load_cases(public_ref,allowed_partitions={"development"})
    must(len(cases)==len({r["scene_group"] for r in cases})==protocol["public_cohort"]["n"]
         and {r["case_id"] for r in cases}==set(freeze["input_banks"]["public"]["ids"])
         and [r["source"]["scene_token"] for r in cases]==meta["source_scenes"]
         and [r["source"]["source_index"] for r in cases]==meta["source_indices"],"Public group membership differs")
    for c in cases:
        request=live["trained-public"][0][c["case_id"]]
        image=c["source"]["image_sha256"]
        must(c["source"]["dataset"]=="Talk2Car-RegSeg" and c["review"]["status"]=="public_annotation"
             and set(c["target"])=={"destination_box_xyxy","destination_point_xy"},"Unexpected public reference role")
        must(file_sha256(request["scene"]["frame"]["image_path"])==image
             and file_sha256(c["source"]["mask_path"])==c["source"]["mask_sha256"]
             and not request["scene"]["tracks"]
             and inference_input_sha256(request,image)==inference_input_sha256(c["request"],image),
             "Public pixels, region annotation or zero-track request differs")
    for label in LABELS:
        source,pred=live[label]
        saved=args.external/label/"outcomes.jsonl"
        if label.endswith("-night"):
            metrics,rows=night_endpoints(refs,source,pred)
            from tools.dropoff_passenger_references import score as score_night
            primary=score_night(args.night_bank,"night",paths["night"],saved)
        else:
            metrics,rows=public_endpoints(cases,pred)
            from tools.dropoff_passenger_references_public import score as score_public
            primary=score_public(args.public_corpus,saved)
        receipts[label]["primary_scorer_cross_check"]=compare_scorers(label,metrics,rows,primary)
        receipts[label]["metrics"]=metrics
        receipts[label]["rows"]=rows
    result={"kind":"independent_external_passenger_audit","audit_utc":datetime.now(timezone.utc).isoformat(),
            "protocol_sha256":file_sha256(args.protocol),"runtime_freeze_sha256":file_sha256(args.runtime_freeze),
            "all_four_outcome_files_sealed_and_verified_before_reference_read":True,
            "inference_completed_at":sequence["end"],"conditions":receipts}
    with args.out.open("x",encoding="utf8",newline="\n") as out:
        json.dump(result,out,indent=2,sort_keys=True,allow_nan=False)
    if summary_path is not None:
        public_summary = published_summary(result, file_sha256(args.out))
        with summary_path.open("x",encoding="utf8",newline="\n") as out:
            out.write(json.dumps(public_summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({label:value["metrics"] for label,value in receipts.items()},indent=2))
    return result


def published_summary(audit, evidence_sha256):
    return {
        "kind":"independent_frozen_passenger_evaluation",
        "audit_sha256":evidence_sha256,
        "release_protocol_sha256":audit["protocol_sha256"],
        "runtime_freeze_sha256":audit["runtime_freeze_sha256"],
        "inference_completed_utc":audit["inference_completed_at"],
        "all_four_outcome_files_sealed_and_verified_before_reference_read":
            audit["all_four_outcome_files_sealed_and_verified_before_reference_read"],
        "design":{
            "conditions":"Matched 8B pretrained and public LoRA selectors with the same geometry, road support and clarification policy.",
            "night":"Twenty instructions on six original frames from one independently held-out EMT recording. Semantic status and coarse spatial labels were authored and checked by AI. Zero human reviews.",
            "public":"Sixteen original Talk2Car-RegSeg participant region masks from sixteen separately selected, previously unused scenes. Passenger requests have no detector tracks. No status or confirmation labels.",
            "missing_destinations":"Zero credit on the full geometry denominator. Operational errors are retained.",
            "scope":"A proposed area always requires passenger confirmation. Road support is a model prediction. No driving safety or human acceptance was measured."
        },
        "conditions":{
            label:{key:row[key] for key in ("metrics","counts","outcomes_sha256","requests_sha256","primary_scorer_cross_check")}
            for label,row in audit["conditions"].items()
        },
    }


if __name__=="__main__":
    import os
    p=argparse.ArgumentParser(description=__doc__)
    defaults={"external":"runs/passenger-landmarks/external","protocol":"experiments/dropoff/passenger-release-protocol-20261122.json",
              "runtime-freeze":"experiments/dropoff/passenger-runtime-freeze-20261122.json",
              "night-bank":"runs/passenger-references/blind-third","night-requests":"runs/passenger-references/night-inputs/requests.jsonl",
              "public-corpus":"runs/passenger-references/public-independent"}
    for name,default in defaults.items():
        p.add_argument("--"+name,type=Path,default=ROOT/default)
    p.add_argument("--vision-cache",type=Path,default=Path(os.environ.get("LOCALAPPDATA","."))/"TrackRAG-method-cache"/"segformer-b0")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--summary",type=Path,help="Optional fresh location for aggregate metrics without private case records")
    run(p.parse_args())
