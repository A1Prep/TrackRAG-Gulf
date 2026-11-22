"""Publish aggregate matched fresh-scene results and checksum the original JSON-only evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image

from tools.dropoff_pack_matched import encoded, json_archive
from tools.dropoff_score_methods import point_hit
from tools.dropoff_spatial_confidence import CONDITIONS, PROTOCOL, PROTOCOL_HASH, required
from trackrag.dropoff.data import file_sha256, load_cases
from trackrag.vision.geometry import iou


def read(path):
    return json.loads(path.read_bytes())


def publish(fresh, calibration, results, out, bundle):
    required(not out.exists() and not bundle.exists(),"Choose fresh aggregate and evidence destinations")
    held=read(results)
    qwen=held["deployed_comparison"]
    queue=read(fresh/"deployed-comparison.json")
    required(queue["state"]=="complete" and queue["end_utc"] and qwen and queue["outcomes_sha256"]==qwen["outcomes_sha256"]
             and queue["requests_sha256"]==qwen["requests_sha256"] and queue["frozen_comparison_sha256"]==PROTOCOL_HASH,
             "A sealed actual release run on precisely the same inputs is required")
    outdir=fresh/"deployed-trained-12"
    cpu_file=fresh/"cpu-predictions.json"
    cpu=read(cpu_file)
    meta=read(fresh/"manifest.json")
    required(held["protocol_sha256"]==meta["protocol_sha256"]==file_sha256(PROTOCOL)==PROTOCOL_HASH
             and file_sha256(cpu_file)==held["predictions_sha256"]
             and file_sha256(calibration/"calibrator.json")==cpu["calibrator_sha256"]==held["calibrator_sha256"]
             and file_sha256(outdir/"outcomes.jsonl")==qwen["outcomes_sha256"]
             and file_sha256(outdir/"run.json")==queue["run_sha256"]
             and file_sha256(fresh/"inputs/requests.jsonl")==meta["request_file_sha256"]==cpu["requests_sha256"]==qwen["requests_sha256"],
             "Frozen model, inputs or results differ")
    references=fresh/"corpus/references.jsonl"
    required(file_sha256(references)==meta["reference_file_sha256"],"Original source reference bank differs")
    cases=load_cases(references,allowed_partitions={"development"})
    qwen_by_id={r["request_id"]:r for r in qwen["primary_rows"]}
    qwen_raw={r["request_id"]:r for r in map(json.loads,(outdir/"outcomes.jsonl").read_bytes().splitlines())}
    required(set(qwen_raw)==set(qwen_by_id) and len(qwen_raw)==len(cases),"Missing or duplicate live deployed outcomes")
    crosses={name:{"both_hit":0,"only_dense_hit":0,"only_deployed_hit":0,"neither_hit":0} for name in CONDITIONS}
    totals={name:Counter() for name in (*CONDITIONS,"deployed_trained_qwen")}
    for prediction,scored,case in zip(cpu["rows"],held["per_case"],cases,strict=True):
        key=case["case_id"]
        required(key==scored["request_id"]==prediction["request_id"] and key in qwen_by_id,"Row bindings differ")
        f=case["request"]["scene"]["frame"]
        with Image.open(case["source"]["mask_path"]) as im:
            mask=im.convert("L")
            qr=qwen_raw[key]
            reply=qr.get("result",{}) if qr["ok"] else {}
            point=reply.get("destination_point_xy")
            qwen_hit=bool(point is not None and point_hit(point,mask,f["width"],f["height"]))
            overlap=iou(reply["destination_box_xyxy"],case["target"]["destination_box_xyxy"]) if point is not None else 0.
            required(qwen_hit==qwen_by_id[key]["point_in_public_mask"] and (point is not None)==qwen_by_id[key]["destination_provided"]
                     and abs(overlap-qwen_by_id[key]["coarse_box_iou"])<1e-10,"Independent deployed-model mask scorer disagrees")
            totals["deployed_trained_qwen"].update(hits=qwen_hit,proposals=point is not None,iou=overlap,errors=not qr["ok"])
            for name in CONDITIONS:
                head=prediction.get("choice",{}).get("head") if name.startswith("selected") else name
                if name=="selected_confidence_030" and not prediction.get("choice",{}).get("meets_selected_threshold"):
                    head=None
                cand=prediction.get("candidates",{}).get(head) or {}
                point=cand.get("destination_point_xy")
                native=bool(point is not None and point_hit(point,mask,f["width"],f["height"]))
                overlap=iou(cand["destination_box_xyxy"],case["target"]["destination_box_xyxy"]) if point is not None else 0.
                required(native==scored["result"][name]["hit"] and abs(overlap-scored["result"][name]["box_iou"])<1e-10,
                         "The second source-mask scorer disagrees")
                totals[name].update(hits=native,proposals=point is not None,iou=overlap,errors="error" in prediction)
                col=("both_hit" if qwen_hit else "only_dense_hit") if native else ("only_deployed_hit" if qwen_hit else "neither_hit")
                crosses[name][col]+=1
    for name,metrics in {**held["metrics"],"deployed_trained_qwen":qwen["metrics"]}.items():
        fresh_counts=totals[name]
        required(metrics["cases"]==len(cases) and metrics["mask_hits_all"]==fresh_counts["hits"]
                 and metrics["proposals"]==fresh_counts["proposals"] and metrics["operational_failures"]==fresh_counts["errors"]
                 and abs(metrics["mean_coarse_iou_all_missing_zero"]-fresh_counts["iou"]/len(cases))<1e-10,
                 "Independent full-denominator aggregate differs for "+name)
    reload=read(fresh/"normal-reload-agreement.json")
    required(reload["all_choice_and_selective_decisions_reproduced"] and reload["cpu_prediction_sha256"]==held["predictions_sha256"],
             "Missing normal-entry-point replay for tested CPU predictions")
    dev=read(calibration/"cross-fitted-scores.json")
    compact={
        "kind":"matched_composite_qwen_and_experimental_dense_destination_geometry",
        "protocol_sha256":PROTOCOL_HASH,"requests_sha256":cpu["requests_sha256"],
        "source_reference_sha256":meta["reference_file_sha256"],"zero_detectors":True,
        "needs_confirmation_when_used_by_a_passenger":True,
        "design":"Two learned 500-command public spatial heads and a logistic selector fit on separate calibration groups. The matched Qwen comparison uses the actual deployed composite, including its ability to ask or abstain.",
        "reference_role":"Original participant destination masks on previously unused scenes from the public Talk2Car-RegSeg training source. No semantic or human-confirmation labels. Prior Qwen or SegFormer pretraining exposure is unknown.",
        "calibration":{"cases":125,"source_groups":65,"report":"Grouped five-fold out-of-sample calibration predictions on the previously used development set.",
                       "metrics":dev["metrics"],"selected_heads":dev["selected_head_counts"]},
        "independent":{"cases":len(cases),"groups":len(set(meta["source_scenes"])),"source_overlap_counts":meta["overlap_counts"],
                       "metrics":{**held["metrics"],"deployed_trained_qwen":qwen["metrics"]},"selected_heads":held["selected_head_counts"],
                       "paired_native_mask_hits_versus_deployed":crosses,"two_mask_scorers_agreed_every_dense_and_deployed_row_and_total":True,
                       "all_initial_outcomes_retained":True},
        "model_checks":{"weights":cpu["conditions"],"calibrator_sha256":cpu["calibrator_sha256"],
                        "spatial_weights_archive":{"name":"passenger-spatial-geometry-20261121.zip",
                            "sha256":"90d6ca62688b8b7081dab46b90e4322d735df68f11dbcedb2fc684baf861fcdb"},
                        "cold_cpu_prediction_sha256":held["predictions_sha256"],"fresh_qwen_outcomes_sha256":qwen["outcomes_sha256"],
                        "verified_cold_reload":reload["models"]},
        "scope":"The dense heads return experimental original-pixel geometry. They do not decide ambiguity, detect unsafe stopping or replace the chosen application release. Twelve scenes give a small out-of-sample diagnostic."
    }
    entries={"protocol/model-and-selection.json":PROTOCOL.read_bytes(),"protocol/bank.json":(fresh/"manifest.json").read_bytes(),
             "protocol/queue.json":(fresh/"deployed-comparison.json").read_bytes(),"inputs/requests.jsonl":(fresh/"inputs/requests.jsonl").read_bytes(),
             "source/mask-identifiers.json":encoded([{"id":c["case_id"],**{k:c["source"][k] for k in ("source_index","scene_token","image_sha256","mask_sha256")}} for c in cases]),
             "source/acquisition-program.json":encoded({"filename":"acquire.py","bytes_sha256":file_sha256(fresh/"acquire.py"),
                 "python":(fresh/"acquire.py").read_text(encoding="utf8")}),
             "cpu/predictions.json":cpu_file.read_bytes(),"cpu/road-support.json":(fresh/"road-support.json").read_bytes(),
             "cpu/normal-reload-agreement.json":(fresh/"normal-reload-agreement.json").read_bytes(),
             "calibration/model.json":(calibration/"calibrator.json").read_bytes(),
             "calibration/grouped-scores.json":(calibration/"cross-fitted-scores.json").read_bytes(),
             "scores/private.json":results.read_bytes(),"scores/aggregate.json":encoded(compact)}
    for name in ("settings.json","run.json","outcomes.jsonl"):
        entries["deployed/"+name]=(outdir/name).read_bytes()
    for name in cpu["conditions"]:
        folder=fresh/("normal-reload-"+name)
        for filename in ("run.json","outcomes.jsonl"):
            entries[f"cpu/normal/{name}/{filename}"]=(folder/filename).read_bytes()
    artifact_sha=json_archive(bundle,entries)
    compact["evidence_zip"]={"name":bundle.name,"sha256":artifact_sha,"bytes":bundle.stat().st_size}
    with out.open("xb") as stream:
        stream.write(encoded(compact))
    print(json.dumps({"aggregate_sha256":file_sha256(out),"evidence_zip":compact["evidence_zip"],"independent":compact["independent"]},indent=2))
    return compact


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("fresh","calibration","results","out","bundle"):
        parser.add_argument("--"+key,type=Path,required=True)
    a=parser.parse_args()
    publish(a.fresh,a.calibration,a.results,a.out,a.bundle)
