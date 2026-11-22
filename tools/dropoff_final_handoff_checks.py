"""Audit only model-visible fields, missing assets and recording overlap for the corrected final cohort."""
import argparse
from collections import Counter
import json
from pathlib import Path

from evaluation.grounding import leakage
from evaluation.grounding_inputs import scene_problems
from tools.dropoff_final_input_inventory import FINAL,PACKAGE,ROOT,TASK1,TASK2,git,project,require,write
from tools.dropoff_pack_matched import encoded,json_archive
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.scenes import SceneError,load_scene,scene_sha256


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def normalized(text):
    return " ".join(text.casefold().split())


def input_overlaps(final, other):
    def sets(bank):
        result={k:set() for k in ("request_ids","recording_ids","normalized_instructions","images_sha256")}
        for entry in bank:
            r=entry.get("request",entry)
            digest=r["scene"].get("provenance",{}).get("image_sha256") or entry.get("source",{}).get("image_sha256")
            require(isinstance(digest,str) and len(digest)==64,"Missing source-image hash in a comparison corpus")
            result["request_ids"].add(r["request_id"]);result["recording_ids"].add(r["scene"]["frame"]["recording_id"])
            result["normalized_instructions"].add(normalized(r["text"]));result["images_sha256"].add(digest)
        return result
    f,o=sets(final),sets(other)
    return {"other_rows":len(other),"shared_unique":{k:len(v&o[k]) for k,v in f.items()},
            "final_rows_with_exact_other_wording":sum(normalized(r["text"]) in o["normalized_instructions"] for r in final)}


def audit(args):
    bank=args.bank
    receipt=json.loads((bank/"receipt.json").read_bytes())
    require(all(file_sha256(bank/k)==v for k,v in receipt["files"].items()),"Input-only handoff changed")
    inputs=rows(bank/"requests-with-saved-tracks.jsonl")
    reviewed=rows(bank/"requests-as-human-reviewed.jsonl")
    bindings=rows(bank/"input-bindings.jsonl");ingredients=rows(bank/"task1-scene-ingredients.jsonl")
    frame_manifest=json.loads(git(TASK1,"experiments/vision/emt_frame_sha256_manifest.json"))
    shared=json.loads(git(TASK1,"experiments/vision/shared_prediction_runs.json"))
    projection={"partition":True,"case_id":True,"scene_group":True,"source":{"mask_path":"presence","mask_sha256":"presence"}}
    index={}
    for entry in git(TASK2,PACKAGE+"/review_cases.jsonl").splitlines():
        record,_=project(entry.decode("utf8"),projection)
        if record["partition"]=="test":
            index[record["case_id"]]=record
    good=Counter();failed=Counter();refs={}
    for new,old,b,s in zip(inputs,reviewed,bindings,ingredients,strict=True):
        key=new["request_id"];f=new["scene"]["frame"]
        require(key==b["case_id"]==s["case_id"]==old["request_id"] and key in index and
                s["expected_task1_scene_sha256"]==scene_sha256(s),"Inconsistent input-only case identity")
        require(new["text"]==old["text"] and new["dialogue"]==old["dialogue"],"Passenger words or dialogue changed")
        require(not scene_problems(s,f,new["scene"]["tracks"],b["image_sha256"]),"Saved tracks do not match the derived Task 1 case")
        alternatives=scene_problems(s,old["scene"]["frame"],old["scene"]["tracks"],b["image_sha256"])
        good["different_reviewed_detector_context"]+=bool(alternatives)
        good["source_mask_nonnull"]+=bool(index[key]["source"].get("mask_path"))
        good["source_mask_hash_nonnull"]+=bool(index[key]["source"].get("mask_sha256"))
        good["reviewed_zero_observations"]+=not old["scene"]["tracks"]
        good["saved_detector_observations"]+=len(new["scene"]["tracks"])
        refs[key]={"recording_id":f["recording_id"],"group_id":index[key]["scene_group"],
                   "scene_sha256":s["expected_task1_scene_sha256"]}
        try:
            native=load_scene(bank/"source",f["recording_id"],f["frame_id"],profile=ROOT/"configs/vision/finetuned.json",
                              predictions_dir=args.predictions/"predictions-finetuned",frame_manifest=frame_manifest,run_manifest=shared)
            require(native["scene_sha256"]==s["expected_task1_scene_sha256"],"Canonical complete-prefix control disagrees with derivation")
            good["canonical_complete_prefix_controls_sha_equal"]+=1
        except SceneError as error:
            require("image frames up to" in str(error) and "differ from the predicted frames" in str(error),
                    "Unexpected canonical scene failure: "+str(error))
            failed[f["recording_id"]]+=1
    require(len(refs)==44 and all(r["recording_id"] in FINAL for r in refs.values()),"Different 44-case audit")
    dev=ROOT/"runs/passenger-references"
    masks=ROOT/"data/dropoff-public-expanded-20261121/splits"
    human=ROOT/"runs/passenger-task2-reviewed-20261122/data"
    prior={"public_spatial_or_qwen_training":masks/"train.jsonl","public_spatial_or_qwen_selection":masks/"development.jsonl",
           "selection_daytime":dev/"development-prepared/requests.jsonl","ai_continuation_training":dev/"train-prepared/requests.jsonl",
           "night_nonfinal_comparison":dev/"night-inputs/requests.jsonl","public16_nonfinal_comparison":dev/"public-independent/inputs/requests.jsonl",
           "new_human_training":human/"train.jsonl","new_human_development":human/"development.jsonl"}
    relationships={}
    for name,p in prior.items():
        original=rows(p)
        comparisons=input_overlaps(inputs,original)
        native=leakage(refs,[p])
        relationships[name]={"sha256":file_sha256(p),**comparisons,"official_identifier_leakage":native["violations"]}
    counts=receipt["per_recording"]
    summary={"kind":"heldout_input_only_readiness_audit","task1_revision":TASK1,"task2_revision":TASK2,
        "confirmed":len(refs),"heldout_recordings":len({r["recording_id"] for r in refs.values()}),
        "heldout_group_ids":len({r["group_id"] for r in refs.values()}),
        "distinct_utterances":len({normalized(r["text"]) for r in inputs}),"input_checks":dict(good),
        "canonical_export_failed_for_sparse_prefix":dict(failed),"physical_prefix":{
            "full_source_frame_ids_needed":sum(r["upstream_required_prefix_count"] for r in counts.values()),
            "current_originals_in_task2_package_including_pending":sum(r["in_corrected_package"] for r in counts.values()),
            "missing_originals_relative_to_corrected_task2_package":sum(r["missing_from_corrected_package"] for r in counts.values())},
        "prior_input_comparisons":relationships,"files":receipt["files"],
        "no_model_run":True,"final_targets_decoded":False,"formal_score":False,
        "release_gate":["Reviewed destination masks bound to source PNGs for all clear cases","Intended masks or missing classification for clarification cases, plus prespecified reply rules",
                        "Agreement on actual detector context: the corrected rows were human-reviewed with zero observed tracks",
                        "Task 1 protocol frozen before collecting any final model output, then four matched typed conditions"],
        "case_history":"The five EMT recordings previously supported Task 1 vision experiments. A historical metadata audit named a draft passenger handoff from video_122233; this is not evidence it trained the passenger model. The separate 20-case night bank used video_174340, the official development split. The formal leakage check includes temporal neighbor video_204347 for the official final recording video_204647."}
    probe=bank/"official-source-prefix-probe.json"
    if probe.is_file():
        v=json.loads(probe.read_bytes())
        summary["original_public_archive_probe"]={"compressed_bytes":v["compressed_bytes"],"verified_originals":len(v["verified_jpeg_members"]),
            "receipt_sha256":file_sha256(probe),"source_member_group":list({m["recording_id"] for m in v["verified_jpeg_members"]})}
    write(args.out,summary)
    if args.bundle:
        entries={k:(bank/k).read_bytes() for k in (*receipt["files"],"receipt.json")}
        entries["coverage.json"]=encoded(summary)
        if probe.is_file():
            entries["source-range-proof.json"]=probe.read_bytes()
        digest=json_archive(args.bundle,entries)
        print("JSON-only ZIP",digest,args.bundle.stat().st_size)
    print(json.dumps({k:summary[k] for k in ("input_checks","canonical_export_failed_for_sparse_prefix","physical_prefix")},indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--predictions",type=Path,default=ROOT/"data/task1-shared-20261119")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--bundle",type=Path)
    audit(parser.parse_args())
