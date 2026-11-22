"""Paired diagnostic for the normal footage answerer with the temporal retrieval profile.

Prepare all public inputs offline first. The run command takes the same full questions and
original saved predictions, reuses a response only for the identical earlier model payload,
and calls the normal three-argument answerer for genuinely changed payloads. The previous
provider attempts and any explicit rate-limit retries remain separate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path
import re
import shutil
import time
import zipfile

from evaluation.run_audit import audit_run
from evaluation.score import read_jsonl
from tools.qa_request_fingerprints import preflight, verify_delivery
from tools.vision_saved_adapters import open_recording, predict_tracks_with_receipt
from trackrag.answering import answer_question, prepare_evidence, validate_answer
from trackrag.review import _answer, review_recording
from trackrag.vision.config import load_config


PROFILE = "trackrag.retrieval_temporal"
RECORDINGS = {"video_054604": "experiments/temporal-development-questions.jsonl",
              "video_054907": "experiments/temporal-second-questions.jsonl"}
SOURCES = ("trackrag/answering.py", "trackrag/model_client.py", "trackrag/privacy.py", "trackrag/questions.py",
           "trackrag/retrieval.py",
           "trackrag/review.py", "trackrag/retrieval_temporal.py", "trackrag/evidence_time.py",
           "tools/vision_saved_adapters.py","tools/qa_request_fingerprints.py")
MODEL = "gemini-3.5-flash-lite"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf8", newline="\n")
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def duplicate_file(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return str(target)


def previous(fingerprints, runs):
    if zipfile.is_zipfile(fingerprints):
        with zipfile.ZipFile(fingerprints) as bundle:
            stored = json.loads(bundle.read("input-fingerprints.json"))
    else:
        stored = read(fingerprints)
    index = {}
    for folder in runs:
        for row in read_jsonl(folder / "results.jsonl"):
            identity = row["question"]["question_id"]
            require(identity in stored and identity not in index, "Unknown or repeated previous question")
            verify_delivery(stored[identity], row, folder)
            index[identity] = (row, folder)
    require(set(index) == set(stored) and len(index) == 36, "Expected exactly the original 36 previous cases")
    return stored, index


def model_source():
    return {name: file_sha(name) for name in SOURCES}


def prepare(args):
    """Write real new ranked evidence, byte-exact provider-input digests and previews, with no network."""
    args.out.mkdir(parents=True, exist_ok=True)
    require(not (args.out / "selection.json").exists(), "Use a new preflight directory")
    stored, original = previous(args.fingerprints, args.previous_run)
    source_snapshot = model_source()
    lookup = import_module(PROFILE)
    rows, sources, previews = {}, {}, {}
    prior_images = {i["sha256"] for value in stored.values() for i in value["images"]}
    for recording, manifest in RECORDINGS.items():
        out = args.out / recording
        rec = open_recording(args.data_root, recording)
        questions = read_jsonl(manifest)
        tracks, source = predict_tracks_with_receipt(rec, directory=args.predicted,
                                                    config=load_config(args.vision_config))
        review_recording(rec, questions, source="predicted", load_tracks=lambda _: tracks,
                         build_index=lookup.build_index, rank_evidence=lookup.rank_evidence,
                         output_dir=out, run_metadata={"answerer": "none", "retrieval": PROFILE,
                         "retrieval_adapter": PROFILE, "adapter_module": "tools.vision_saved_adapters"})
        refs = {q["question_id"]:q for q in questions}
        audited = audit_run(out, refs)
        sources[recording] = {"tracks_sha256": file_sha(out / "tracks.jsonl"),
                              "public_questions_sha256": file_sha(manifest),
                              "run_sha256": file_sha(out / "run.json"), "results_sha256": file_sha(out / "results.jsonl"),
                              "original_config_identity": source["vision_identity_hash"],
                              "original_provenance_sha256": source["provenance_sha256"],
                              "complete_saved_track_file_sha256": source["tracks_sha256"]}
        for qid,row in audited["results"].items():
            old, old_dir = original[qid]
            require(row["question"] == old["question"] and
                    file_sha(old_dir / "tracks.jsonl") == sources[recording]["tracks_sha256"],
                    "Original question, cutoff or actual predicted tracks changed")
            frozen = preflight(row, out, refs[qid], MODEL)
            before, after = stored[qid]["request_arguments_sha256"], frozen["request_arguments_sha256"]
            same = before == after
            if same:
                require(frozen["evidence_intervals_sha256"] == stored[qid]["evidence_intervals_sha256"]
                        and [(i["evidence_id"],i["frame_id"],i["sha256"]) for i in frozen["images"]]
                            == [(i["evidence_id"],i["frame_id"],i["sha256"]) for i in stored[qid]["images"]],
                        "Identical model digest has different evidence")
            for item in frozen["images"]:
                if not same and item["sha256"] not in prior_images:
                    previews.setdefault(item["sha256"], {"question_id":qid, **item})
            rows[qid] = {"input":frozen,"previous_arguments_sha256":before, "previous_run":old_dir.name,
                         "dispatch":("unchanged_prior" if same else "changed_live" if after else "changed_empty")}
    require(len(rows)==36 and len({r["input"]["request_arguments_sha256"] for r in rows.values()
            if r["input"]["request_arguments_sha256"]}) == sum(bool(r["input"]["request_arguments_sha256"]) for r in rows.values()),
            "New model inputs are missing or duplicated")
    write(args.out/"new-images.json",list(previews.values()))
    require(model_source() == source_snapshot, "Input preparation overlapped a source-code change; preserve and rerun")
    write(args.out/"selection.json",{"prepared_utc":stamp(),"source_sha256":source_snapshot,"model":MODEL,
          "profile":PROFILE,"fingerprints_source_sha256":file_sha(args.fingerprints),"groups":sources,"rows":rows,
          "new_images_sha256":file_sha(args.out/"new-images.json"),"distinct_new_images":len(previews),
          "dispatch_counts":dict(Counter(r["dispatch"] for r in rows.values())),"provider_calls_in_preparation":0})
    print(json.dumps({"selection_sha256":file_sha(args.out/"selection.json"),"new_images":len(previews),
                      "dispatch":dict(Counter(r["dispatch"] for r in rows.values()))}))


def stage(prepared, destination, group):
    destination.mkdir(parents=True,exist_ok=True)
    if not (destination / "run.json").exists():
        if (prepared/"evidence").is_dir():
            shutil.copytree(prepared/"evidence",destination/"evidence",copy_function=duplicate_file,dirs_exist_ok=True)
        if not (destination/"tracks.jsonl").exists():
            duplicate_file(prepared/"tracks.jsonl",destination/"tracks.jsonl")
        meta=read(prepared/"run.json")
        meta["configuration"].update(answerer="gemini",answer_adapter="trackrag.answering",
            dispatch="paired offline-ranked evidence; exact prior payload reuse or ordinary three-argument answerer")
        write(destination/"run.json",meta)
    require(file_sha(destination/"tracks.jsonl")==group["tracks_sha256"],
            "Staged tracks differ from the reviewed runtime inputs")


def resume_attempt(event, archived, host, question, row, frozen, run_dir):
    """Recover the real response after a process interruption without sending it again."""
    require(event["input_sha256"] == frozen["request_arguments_sha256"], "Attempt belongs to another input")
    receipt = host / "receipt.json"
    if archived.exists():
        if event["state"] == "recorded":
            require(file_sha(archived) == event["receipt_sha256"], "Archived receipt changed")
        shutil.copyfile(archived, receipt)
    require(receipt.is_file(),
            f"{question['question_id']} has an interrupted dispatch with unknown service status; no repeat made")
    provider = read(receipt)
    if event["state"] == "recorded":
        answer = event["answer"]
    elif provider.get("response") and not provider.get("delivery_error"):
        prepared = prepare_evidence(row["evidence"], run_dir, question)
        answer = validate_answer(provider["response"], prepared, question)
    else:
        require(provider.get("delivery_error") and provider.get("error_type"), "Cannot reconstruct incomplete receipt")
        answer = {"answer_status": "error", "answer": f"{provider['error_type']}: {provider['delivery_error']}",
                  "cited_evidence_ids": []}
    row["answer"] = answer
    checked = verify_delivery(frozen, row, run_dir)
    if not archived.exists():
        shutil.copyfile(receipt, archived)
    return {**event, "state": "recorded", "receipt_sha256": file_sha(archived), "answer": answer, "delivery": checked}


def collect(args):
    plan=read(args.prepared/"selection.json")
    require(file_sha(args.prepared/"selection.json")==args.selection_sha256 and
            file_sha(args.prepared/"new-images.json")==plan["new_images_sha256"] and plan["source_sha256"]==model_source(),
            "The reviewed preflight or its model/retrieval code changed")
    require(MODEL==plan["model"] and file_sha(args.fingerprints)==plan["fingerprints_source_sha256"],"Baseline differed")
    _,original=previous(args.fingerprints,args.previous_run)
    os.environ["GEMINI_MODEL"]=MODEL
    destination=args.out;destination.mkdir(parents=True,exist_ok=True)
    ledger=destination/"attempts";ledger.mkdir(exist_ok=True)
    results=destination/"rows";results.mkdir(exist_ok=True)
    last_attempt=time.monotonic()-args.spacing
    # Check the two previously documented errors first, then publish in the untouched
    # original question order. These IDs control trial scheduling only.
    waves = [(recording, priority) for priority in (True, False) for recording in RECORDINGS]
    for recording, priority in waves:
        source=args.prepared/recording
        info=plan["groups"][recording]
        require(file_sha(source/"results.jsonl")==info["results_sha256"] and file_sha(source/"run.json")==info["run_sha256"],
                "Preflight retrieval output or metadata changed")
        out=destination/recording;stage(source,out,info)
        published=[]
        for row in read_jsonl(source/"results.jsonl"):
            q=row["question"];qid=q["question_id"]
            require(re.fullmatch(r"(?:temporal|second)-[0-9]{2}",qid),"Unexpected question ID")
            if priority and qid not in {"temporal-01", "second-06"}:
                continue
            spec=plan["rows"][qid];frozen=spec["input"]
            final=results/(qid+".json"); old,old_dir=original[qid]
            host=out/"hosted"/sha256(qid.encode()).hexdigest()
            if final.exists():
                delivered=read(final)
                require(delivered["input_sha256"]==frozen["request_arguments_sha256"],"Saved outcome uses a different input")
                row["answer"]=delivered["answer"]
                verify_delivery(frozen,row,out)
                published.append(row);continue
            mode=spec["dispatch"]
            attempts=[]
            if mode=="unchanged_prior":
                answer=old["answer"]
                if row["evidence"]:
                    shutil.copytree(old_dir/"hosted"/host.name,host,dirs_exist_ok=True,copy_function=shutil.copyfile)
            elif mode=="changed_empty":
                require(not row["evidence"],"Missing new provider input")
                answer=_answer(q,[],out,answer_question)
            else:
                # Recompute from the real staged image bytes immediately before transmitting.
                assert_projection=preflight(row,out,q,MODEL)
                require(assert_projection["request_arguments_sha256"]==frozen["request_arguments_sha256"],
                        "Actual model input changed after the review")
                for attempt in (1,2):
                    number=ledger/(qid+"-"+str(attempt)+".json")
                    durable=ledger/(qid+"-"+str(attempt)+"-receipt.json")
                    if attempt == 1 and (ledger/(qid+"-2.json")).exists():
                        # The shared hosted directory may already contain the second response.
                        # Leave it intact and recover the first attempt only from its archive.
                        first = read(number)
                        require(first["state"] == "recorded" and first["input_sha256"] == frozen["request_arguments_sha256"]
                                and first["receipt_sha256"] == file_sha(durable)
                                and first["delivery"].get("http_status") == 429, "Invalid earlier retry record")
                        attempts.append(first)
                        continue
                    if number.exists():
                        event = resume_attempt(read(number), durable, host, q, row, frozen, out)
                    else:
                        delay=max(0,args.spacing-(time.monotonic()-last_attempt))
                        if delay: time.sleep(delay)
                        if (host/"receipt.json").exists():
                            require(attempt == 2 and attempts[0]["receipt_sha256"] == file_sha(host/"receipt.json"),
                                    "Found an unaccounted earlier request")
                            (host/"receipt.json").rename(ledger/(qid+"-before-retry.json"))
                        event={"question":qid,"attempt":attempt,"state":"started","utc":stamp(),
                               "input_sha256":frozen["request_arguments_sha256"]}
                        write(number,event)
                        last_attempt=time.monotonic()
                        answer=_answer(q,row["evidence"],out,answer_question)
                        row["answer"]=answer
                        outcome=verify_delivery(frozen,row,out)
                        shutil.copyfile(host/"receipt.json",durable)
                        event.update(state="recorded",completed_utc=stamp(),receipt_sha256=file_sha(durable),
                                     answer=answer,delivery=outcome)
                    write(number,event);attempts.append(event)
                    answer=event["answer"]
                    outcome=event["delivery"]
                    if attempt==2 or outcome.get("http_status")!=429: break
                    wait=re.search(r"retry in (\d+)\s*s",answer.get("answer") or "",re.I)
                    time.sleep(max(60,min(120,int(wait.group(1))+2 if wait else 60)))
            row["answer"]=answer
            evidence=verify_delivery(frozen,row,out)
            write(final,{"question_id":qid,"input_sha256":frozen["request_arguments_sha256"],"dispatch":mode,
                         "delivery":evidence,"answer":answer,"attempts":attempts})
            published.append(row)
            print(qid,mode,answer["answer_status"],flush=True)
        if priority:
            continue
        complete=out/"results.jsonl"
        payload="".join(json.dumps(r,allow_nan=False)+"\n" for r in published).encode()
        if complete.exists(): require(complete.read_bytes()==payload,"A previously completed recording differs")
        else: complete.write_bytes(payload)
        audit_run(out,{r["question"]["question_id"]:r["question"] for r in published})
    write(destination/"complete.json",{"completed_utc":stamp(),"selection_sha256":args.selection_sha256,
          "rows":len(list(results.glob("*.json"))),"provider_attempts":len(list(ledger.glob("*-receipt.json"))),
          "profile":PROFILE,"model":MODEL})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    commands=ap.add_subparsers(dest="mode",required=True)
    p=commands.add_parser("prepare");r=commands.add_parser("run")
    for sub in (p,r):
        sub.add_argument("--fingerprints",type=Path,required=True,help="Prior byte-fingerprints JSON or verified baseline ZIP")
        sub.add_argument("--previous-run",action="append",type=Path,required=True)
        sub.add_argument("--out",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--predicted",type=Path,required=True)
    p.add_argument("--vision-config",type=Path,required=True)
    r.add_argument("--prepared",type=Path,required=True)
    r.add_argument("--selection-sha256",required=True)
    r.add_argument("--spacing",type=float,default=18,help="Minimum seconds between fresh provider requests")
    args=ap.parse_args()
    if args.mode=="run":
        require(args.spacing>=0,"Spacing must be nonnegative")
        collect(args)
    else: prepare(args)


if __name__=="__main__": main()
