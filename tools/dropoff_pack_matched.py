"""Package the sealed, matched passenger evaluation as source JSON without images or reviewer records."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from tools.dropoff_audit_passenger_release import LABELS, ROOT, json_file, must, require_completed
from tools.dropoff_passenger_references import score as score_night
from tools.dropoff_passenger_references_public import score as score_public


def encoded(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf8")


def json_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def json_archive(path, entries):
    """Write a deterministic, allow-listed JSON ZIP and reread every member before publishing it."""
    if path.exists():
        raise FileExistsError(path)
    inventory={}
    for name,content in entries.items():
        if (not name.endswith((".json",".jsonl")) or ".." in Path(name).parts or name.startswith(("/","\\"))):
            raise ValueError("Only relative JSON members may be archived")
        if name.endswith(".jsonl"):
            for line in content.splitlines():
                json.loads(line)
        else:
            json.loads(content)
        if re.search(rb"(?i)(?:[a-z]:[/\\]+users[/\\]|/home/[^/\s]+/|data:image/|;base64,|-----BEGIN [A-Z ]*PRIVATE KEY)",content):
            raise ValueError("User home, embedded media or private key in "+name)
        inventory[name]={"sha256":sha256(content).hexdigest(),"bytes":len(content)}
    complete={**entries,"inventory.json":encoded(inventory)}
    with path.open("xb") as stream:
        with ZipFile(stream,"w",compression=ZIP_DEFLATED,compresslevel=9) as z:
            for name in sorted(complete):
                info=ZipInfo(name,(1980,1,1,0,0,0))
                info.external_attr=0o100644<<16
                z.writestr(info,complete[name],compress_type=ZIP_DEFLATED,compresslevel=9)
    with ZipFile(path) as z:
        must(z.testzip() is None and set(z.namelist())==set(complete),"Invalid ZIP member set or CRC")
        for name,content in complete.items():
            must(z.read(name)==content,"Archived bytes differ from verified source")
    return sha256(path.read_bytes()).hexdigest()


def pack(path, root=ROOT):
    ext=root/"runs/passenger-landmarks/external"
    refs=root/"runs/passenger-references"
    public=refs/"public-independent"
    inputs={"night":refs/"night-inputs/requests.jsonl","public":public/"inputs/requests.jsonl"}
    protocol=root/"experiments/dropoff/passenger-release-protocol-20261122.json"
    runtime=root/"experiments/dropoff/passenger-runtime-freeze-20261122.json"
    public_audit=root/"experiments/dropoff/passenger-external-independent-audit-20261122.json"
    detail=ext/"independent-audit.json"
    require_completed(ext)
    audit=json_file(detail)
    digest=lambda p: sha256(p.read_bytes()).hexdigest()
    must(audit["all_four_outcome_files_sealed_and_verified_before_reference_read"]
         and json_file(public_audit)["audit_sha256"]==digest(detail)
         and audit["protocol_sha256"]==digest(protocol) and audit["runtime_freeze_sha256"]==digest(runtime),
         "Saved audit, publication or release freeze differs")
    entries={"protocol/release.json":protocol.read_bytes(),"protocol/runtime.json":runtime.read_bytes(),
             "protocol/inference-only-execution.json":(ext/"execution.json").read_bytes(),
             "scores/independent.json":detail.read_bytes(),"scores/independent-summary.json":public_audit.read_bytes()}
    for name,source in inputs.items():
        entries[f"inputs/{name}.jsonl"]=source.read_bytes()
    night_refs=refs/"blind-third/references.jsonl"
    pub_refs=public/"corpus/references.jsonl"
    identifiers={
        "night":{"source_sha256":digest(night_refs),"role":"AI reference; zero human reviews","cases":[
            {k:r[k] for k in ("request_id","recording_id","frame_id","image_sha256")} for r in json_lines(night_refs)]},
        "public":{"source_sha256":digest(pub_refs),"role":"Original participant region masks; no semantic labels","cases":[
            {"request_id":r["case_id"],**{k:r["source"][k] for k in ("scene_token","source_index","image_sha256","mask_sha256")}}
            for r in json_lines(pub_refs)]},
    }
    entries["references/source-identifiers.json"]=encoded(identifiers)
    for label in LABELS:
        _,cohort=label.split("-")
        folder=ext/label
        for name in ("run.json","settings.json","outcomes.jsonl"):
            entries[f"conditions/{label}/{name}"]=(folder/name).read_bytes()
        outcome=folder/"outcomes.jsonl"
        receipt=audit["conditions"][label]
        must(receipt["outcomes_sha256"]==digest(outcome) and receipt["requests_sha256"]==digest(inputs[cohort])
             and receipt["primary_scorer_cross_check"]["all_per_case_geometry_and_predeclared_totals_agree"],
             "Mismatched sealed input, outcomes or dual score")
        official=(score_night(refs/"blind-third","night",inputs["night"],outcome)
                  if cohort=="night" else score_public(public,outcome))
        entries[f"scores/{label}-primary.json"]=encoded(official)
    entries["purpose.json"]=encoded({
        "role":"Original matched composite passenger experiment for Task 1; includes all 72 attempts.",
        "model_roles":"Native local Qwen image-language selectors, publicly fine-tuned and pretrained, in otherwise identical composite conditions.",
        "reference_roles":"Night status/geometry is AI-authored and independently AI-reviewed. Public geometry derives from original participant masks.",
        "verification":"Hash each member against inventory.json. Original outcomes and model requests are byte-identical to their sealed manifests. Relative image paths refer to the original datasets. Reference identifiers and hashes support separate authorized source retrieval.",
        "intervention":"No passenger clicks or software fixtures in these runs. No current-frame detectors in the public image-only condition. The observed failure is retained."
    })
    return json_archive(path,entries)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({"zip":str(args.out),"sha256":pack(args.out),"bytes":args.out.stat().st_size},indent=2))
