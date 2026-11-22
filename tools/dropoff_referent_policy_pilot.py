"""Run the pre-frozen ask-only policy on exactly the 19 development inputs without any target data."""
import argparse
from collections import Counter
from hashlib import sha256
import io
import importlib.util
import json
from pathlib import Path

from PIL import Image

from trackrag.dropoff import referent_policy
from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.passenger import instruction_context
from trackrag.dropoff.referent import ReferentRanker


DEV_REQUESTS = "0160bf67789bc6d7f3a362f2a6d607b1028b1b81ca0490863a5971ede224846f"
CHECKPOINT = "d96a9a69e2e52ddbc21851d5ff6f79b344de6c8558600e54be5cb2712480a07d"
FROZEN_POLICY = "180eaa1eb8acc8a442c974c0942452dfa6dee8cf445b31a657c664afcce134f0"
REVISED_POLICY = "4c5cd8be4f8e3b22e11deaf98bf40a38a6544745f11bb27e33a85eaeff3340d1"


def run(requests, model_path, output, original_policy_source=None):
    if output.exists():
        raise ValueError("Use a fresh output file")
    selected = referent_policy
    source = original_policy_source or Path(referent_policy.__file__)
    src = source.read_bytes().replace(b"\r\n",b"\n")
    expected = FROZEN_POLICY if original_policy_source else REVISED_POLICY
    if sha256(src).hexdigest() != expected:
        raise ValueError("Policy differs from the selected deployment pilot")
    if original_policy_source:
        spec = importlib.util.spec_from_file_location("trackrag.dropoff._frozen_referent_policy",source)
        selected = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selected)
    payload = requests.read_bytes()
    if sha256(payload).hexdigest() != DEV_REQUESTS:
        raise ValueError("Only the frozen 19 development inputs are allowed")
    cases = [validate_request(json.loads(line)) for line in payload.splitlines()]
    if len(cases)!=19 or any(r["scene"]["frame"]["recording_id"] != "video_054907" for r in cases):
        raise ValueError("Unexpected development cohort")
    ranker=ReferentRanker.load(model_path,expected_sha256=CHECKPOINT)
    rows=[]
    for r in cases:
        frame=r["scene"]["frame"]
        image_path=Path(frame["image_path"])
        if not image_path.is_absolute():
            image_path=requests.parent/image_path
        raw_image=image_path.read_bytes()
        if sha256(raw_image).hexdigest()!=r["scene"]["provenance"]["image_sha256"]:
            raise ValueError("Changed development scene image")
        with Image.open(io.BytesIO(raw_image)) as image:
            text=instruction_context(r)
            candidates=ranker.rank(image,text,r["scene"]["tracks"],frame=frame)
            advice=selected.clarify_reference(text,candidates,image.size)
        rows.append({"request_id":r["request_id"],"input_sha256":sha256(json.dumps(r,sort_keys=True).encode()).hexdigest(),
                     "current_text":r["text"],"context":text,"frame":frame["frame_id"],
                     "image_sha256":r["scene"]["provenance"]["image_sha256"],"current_candidates":candidates["candidate_count"],
                     "distinct":candidates["distinct_candidate_count"],"decision":advice,
                     "top_actual_candidates":[{k:row[k] for k in ("track_id","class_name","bbox_xyxy","score")}
                                              for row in candidates["alternatives"][:3]]})
    counts={"rows":len(rows),"ask":sum(r["decision"] is not None for r in rows),
            "defer":sum(r["decision"] is None for r in rows),"reasons":dict(Counter(
                    r["decision"]["reason"] for r in rows if r["decision"]))}
    receipt={"policy_source_sha256":expected,"development_input_sha256":DEV_REQUESTS,"weights_sha256":CHECKPOINT,
             "policy_version":selected.VERSION,"development_informed_revision":not bool(original_policy_source),
             "inputs_only":True,"evaluation_labels_read":False,"generates_destination_or_confirms":False,"counts":counts,"rows":rows}
    with output.open("x",encoding="utf8",newline="\n") as f:
        json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False)
    return counts


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("requests","model","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--original-policy-source",type=Path,help="Replay the original SHA-pinned pilot recipe")
    cmd=parser.parse_args()
    print(json.dumps(run(cmd.requests,cmd.model,cmd.out,cmd.original_policy_source)))
