"""Extract only detector proposals for the frozen public training/development cohorts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

from tools.dropoff_anchor_fit import DEV_SHA, TRAIN_SHA
from tools.dropoff_bootstrap_regseg import METADATA_SHA256, METADATA_SIZE, METADATA_URL


CANDIDATES_SHA256 = "0e0a00f2da41b4e9f148b084e55bb1a54d5dabfa9c3ad22d13ed666747ebeedc"


def canonical(data):
    return json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def read_candidates(path):
    blob = Path(path).read_bytes()
    if len(blob) > 16*1024**2:
        raise ValueError("Unexpected public detector-bank size")
    value = json.loads(blob)
    if sha256(canonical(value)).hexdigest() != CANDIDATES_SHA256:
        raise ValueError("Detector bank differs from the pinned 625 public cases")
    return value


def export(splits, source, out):
    ids = set()
    for group, expected in (("train",TRAIN_SHA),("development",DEV_SHA)):
        raw = (splits/f"{group}.jsonl").read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError("Unexpected public split")
        ids.update(json.loads(line)["source"]["source_index"] for line in raw.splitlines())
    if source:
        source_bytes = source.read_bytes()
    else:
        with urllib.request.urlopen(METADATA_URL,timeout=60) as response:
            source_bytes = response.read(METADATA_SIZE+1)
    if len(source_bytes) != METADATA_SIZE or sha256(source_bytes).hexdigest() != METADATA_SHA256:
        raise ValueError("Official public metadata failed verification")
    records = json.loads(source_bytes)["train"]
    subset = {str(i):{"predictions":records[str(i)]["centernet"],
                      "command_token":records[str(i)]["command_token"]} for i in ids}
    data = canonical(subset)
    if sha256(data).hexdigest() != CANDIDATES_SHA256:
        raise ValueError("Exported detector bank differs from frozen inputs")
    with out.open("xb") as target:
        target.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits",type=Path,required=True)
    parser.add_argument("--source",type=Path,help="Optional local copy of the original pinned metadata")
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    export(args.splits,args.source,args.out)
