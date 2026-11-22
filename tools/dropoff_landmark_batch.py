"""Save raw decisions of the landmark resolver, with no destination-reference inputs."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time

from tools.dropoff_predict import read_requests
from trackrag.dropoff.landmarks import GUIDE, LandmarkResolver, resolve_image
from trackrag.dropoff.model import GroundingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pretrained", action="store_true", help="Explicitly disable the adapter for comparison")
    args = parser.parse_args()
    config = GroundingConfig.from_mapping(json.loads(args.config.read_text(encoding="utf8")))
    if args.pretrained:
        config = replace(config, adapter_dir=None)
    payload = args.requests.read_bytes()
    rows = read_requests(args.requests, payload=payload)
    resolver = LandmarkResolver(config)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"settings.json").write_text(json.dumps({"request_sha256":sha256(payload).hexdigest(),
        "configuration":asdict(config),"requested":len(rows),"prompt":GUIDE,
        "source_sha256":sha256(Path(__file__).with_name("dropoff_landmark_batch.py").read_bytes()).hexdigest(),
        "resolver_source_sha256":sha256(Path(__file__).resolve().parents[1].joinpath(
            "trackrag/dropoff/landmarks.py").read_bytes()).hexdigest()},indent=2),encoding="utf8")
    try:
        with (args.out/"decisions.jsonl").open("x", encoding="utf8") as output:
            for request in rows:
                t = time.monotonic()
                record = {"request_id":request["request_id"],"request_sha256":sha256(json.dumps(
                    request, sort_keys=True, separators=(",",":")).encode()).hexdigest()}
                try:
                    image, record["image"] = resolve_image(request, config)
                    with image:
                        record.update(resolver.resolve(request,image))
                except Exception as exc:
                    record["error"] = {"type":type(exc).__name__, "message":str(exc)}
                record["elapsed_s"] = time.monotonic()-t
                output.write(json.dumps(record, allow_nan=False)+"\n")
                output.flush()
                print(request["request_id"],record.get("raw_text") or record.get("error"),flush=True)
    finally:
        resolver.close()


if __name__ == "__main__":
    main()
