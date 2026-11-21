"""Run or explicitly replay the passenger model without reading evaluation references."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

from tools.dropoff_predict import read_requests
from trackrag.dropoff.contracts import error_outcome
from trackrag.dropoff.landmarks import resolve_image
from trackrag.dropoff.passenger import load_passenger_model, operational_code


def run(args):
    input_bytes = args.requests.read_bytes()
    inputs = read_requests(args.requests, payload=input_bytes)
    saved, saved_digest = None, None
    if args.replay is not None:
        data = args.replay.read_bytes()
        saved_digest = sha256(data).hexdigest()
        saved = {}
        for line in data.decode("utf8").splitlines():
            value = json.loads(line)
            key = value["request_id"]
            if key in saved:
                raise ValueError("Duplicate recorded resolver decision")
            saved[key] = value
        if set(saved) != {r["request_id"] for r in inputs}:
            raise ValueError("Saved landmark batch differs from selected request cohort")
    if args.verify_landmarks and saved is not None:
        raise ValueError("Visual-verification comparisons require a live resolver")
    model = load_passenger_model(args.config,geometry_path=args.geometry,vision_cache=args.vision_cache,
                                 verify_landmarks=args.verify_landmarks)
    args.out.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    names = ["trackrag/dropoff/"+n+".py" for n in ("passenger","landmarks","landmark_check","anchor_geometry","streamed","interaction")]
    summary = {"requested":len(inputs),"requests_sha256":sha256(input_bytes).hexdigest(),
               "source":{n:sha256((source_root/n).read_bytes()).hexdigest() for n in names},
               "geometry_sha256":model.geometry.sha256,"config":asdict(model.config),
               "resolver_execution":"saved_decisions" if saved is not None else "live",
               "saved_decisions_sha256":saved_digest,"reference_access":False,
               "verify_landmarks":args.verify_landmarks}
    (args.out/"settings.json").write_text(json.dumps(summary,indent=2), encoding="utf8")
    errors = 0
    try:
        with (args.out/"outcomes.jsonl").open("x",encoding="utf8") as output:
            for request in inputs:
                start = time.monotonic()
                key = request["request_id"]
                try:
                    if saved is None:
                        outcome = model.interpret(request)
                    elif "error" in saved[key]:
                        error = saved[key]["error"]
                        outcome = error_outcome("resolver_failed",str(error),key,{"saved_error":error})
                    else:
                        image, binding = resolve_image(request,model.config)
                        with image:
                            outcome = model.from_decision(request,image,binding,saved[key])
                except Exception as exc:
                    outcome = error_outcome(operational_code(exc),str(exc) or type(exc).__name__,key)
                outcome["provenance"]["resolver_execution"] = summary["resolver_execution"]
                outcome["provenance"]["batch_elapsed_s"] = time.monotonic()-start
                output.write(json.dumps(outcome,sort_keys=True,allow_nan=False)+"\n")
                output.flush()
                print(key,outcome.get("result") or outcome.get("error"),flush=True)
                errors += not outcome["ok"]
    finally:
        model.close()
    summary.update(errors=errors,outcomes_sha256=sha256((args.out/"outcomes.jsonl").read_bytes()).hexdigest())
    (args.out/"run.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    return int(bool(errors))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("requests","config","geometry","vision-cache","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--replay",type=Path)
    p.add_argument("--verify-landmarks",action="store_true")
    raise SystemExit(run(p.parse_args()))
