"""Build a standalone, blank human-review form for the diagnostic train recording.

Only the request specification, frame image and portable input binding enter the form.
Keep each completed JSON export as an independent review pending comparison/adjudication.
"""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path

from tools.dropoff_predict import read_requests
from trackrag.dropoff.data import inference_input_sha256
from trackrag.emt import open_recording


TEMPLATE = Path(__file__).with_name("dropoff_passenger_references_pack.html")
TRAIN_RECORDING = "video_054604"


def build_pack(bank, requests, out, *, image_root=None):
    freeze = json.loads((bank / "freeze.json").read_bytes())
    specs = (bank / "train-instructions.jsonl").read_bytes().replace(b"\r\n", b"\n")
    bindings = (bank / "input-bindings.jsonl").read_bytes().replace(b"\r\n", b"\n")
    receipt = json.loads((bank / "input-bindings-receipt.json").read_bytes())
    if (sha256(specs).hexdigest() != freeze["files"]["train-instructions.jsonl"]["sha256"]
            or sha256(bindings).hexdigest() != receipt["bindings_file_sha256"]
            or sha256((bank / "freeze.json").read_bytes()).hexdigest() != receipt["freeze_file_sha256"]):
        raise ValueError("instruction or verified input binding changed after freeze")
    sources = (bank / "frames.jsonl").read_bytes().replace(b"\r\n", b"\n")
    if sha256(sources).hexdigest() != freeze["files"]["frames.jsonl"]["sha256"]:
        raise ValueError("original frame registry differs from the frozen inputs")
    expected = {s["request_id"]: s for s in (json.loads(line) for line in specs.splitlines())}
    keys = {s["request_id"]: s for s in (json.loads(line) for line in bindings.splitlines())}
    frames = {(s["recording_id"], s["frame_id"]): s for s in (json.loads(line) for line in sources.splitlines())}
    rows, images, image_digests = [], {}, {}
    if (requests is None) == (image_root is None):
        raise ValueError("provide the original EMT image folder or the verified request file")
    bank_requests = {r["request_id"]: r for r in read_requests(requests)} if requests else None
    if bank_requests is not None and bank_requests.keys() != expected.keys():
        raise ValueError("the blind pack accepts the complete train recording only")
    recording = open_recording(image_root, TRAIN_RECORDING, split="train") if image_root else None
    for identity, spec in expected.items():
        source = frames[spec["recording_id"], spec["frame_id"]]
        if source["recording_id"] != TRAIN_RECORDING:
            raise ValueError("this review pack is restricted to the train recording")
        request = bank_requests[identity] if bank_requests else None
        frame = request["scene"]["frame"] if request else recording.get_frame(spec["frame_id"])
        if ((frame["recording_id"], frame["frame_id"]) != (spec["recording_id"], spec["frame_id"])
                or (frame["width"], frame["height"]) != (source["width"], source["height"])):
            raise ValueError("review frame identifier or dimensions differ from the frozen original")
        path = Path(frame["image_path"])
        if path not in image_digests:
            raw = path.read_bytes()
            digest = sha256(raw).hexdigest()
            if not raw.startswith(b"\xff\xd8\xff") or len(raw) > 16 * 1024**2:
                raise ValueError("expected a bounded original JPEG")
            image_digests[path] = digest
            images[digest] = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        digest = image_digests[path]
        input_hash = keys[identity]["inference_input_sha256"]
        if (digest != keys[identity]["image_sha256"] or digest != source["image_sha256"]
                or (request and inference_input_sha256(request, digest) != input_hash)):
            raise ValueError("review inputs differ from the verified frame and instruction")
        rows.append(dict(request_id=identity, recording_id=TRAIN_RECORDING, frame_id=frame["frame_id"],
                         width=frame["width"], height=frame["height"], image_sha256=digest,
                         input_sha256=input_hash, text=spec["text"], dialogue=spec.get("dialogue", [])))
    metadata = {"version": 1, "kind": "blank_human_review", "partition": "train", "rows": rows,
                "images": images, "input_bank_sha256": sha256(specs).hexdigest()}
    # Script text cannot contain a literal closing tag from a user instruction.
    raw = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    safe = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    if out.exists():
        raise FileExistsError("choose a new output HTML file")
    with out.open("x", encoding="utf-8", newline="\n") as file:
        file.write(TEMPLATE.read_text(encoding="utf-8").replace("{{BLANK_REVIEW_DATA}}", safe))
    return {"requests": len(rows), "original_frames": len(images),
            "model_outputs_included": False, "reference_answers_included": False,
            "sha256": sha256(out.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("annotations/dropoff/passenger-development"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--requests", type=Path, help="Existing prepared train requests with local original frames")
    source.add_argument("--image-root", type=Path, help="Original EMT image folder containing images/train/")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_pack(args.bank, args.requests, args.out, image_root=args.image_root), indent=2))


if __name__ == "__main__":
    main()
