"""Build a local-only map archive from original photos with verified Mask2Former weights."""
import argparse
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from tools.passenger_parking_study import ParkingSegmenter, SEG_REVISION, SEG_WEIGHTS
from trackrag.dropoff.manual_surfaces import RECIPE
from trackrag.dropoff.parking_evidence import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, map_shape, signed_bytes


def prepare(sources, checkpoint, out, device):
    data = json.loads(Path(sources).read_bytes())["sources"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = ParkingSegmenter(checkpoint, device=device)
    rows = {}
    try:
        with TemporaryDirectory(dir=out) as temp:
            archive = Path(temp) / "passenger-local-image-surfaces-v1.zip"
            with ZipFile(archive, "w", ZIP_DEFLATED) as bank:
                for record in sorted(data, key=lambda row: row["sha256"]):
                    started = monotonic()
                    path = Path(sources).resolve().parent / record["original_path"]
                    if not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
                        raise ValueError("Invalid original photo size.")
                    original = path.read_bytes()
                    key = sha256(original).hexdigest()
                    if key != record["sha256"] or key in rows:
                        raise ValueError("Photo identity changed or is duplicated.")
                    width, height = record["size"]
                    frame = {"width":width, "height":height}
                    with Image.open(BytesIO(original)) as image:
                        if image.size != (width,height) or width*height > MAX_IMAGE_PIXELS:
                            raise ValueError("Photo dimensions differ from the source record.")
                        classes, _ = model.predict(image.convert("RGB"), map_shape(frame))
                    stream = BytesIO()
                    Image.fromarray(classes).save(stream, format="PNG")
                    png = stream.getvalue()
                    body = {"recipe":RECIPE, "model_revision":SEG_REVISION, "weights_sha256":SEG_WEIGHTS,
                            "image_sha256":key, "size":[width,height], "recording_id":record["recording_id"],
                            "frame_id":record["frame_id"], "class_map":{"sha256":sha256(png).hexdigest(),
                            "png":base64.b64encode(png).decode("ascii")}}
                    pin = sha256(signed_bytes(body)).hexdigest()
                    bank.writestr(key + ".json", signed_bytes({"body":body,"origin":"local_release_v1","signature":pin}))
                    rows[key] = {"body_sha256":pin, "recording_id":record["recording_id"], "frame_id":record["frame_id"],
                                 "size":[width,height], "source_url":record["source_url"]}
                    print(f"{record['recording_id']} frame {record['frame_id']}: {monotonic()-started:.2f}s", flush=True)
            index = archive.with_suffix(".json")
            index.write_bytes(signed_bytes({"version":1,"weights_sha256":SEG_WEIGHTS,"sources":rows})+b"\n")
            os.replace(archive, out/archive.name)
            os.replace(index, out/index.name)
            print(f"Surface index SHA-256 {sha256((out/index.name).read_bytes()).hexdigest()}",flush=True)
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu","cuda:0"))
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    def offline(event, args):
        if event == "socket.connect":
            raise RuntimeError("Local surface preparation cannot connect to a network.")
    sys.addaudithook(offline)
    import torch
    torch.set_num_threads(3)
    prepare(args.sources,args.checkpoint,args.out,args.device)


if __name__ == "__main__":
    main()
