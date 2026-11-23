"""Authenticated bridge to the workspace's private, scale-to-zero place model."""
import base64
import json
import sys

APP = "trackrag-passenger-places"
PROCESSOR_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def main():
    import modal
    raw = sys.stdin.buffer.read(58*1024*1024+1)
    if len(raw) > 58*1024*1024:
        raise ValueError("Place request exceeds transport limit")
    data = json.loads(raw)
    worker = modal.Cls.from_name(APP,"PassengerPlaces")()
    options = {"recipe": data["recipe"]} if "recipe" in data else {}
    result = worker.interpret.remote(data["context"],data["input_sha256"],base64.b64decode(data["photo"],validate=True), **options)
    sys.stdout.buffer.write(json.dumps(result,allow_nan=False).encode("utf8"))


if __name__ == "__main__":
    main()
