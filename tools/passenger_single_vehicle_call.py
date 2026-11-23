"""Authenticated bridge to the private, scale-to-zero single-vehicle visual reader."""
import base64
import json
import sys

APP="trackrag-passenger-single-vehicle"


def main():
    import modal
    raw=sys.stdin.buffer.read(58*1024*1024+1)
    if len(raw)>58*1024*1024:raise ValueError("Single-vehicle request is too large")
    job=json.loads(raw)
    service=modal.Cls.from_name(APP,"SingleVehicleReader")()
    packet=service.observe.remote(job["context"],job["key"],base64.b64decode(job["photo"],validate=True))
    sys.stdout.buffer.write(json.dumps(packet,allow_nan=False).encode("utf8"))


if __name__=="__main__":main()
