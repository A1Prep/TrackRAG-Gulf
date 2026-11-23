"""Use the current Modal user's private GPU function; stdin/stdout contain one bounded request/response."""
import base64
import json
import sys

APP = "trackrag-passenger-parking"
CLASS = "PassengerParking"


def call(request, image, recipe=None):
    import modal
    worker = modal.Cls.from_name(APP, CLASS)()
    if recipe is None:
        return worker.interpret.remote(request, image)
    return worker.interpret.remote(request, image, recipe)


def main():
    data=sys.stdin.buffer.read(58*1024*1024+1)
    if len(data)>58*1024*1024:
        raise ValueError("Parking request exceeds the transport limit")
    payload=json.loads(data)
    result=call(payload["request"],base64.b64decode(payload["image"],validate=True),payload.get("recipe"))
    sys.stdout.buffer.write(json.dumps(result,allow_nan=False).encode("utf8"))


if __name__=="__main__":
    main()
