"""Request a signed frontage packet for an explicitly selected public photograph."""
import argparse
import asyncio
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


APP = "trackrag-passenger-frontage"
CLASS = "FrontageEvidence"
VERSION = 6
RECIPE = "molmo2_frontage_confidence_v1"
MAX_TRANSPORT = 58 * 1024 * 1024
MAX_PHOTO = 40 * 1024 * 1024


def require_releases(*deployments):
    """Check active release IDs through the same read-only RPCs used by `modal app history`."""
    from modal.client import _Client
    from modal._utils.async_utils import synchronizer
    from modal_proto import api_pb2

    async def check():
        client = await _Client.from_env()

        async def check_one(app, version):
            named = await client._stub.AppGetByDeploymentName(api_pb2.AppGetByDeploymentNameRequest(
                name=app,environment_name="main"))
            if not named.app_id:
                raise ValueError("Expected a deployed private frontage service")
            history = await client._stub.AppDeploymentHistory(api_pb2.AppDeploymentHistoryRequest(app_id=named.app_id))
            latest = next(iter(history.app_deployment_histories),None)
            if latest is None or latest.version!=version or latest.tag!=f"frontage-20261123-v{version}":
                raise ValueError(f"Unexpected active private frontage deployment for {app}")

        await asyncio.wait_for(asyncio.gather(*(check_one(app,version) for app,version in deployments)),timeout=15)

    synchronizer.create_blocking(check)()


def allowed_hashes(manifests):
    accepted = set()
    for path in manifests:
        for row in json.loads(path.read_bytes())["photos"]:
            if (not re.fullmatch(r"https://kartaview\.org/details/[1-9]\d*/\d+/?", row["photo_url"])
                    or not re.fullmatch(r"[a-f0-9]{64}",row["sha256"])):
                raise ValueError("Expected original public KartaView photo URLs and SHA-256 hashes")
            accepted.add(row["sha256"])
    return accepted


def main():
    from tools.passenger_release import PUBLIC_PHOTO_LISTS, ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", action="append", type=Path, help="Selected public photo source manifest (repeatable)")
    args = parser.parse_args()
    sources = args.sources or [ROOT/"experiments/dropoff"/name for name in PUBLIC_PHOTO_LISTS]
    allowed = allowed_hashes(sources)
    raw = sys.stdin.buffer.read(MAX_TRANSPORT+1)
    if len(raw)>MAX_TRANSPORT:
        raise ValueError("Frontage request exceeds the transport limit")
    payload = json.loads(raw)
    if set(payload) != {"context", "input_sha256", "photo"}:
        raise ValueError("Only the original photograph and current context may be submitted")
    photo = base64.b64decode(payload["photo"],validate=True)
    key = sha256(photo).hexdigest()
    if len(photo)>MAX_PHOTO or key != payload["context"]["image_sha256"] or key not in allowed:
        raise ValueError("Only the selected original public photographs can use frontage grounding")
    import modal

    require_releases((APP,VERSION))
    worker = modal.Cls.from_name(APP, CLASS, environment_name="main")()
    packet = worker.collect.remote(payload["context"], payload["input_sha256"], photo)
    require_releases((APP,VERSION))
    sys.stdout.buffer.write(json.dumps(packet,allow_nan=False).encode("utf8"))


if __name__ == "__main__":
    main()
