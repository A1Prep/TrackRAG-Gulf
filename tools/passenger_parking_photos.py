"""Fetch the original privacy-processed UAE photographs for the passenger release."""
import argparse
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse

from .passenger_release import ROOT, download


def write_request_bank(folder, metadata):
    """Resolve photographs beside the bank, as the standard passenger app expects."""
    published = {p["id"]: p["sha256"] for p in metadata["photos"]}
    prepared = {}
    for name in ("passenger-uae-gap-requests-20261123", "passenger-uae-parking-requests-20261123"):
        bank = ROOT / "experiments/dropoff" / (name + ".jsonl")
        for line in bank.read_text(encoding="utf8").splitlines():
            request = json.loads(line)
            case = request["request_id"]
            checksum = request["scene"]["provenance"]["image_sha256"]
            if case in prepared or checksum != published.get(case):
                raise ValueError(f"Passenger request photo does not match the manifest for {case}")
            request["scene"]["frame"]["image_path"] = checksum + ".jpg"
            prepared[case] = request
    if set(prepared) != set(published):
        raise ValueError("The prepared requests do not cover the published photo bank")
    target = folder / "passenger-requests.jsonl"
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as temp:
        pending = Path(temp.name)
        try:
            for case in published:
                temp.write((json.dumps(prepared[case], ensure_ascii=False) + "\n").encode("utf8"))
        except BaseException:
            temp.close()
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/uae-parking-study")
    args = parser.parse_args()
    manifest = ROOT / "experiments/dropoff/passenger-uae-parking-photos-20261123.json"
    metadata = json.loads(manifest.read_text(encoding="utf8"))
    for photo in metadata["photos"]:
        checksum, url = photo["sha256"], photo["image_url"]
        location = urlparse(url)
        if (not re.fullmatch("[0-9a-f]{64}", checksum) or location.scheme != "https"
                or location.hostname != "cdn.kartaview.org"):
            raise ValueError(f"Invalid published photograph entry for {photo['id']}")
        download(url, args.out / (checksum + ".jpg"), checksum, 40 * 1024**2)
    bank = write_request_bank(args.out, metadata)
    print(f"{len(metadata['photos'])} verified UAE photographs. Passenger request bank: {bank}")


if __name__ == "__main__":
    main()
