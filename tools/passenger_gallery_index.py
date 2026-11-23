"""Measure gallery outline hints once for the public photo release."""
import argparse
import json
from pathlib import Path

from trackrag.dropoff import bay_anchors, whole_bay
from trackrag.passenger_app import load_scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(2)
    entries = {}
    for scene in load_scenes([args.requests]).values():
        frame = scene["frame"]
        key = scene["provenance"]["image_sha256"]
        anchor = bay_anchors.get_packet(key)
        if anchor is None and bay_anchors.get_row_packet(key) is None:
            continue
        shapes = None
        if anchor:
            body, _ = bay_anchors.checked_context(anchor,scene,(frame["width"],frame["height"]))
            shapes = {row["shape"] for row in body["rows"]}
        photo = Path(frame["image_path"]).read_bytes()
        stamp, visible = whole_bay.gallery_outline_entry(scene,photo,shapes,measure=True)
        if stamp:
            entries[stamp] = bool(visible)
    destination = whole_bay._GALLERY_INDEX
    pending = destination.with_suffix(".tmp")
    pending.write_text(json.dumps({"policy":whole_bay.CLEARANCE_POLICY, "outlines":entries},
                                  sort_keys=True, indent=2)+"\n", encoding="utf-8", newline="\n")
    pending.replace(destination)
    print(f"Measured {len(entries)} originals with a parking index; {sum(entries.values())} have an outline.")


if __name__ == "__main__":
    main()
