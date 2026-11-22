"""Export the frozen visible-road point research comparator without running a VLM.

Run from a checkout with the integration dependencies installed::

    python -m tools.passenger_road_points --requests requests.jsonl \
        --road-cache /path/to/cache --out /path/to/new-output

Requests use the ordinary annotation-free grounding contract.  The cache must
already contain the actual image-bound road masks.  This program never downloads
weights, loads a vision/language model, or changes the passenger application's
normal route.  Generated points are research alternatives, not verified roadside
gaps, stopping recommendations, or passenger confirmations.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.dropoff_predict import read_requests
from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.projected_pair import MOTOR
from trackrag.dropoff.road_cache import ROAD_RECIPE_SHA256, mask_sha256, mask_shape, read_mask


VERSION = "frozen_centroid_road_point_comparator_v1"
MARKED = (
    "You are looking at a forward-facing road photo. Each letter has a little pointer whose white dot is a proposed location. "
    "Only choose a letter when its dot is clearly on the requested roadside opening between the two named vehicles. "
    "Reject points beside the wrong cars, in through traffic, in a junction, or when a pair/its gap cannot be identified. "
    'Only answer with JSON {"choice":"A"} using an offered letter, or {"choice":null} to ask again. '
    "The mere presence of a mark never implies it should be chosen."
)


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _dimensions(track: dict) -> tuple[float, float, float, float]:
    left, top, right, bottom = track["bbox_xyxy"]
    return right - left, bottom - top, (left + right) / 2, bottom


def _current_request(request: dict) -> dict:
    """Reject historical observations rather than treating their boxes as current."""
    clean = validate_request(request)
    frame_id = clean["scene"]["frame"]["frame_id"]
    if any(track["frame_id"] != frame_id for track in clean["scene"]["tracks"]):
        raise ValueError("Road-point research requests must contain current-frame observations only")
    return clean


def pair_point(request: dict, track_ids: list[int], road: np.ndarray) -> list[float] | None:
    """Reproduce centered_gap.calculate's projected free-road centroid choice.

    This is an image-space geometry hypothesis.  It deliberately has no roadside
    or semantic acceptance decision; a separate model/reviewer can reject it.
    """
    scene = request["scene"]
    width, height = scene["frame"]["width"], scene["frame"]["height"]
    boxes = {track["track_id"]: track["bbox_xyxy"] for track in scene["tracks"]}
    pair = [boxes[identity] for identity in track_ids]
    mask_height, mask_width = road.shape
    dx, dy = width / mask_width, height / mask_height
    grid_y, grid_x = np.mgrid[:mask_height, :mask_width]
    x, y = (grid_x + .5) * dx, (grid_y + .5) * dy

    left, right = sorted(pair, key=lambda box: (box[0] + box[2]) / 2)
    far, near = sorted(pair, key=lambda box: box[3])
    contact_depth = near[3] - far[3]
    vehicle_height = min(box[3] - box[1] for box in pair)
    if left[2] < right[0]:
        region = (
            (x >= left[2]) & (x <= right[0])
            & (y >= far[3] - .20 * vehicle_height)
            & (y <= near[3] + .20 * vehicle_height)
        )
    elif contact_depth > 0:
        fraction = (y - far[3]) / contact_depth
        region = (
            (fraction >= 0) & (fraction <= 1)
            & (x > far[0] + fraction * (near[0] - far[0]))
            & (x < far[2] + fraction * (near[2] - far[2]))
        )
    else:
        return None

    valid = region & road
    for left, top, right, bottom in boxes.values():
        valid &= ~(
            (x + dx / 2 > left) & (x - dx / 2 < right)
            & (y + dy / 2 > top) & (y - dy / 2 < bottom)
        )
    free = cv2.erode(
        valid.astype(np.uint8), np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    if not free.any():
        return None

    rows, columns = np.nonzero(free)
    center_x = float(x[rows, columns].mean())
    center_y = float(y[rows, columns].mean())
    distance = np.where(free, (x - center_x) ** 2 + (y - center_y) ** 2, np.inf)
    row, column = np.unravel_index(distance.argmin(), road.shape)
    return [float(x[row, column]), float(y[row, column])]


def generate_options(request: dict, road: np.ndarray) -> list[dict]:
    """Reproduce the original 14-anchor search, ranking, deduplication and A–H IDs."""
    request = _current_request(request)
    frame = request["scene"]["frame"]
    size = (frame["width"], frame["height"])
    if not isinstance(road, np.ndarray) or road.dtype != np.bool_ or road.shape != mask_shape(size):
        raise ValueError("Road mask must match the registered recipe and original image dimensions")
    if "between" not in request["text"].lower():
        return []

    vehicles = [track for track in request["scene"]["tracks"] if track["class_name"] in MOTOR]
    largest = sorted(vehicles, key=lambda track: -(_dimensions(track)[0] * _dimensions(track)[1]))[:14]
    ranked = []
    for first, second in combinations(largest, 2):
        aw, ah, ax, ay = _dimensions(first)
        bw, bh, bx, by = _dimensions(second)
        if (max(ah, bh) > 3 * min(ah, bh) or abs(ay - by) > 1.25 * max(ah, bh)
                or abs(ax - bx) > 2.4 * max(aw, bw)):
            continue
        if abs(ax - bx) < .1 * min(aw, bw):
            continue
        blocked = False
        for other in vehicles:
            if other is first or other is second:
                continue
            _, ch, cx, cy = _dimensions(other)
            fraction = (cx - ax) / (bx - ax)
            if (.10 < fraction < .90 and min(ah, bh) * .6 < ch < max(ah, bh) * 1.5
                    and abs(cy - (ay + fraction * (by - ay))) < .5 * min(ah, bh)):
                blocked = True
                break
        if blocked:
            continue
        track_ids = [first["track_id"], second["track_id"]]
        point = pair_point(request, track_ids, road)
        if point is None:
            continue
        priority = math.sqrt(aw * ah * bw * bh) / (
            1 + abs(ax - bx) / max(aw, bw) + abs(ay - by) / max(ah, bh)
        )
        ranked.append((priority, {
            "point": point, "tracks": track_ids,
            "anchor_boxes": [first["bbox_xyxy"], second["bbox_xyxy"]],
        }))

    options = []
    for _, option in sorted(ranked, key=lambda item: -item[0]):
        x, y = option["point"]
        if any(math.hypot((x - prior["point"][0]) / size[0],
                          (y - prior["point"][1]) / size[1]) < .027 for prior in options):
            continue
        option["choice"] = chr(65 + len(options))
        options.append(option)
        if len(options) == 8:
            break
    return options


def render_choices(image_path: Path, options: list[dict], original_size: tuple[int, int],
                   output: Path, *, font_file: Path | None = None) -> tuple[list[int], list[dict]]:
    """Use the original thumbnail, pointer and JPEG recipe; font choice is explicit."""
    font = (ImageFont.truetype(str(font_file), 18) if font_file is not None
            else ImageFont.load_default(size=18))
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    try:
        image.thumbnail((1280, 800))
        draw = ImageDraw.Draw(image)
        scaled = []
        for option in options:
            x, y = [coordinate * factor for coordinate, factor in
                    zip(option["point"], (image.width / original_size[0], image.height / original_size[1]))]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="white", outline="black", width=2)
            text_x, text_y = min(image.width - 19, x + 7), max(0, y - 22)
            draw.line((x, y, text_x + 4, text_y + 17), fill="#f5c847", width=2)
            draw.rounded_rectangle((text_x, text_y, text_x + 19, text_y + 22), radius=3,
                                   fill="#181818", outline="#f5c847", width=1)
            draw.text((text_x + 3, text_y), option["choice"], font=font, fill="white")
            scaled.append({"choice": option["choice"], "point_marked_xy": [x, y],
                           "point_normalized_0_1000_xy": [
                               option["point"][0] * 1000 / original_size[0],
                               option["point"][1] * 1000 / original_size[1]]})
        image.save(output, quality=94)
        return [image.width, image.height], scaled
    finally:
        image.close()


def export(requests_path: Path, road_cache: Path, output: Path, *, font_file: Path | None = None) -> dict:
    """Validate sources first, then export annotation-free comparator artifacts."""
    requests_path, road_cache, output = map(Path, (requests_path, road_cache, output))
    if requests_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Request file exceeds the snapshot byte budget")
    snapshot = requests_path.read_bytes()
    requests = read_requests(requests_path, payload=snapshot)
    if font_file is not None and not Path(font_file).is_file():
        raise FileNotFoundError("The selected font file is missing")

    prepared = []
    for request in requests:
        request = _current_request(request)
        frame = request["scene"]["frame"]
        image_path = Path(frame["image_path"])
        image_sha = _digest(image_path.read_bytes())
        declared = request["scene"].get("provenance", {}).get("image_sha256")
        if declared != image_sha:
            raise ValueError(f"Original image SHA256 differs or is missing: {request['request_id']}")
        size = (frame["width"], frame["height"])
        with Image.open(image_path) as image:
            if image.size != size:
                raise ValueError(f"Original image dimensions differ: {request['request_id']}")
        road = read_mask(road_cache, image_sha, size)
        if road is None:
            raise FileNotFoundError(f"Prepare the original-bound road cache first: {request['request_id']}")
        prepared.append((request, image_path, image_sha, size, road))

    output.mkdir(parents=True, exist_ok=False)
    (output / "input-snapshot.jsonl").write_bytes(snapshot)
    (output / "selection-system-prompt.txt").write_text(MARKED + "\n", encoding="utf-8")
    records = []
    for number, (request, image_path, image_sha, size, road) in enumerate(prepared, 1):
        options = generate_options(request, road)
        # Never use an untrusted request ID as a filesystem path.
        marked = output / f"{number:04d}-{image_sha[:12]}-road-points.jpg"
        marked_size, scaled = render_choices(image_path, options, size, marked, font_file=font_file)
        system = MARKED + " Offered letters: " + ", ".join(option["choice"] for option in options)
        records.append({
            "id": request["request_id"], "request": request["text"],
            "original": str(image_path), "sha256": image_sha, "original_size": list(size),
            "marked": str(marked.resolve()), "options": options,
            "is_pair": "between" in request["text"].lower(),
            "marked_size": marked_size, "scaled_points": scaled,
            "marked_sha256": _digest(marked.read_bytes()), "road_mask_sha256": mask_sha256(road),
            "road_recipe_sha256": ROAD_RECIPE_SHA256, "system": system,
            "system_sha256": _digest(system.encode()),
        })
    choices_path = output / "road-choices.json"
    choices_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "kind": "offline_research_comparator_not_application_output", "generator": VERSION,
        "requests_sha256": _digest(snapshot), "source_sha256": _digest(Path(__file__).read_bytes()),
        "choices_sha256": _digest(choices_path.read_bytes()), "request_count": len(records),
        "candidate_count": sum(len(record["options"]) for record in records),
        "base_prompt_sha256": _digest(MARKED.encode()),
        "font": ({"file_name": Path(font_file).name, "sha256": _digest(Path(font_file).read_bytes())}
                 if font_file is not None else {"kind": "Pillow bundled default", "size": 18}),
        "rendering": {"thumbnail_max_wh": [1280, 800], "jpeg_quality": 94, "exif_transpose": False},
        "models_run": False, "labels_read": False, "application_route_changed": False,
        "limitations": [
            "These are free-road geometry alternatives; road pixels alone do not establish a roadside parking gap.",
            "The unchanged experimental prompt permits the VLM to reject every option using choice null.",
            "Through traffic, junctions, wrong references and no eligible gap remain possible generator errors.",
            "No physical-clearance, stopping-legality or vehicle-control claim is made.",
            "Exact rendering reproduction requires the original font and compatible Pillow/JPEG versions.",
        ],
    }
    (output / "research-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", required=True, type=Path, help="Annotation-free current-frame requests JSONL")
    parser.add_argument("--road-cache", required=True, type=Path, help="Existing cache root containing passenger-road/")
    parser.add_argument("--out", required=True, type=Path, help="New directory for marked photographs, options and prompt")
    parser.add_argument("--font-file", type=Path, help="Optional TTF/OTF font; use the original bold font for matching renders")
    args = parser.parse_args(argv)
    try:
        result = export(args.requests, args.road_cache, args.out, font_file=args.font_file)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Road-point research export failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
