r"""Scores saved drop-off grounding outputs against separate reference files.

It never runs or imports a model and never repairs an output. A reserved case is a case in the references file. A
missing output, a request error and an invalid output all count as failures and stay in the denominator of every main
rate, so declining to answer cannot raise a score.

    python -m evaluation.grounding --references refs/references.jsonl --scenes scenes/ \
        --outputs outputs-pretrained-direct.jsonl outputs-trained-direct.jsonl ... [--replies replies.jsonl] \
        [--voice voice.jsonl] [--protocol protocol.json] [--train-manifests M ...] --out summary.json \
        [--cases-out cases.csv]

Task 2's test cases and Task 5's prediction runs, in their own formats, are turned into these inputs by
``python -m evaluation.grounding_inputs``. Every input format and scoring rule is described in
docs/task1-grounding-evaluation.md.

Conventions shared with trackrag/dropoff/contracts.py and trackrag/dropoff/data.py: coordinates are original-image
pixels; a point lies in [0, W) x [0, H) and a box in [0, W] x [0, H] with positive area. A PNG mask is read as
greyscale, nonzero is inside, and it may have another size than the image: a point maps to mask pixel
(floor(x * mask_width / W), floor(y * mask_height / H)), and the mask's box is its pixel extent scaled to the image.
Text files are hashed with CRLF read as LF; masks by their exact bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = ("clear", "ambiguous", "missing_reference")
STATUSES = ("proposed", "needs_clarification", "insufficient_visual_evidence")
FAILURES = ("missing", "request_error", "wrong_scene", "wrong_instruction", "invalid_output")
EXPECTED = {"clear": {"proposed"}, "ambiguous": {"needs_clarification"},
            "missing_reference": {"needs_clarification", "insufficient_visual_evidence"}}
MODELS, MODES, TRANSCRIPTS = ("pretrained", "trained"), ("direct", "clarification"), ("typed", "human", "asr")
REQUIRED_CONDITIONS = tuple(f"{model}/{mode}/typed" for model in MODELS for mode in MODES)
REQUIRED_COMPARISONS = tuple((f"pretrained/{mode}/typed", f"trained/{mode}/typed") for mode in MODES)
REPLY_KINDS = ("answer", "rejection")
PROVENANCE_FIELDS = ("base_revision", "adapter_sha256", "settings_sha256", "image_budget", "context_policy",
                     "coordinate_convention", "turn_budget")
MATCH_ACROSS_MODELS = ("base_revision", "image_budget", "context_policy", "coordinate_convention", "turn_budget")
MATCH_ACROSS_MODES = ("base_revision", "adapter_sha256", "image_budget", "context_policy", "coordinate_convention")
MATCH_ACROSS_TRANSCRIPTS = MATCH_ACROSS_MODELS + ("adapter_sha256", "settings_sha256")
# Recordings that run straight into each other (see tools/vision_prepare_dataset.py, EXCLUDED_FROM_TRAINING).
ADJACENT = {"video_054604": "video_054907", "video_054907": "video_054604",
            "video_204347": "video_204647", "video_204647": "video_204347"}
PROTOCOL_DEFAULTS = {"box_iou": 0.5, "landmark_iou": 0.5, "baseline_point": [0.5, 0.8], "near_constant_spread": 0.01,
                     "bootstrap": {"resamples": 2000, "seed": 20261122}}
PROTOCOL_REQUIRED = ("frozen_utc", "evaluator_commit", "references_sha256", "responder_rules")
PROTOCOL_NULLABLE = ("replies_sha256", "voice_sha256")
STUB_LABEL = "scenes made by a stand-in detector (tests only)"
TASK5_FIELDS = {"destination_point_xy": "point", "destination_box_xyxy": "box",
                "clarification_question": "clarification", "anchor_boxes_xyxy": "landmark_boxes"}
TASK5_REQUESTED = {"pretrained": "pretrained", "trained": "fine_tuned"}  # provenance.requested_condition
APOSTROPHES = ("'", "’")


class GroundingError(ValueError):
    """Inputs that cannot be scored as given."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GroundingError(message)


def _number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _text(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path) -> bytes:
    try:
        return Path(path).read_bytes()
    except (OSError, ValueError) as error:
        raise GroundingError(f"Cannot read {path!r}: {error}") from error


def sha256_file(path) -> str:
    """sha256 of a file's exact bytes (masks, images, Task 5 run files)."""
    return _sha256_bytes(_read_bytes(path))


def text_sha256(path) -> str:
    """sha256 of a text file with CRLF line endings read as LF, so Windows and Linux checkouts agree."""
    return _sha256_bytes(_read_bytes(path).replace(b"\r\n", b"\n"))


def instruction_sha256(text: str, dialogue=()) -> str:
    """sha256 of the instruction a model received: its text and any earlier dialogue turns, as canonical JSON."""
    payload = json.dumps({"dialogue": list(dialogue), "text": text}, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False)
    return _sha256_bytes(payload.encode("utf-8"))


def read_json(path):
    try:
        return json.loads(_read_bytes(path).decode("utf-8-sig"))
    except (ValueError, RecursionError) as error:
        raise GroundingError(f"Cannot read {path}: {error}") from error


def read_jsonl(path) -> list[dict]:
    """One JSON object per line. Lines end at LF (with an optional CR) only, so a U+2028 inside a string is safe."""
    try:
        text = _read_bytes(path).decode("utf-8-sig")
    except ValueError as error:
        raise GroundingError(f"Cannot read {path}: {error}") from error
    rows = []
    for number, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if line.strip():
            try:
                row = json.loads(line)
            except (ValueError, RecursionError) as error:
                raise GroundingError(f"{path}:{number}: not JSON ({error})") from error
            _require(isinstance(row, dict), f"{path}:{number}: each line must be a JSON object")
            rows.append(row)
    return rows


# --- geometry ----------------------------------------------------------------------------------------------------

def valid_point(point, width: int, height: int) -> bool:
    return (isinstance(point, list) and len(point) == 2 and all(_number(v) for v in point)
            and 0 <= point[0] < width and 0 <= point[1] < height)


def valid_box(box, width: int, height: int) -> bool:
    return (isinstance(box, list) and len(box) == 4 and all(_number(v) for v in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def box_iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_mask(path):
    """A boolean mask from a PNG file, read as greyscale with nonzero inside (as Task 5's case loader reads it)."""
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as image:
            _require(image.format == "PNG", f"{path}: masks must be PNG files")
            return np.array(image.convert("L")) != 0
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as error:
        raise GroundingError(f"Cannot read mask {path!r}: {error}") from error


def mask_cell(mask, point, width: int, height: int) -> tuple[int, int]:
    rows, columns = mask.shape
    return (min(int(math.floor(point[0] * columns / width)), columns - 1),
            min(int(math.floor(point[1] * rows / height)), rows - 1))


def in_mask(mask, point, width: int, height: int) -> bool:
    x, y = mask_cell(mask, point, width, height)
    return bool(mask[y, x])


def mask_box(mask, width: int, height: int) -> list[float] | None:
    """The mask's pixel extent [x0, y0, x1 + 1, y1 + 1], scaled to the image ("pixel_bbox")."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    rows, columns = mask.shape
    return [float(xs.min()) * width / columns, float(ys.min()) * height / rows,
            float(xs.max() + 1) * width / columns, float(ys.max() + 1) * height / rows]


def distance_to_mask(mask, point, width: int, height: int) -> float:
    """Distance in image pixels from the point's mask cell to the nearest mask cell (0 inside)."""
    import numpy as np

    x, y = mask_cell(mask, point, width, height)
    if mask[y, x]:
        return 0.0
    rows, columns = mask.shape
    ys, xs = np.nonzero(mask)
    return float(np.sqrt((((xs - x) * width / columns) ** 2 + ((ys - y) * height / rows) ** 2).min()))


# --- statistics --------------------------------------------------------------------------------------------------

def wilson(count: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.959963984540054, count / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def mcnemar_exact(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_first + only_second
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_first, only_second) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def group_bootstrap(values_by_group: dict, resamples: int, seed: int) -> list[float] | None:
    """95% percentile interval of the overall mean, resampling whole groups with replacement (needs 2+ groups)."""
    groups = [g for g in sorted(values_by_group) if values_by_group[g]]
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        picked = [values_by_group[rng.choice(groups)] for _ in groups]
        means.append(sum(sum(v) for v in picked) / sum(len(v) for v in picked))
    means.sort()
    return [means[int(0.025 * (resamples - 1))], means[int(0.975 * (resamples - 1))]]


def rate(cases: list[dict], key: str, protocol: dict) -> dict:
    """How many of the given cases have ``key`` true, with a Wilson interval and a group bootstrap interval."""
    by_group = defaultdict(list)
    for case in cases:
        by_group[case["group_id"]].append(1.0 if case[key] else 0.0)
    count, n = sum(bool(c[key]) for c in cases), len(cases)
    boot = protocol["bootstrap"]
    return {"count": count, "n": n, "rate": count / n if n else None, "groups": len(by_group),
            "wilson95_assumes_independent_cases": wilson(count, n),
            "group_bootstrap95": group_bootstrap(by_group, boot["resamples"], boot["seed"])}


def mean_with_interval(cases: list[dict], key: str, protocol: dict) -> dict:
    by_group = defaultdict(list)
    for case in cases:
        if case.get(key) is not None:
            by_group[case["group_id"]].append(float(case[key]))
    values = [v for group in by_group.values() for v in group]
    boot = protocol["bootstrap"]
    return {"mean": statistics.fmean(values) if values else None, "n": len(values),
            "group_bootstrap95": group_bootstrap(by_group, boot["resamples"], boot["seed"])}


# --- loading and checking inputs ---------------------------------------------------------------------------------

def load_protocol(path) -> tuple[dict, str | None]:
    protocol = json.loads(json.dumps(PROTOCOL_DEFAULTS))
    if path is None:
        return protocol, None
    given = read_json(path)
    _require(isinstance(given, dict) and given.get("version") == 1, "The protocol needs version 1")
    missing = [k for k in PROTOCOL_REQUIRED if not _text(given.get(k))]
    _require(not missing, f"The frozen protocol must record {', '.join(missing)}")
    for key in PROTOCOL_NULLABLE:
        _require(key in given and (given[key] is None or _text(given[key])),
                 f"The frozen protocol must record {key} (null when that file is not used)")
    protocol.update({k: given[k] for k in PROTOCOL_DEFAULTS if k in given})
    for name in ("box_iou", "landmark_iou"):
        _require(_number(protocol[name]) and 0 < protocol[name] <= 1, f"protocol.{name} must be in (0, 1]")
    _require(_number(protocol["near_constant_spread"]) and 0 < protocol["near_constant_spread"] < 1,
             "protocol.near_constant_spread must be in (0, 1)")
    point = protocol["baseline_point"]
    _require(isinstance(point, list) and len(point) == 2 and all(_number(v) and 0 <= v < 1 for v in point),
             "protocol.baseline_point must be two fractions in [0, 1)")
    boot = protocol["bootstrap"]
    _require(isinstance(boot, dict) and type(boot.get("resamples")) is int and 100 <= boot["resamples"] <= 100000
             and type(boot.get("seed")) is int, "protocol.bootstrap needs integer resamples (100 to 100000) and seed")
    protocol["bootstrap"] = {"resamples": boot["resamples"], "seed": boot["seed"]}
    protocol.update({k: given[k] for k in PROTOCOL_REQUIRED + PROTOCOL_NULLABLE})
    return protocol, text_sha256(path)


def load_scenes(folder, allow_stub: bool = False) -> dict:
    """Scene files by scene_sha256. The folder holds what ``trackrag.dropoff.scenes export`` writes: scene files
    and index.json."""
    from trackrag.dropoff.scenes import SceneError, validate_scene

    scenes = {}
    for path in sorted(Path(folder).glob("*.json")):
        if path.name == "index.json":
            continue
        scene = read_json(path)
        try:
            validate_scene(scene, allow_stub=allow_stub)
        except SceneError as error:
            raise GroundingError(f"Scene file {path.name} was refused: {error}") from error
        scenes[scene["scene_sha256"]] = scene
    _require(len(scenes) > 0, f"No scene files in {folder}")
    return scenes


def _pinned_mask(folder: Path, row: dict, field: str, where: str):
    _require(_text(row.get(field)), f"{where}: {field} is required")
    path = folder / row[field]
    _require(row.get(f"{field}_sha256") == sha256_file(path),
             f"{where}: {field}_sha256 does not match the file {row[field]}")
    mask = load_mask(path)
    _require(bool(mask.any()), f"{where}: the {field.replace('_', ' ')} is empty; fix the reference before scoring")
    return mask


def load_references(path, scenes: dict, *, allow_unreviewed: bool = False) -> tuple[dict, list[str]]:
    """Reference cases by case_id, with masks loaded and every documented rule checked."""
    path = Path(path)
    warnings, references = [], {}
    for row in read_jsonl(path):
        case = row.get("case_id")
        _require(_text(case) and case not in references, f"Missing or repeated case_id {case!r}")
        where = f"reference {case}"
        for key in ("group_id", "scene_sha256", "recording_id", "category", "expected_status", "instruction"):
            _require(_text(row.get(key)), f"{where}: {key} is required")
        _require(row.get("partition") == "final_evaluation", f"{where}: partition must be final_evaluation")
        _require(row["category"] in CATEGORIES, f"{where}: unknown category {row['category']!r}")
        _require(row["expected_status"] in EXPECTED[row["category"]],
                 f"{where}: expected_status {row['expected_status']!r} does not fit category {row['category']!r}")
        dialogue = row.get("dialogue") if row.get("dialogue") is not None else []
        _require(isinstance(dialogue, list), f"{where}: dialogue must be a list")
        scene = scenes.get(row["scene_sha256"])
        _require(scene is not None, f"{where}: its scene {row['scene_sha256'][:12]} is not among the scene files")
        model_input = scene["model_input"]
        for key in ("recording_id", "frame_id", "image_sha256", "width", "height"):
            _require(row.get(key) == model_input[key], f"{where}: {key} differs from its scene")
        width, height = model_input["width"], model_input["height"]
        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        reviews = review.get("independent_reviews")
        if not (type(reviews) is int and reviews >= 2 and review.get("disagreements_resolved") is True):
            _require(allow_unreviewed, f"{where}: needs two independent reviews with disagreements resolved")
            warnings.append(f"{case}: reference not fully reviewed")
        ref = {"case_id": case, "group_id": row["group_id"], "category": row["category"],
               "expected_status": row["expected_status"], "scene_sha256": row["scene_sha256"],
               "recording_id": row["recording_id"], "width": width, "height": height, "dialogue": dialogue,
               "instruction_sha256": instruction_sha256(row["instruction"], dialogue),
               "mask": None, "target_box": None, "intended_mask": None,
               "anchor_roles_ordered": row.get("anchor_roles_ordered", False) is True}
        if row["category"] == "clear":
            mask = _pinned_mask(path.parent, row, "mask", where)
            _require(row.get("box_rule") == "pixel_bbox", f"{where}: unknown box rule {row.get('box_rule')!r}")
            expected_box, given = mask_box(mask, width, height), row.get("target_box")
            _require(valid_box(given, width, height) and all(abs(a - b) <= 1e-6 for a, b in zip(given, expected_box)),
                     f"{where}: target_box {given} differs from the mask's pixel_bbox {expected_box}")
            point = row.get("reference_point")
            _require(valid_point(point, width, height) and in_mask(mask, point, width, height),
                     f"{where}: reference_point must lie inside the mask")
            ref.update(mask=mask, target_box=given)
        elif row["category"] == "ambiguous":
            _require(row.get("mask") is None, f"{where}: an ambiguous case has no single destination mask; "
                                              "its intended destination goes in intended_mask")
            ref["intended_mask"] = _pinned_mask(path.parent, row, "intended_mask", where)
        else:
            _require(row.get("mask") is None and row.get("intended_mask") is None,
                     f"{where}: a missing-reference case has no destination mask")
        anchors = row.get("anchors") if row.get("anchors") is not None else []
        _require(isinstance(anchors, list), f"{where}: anchors must be a list")
        for anchor in anchors:
            _require(isinstance(anchor, dict) and _text(anchor.get("role"))
                     and valid_box(anchor.get("bbox_xyxy"), width, height), f"{where}: invalid anchor {anchor!r}")
        _require(len({a["role"] for a in anchors}) == len(anchors), f"{where}: anchor roles must be unique")
        ref["anchors"] = anchors
        references[case] = ref
    _require(len(references) > 0, f"No reference cases in {path}")
    return references, warnings


def load_replies(path, references: dict) -> dict:
    replies = {}
    if path is None:
        return replies
    for number, row in enumerate(read_jsonl(path), 1):
        where = f"replies line {number}"
        _require(_text(row.get("case_id")) and row["case_id"] in references, f"{where}: case_id must be a reserved case")
        _require(_text(row.get("reply_id")), f"{where}: reply_id is required")
        _require(row.get("kind") in REPLY_KINDS, f"{where}: kind must be one of {', '.join(REPLY_KINDS)}")
        _require(_text(row.get("text")), f"{where}: text is required")
        key = (row["case_id"], row["reply_id"])
        _require(key not in replies, f"{where}: reply {row['reply_id']} is repeated for case {row['case_id']}")
        replies[key] = dict(row, sha256=_sha256_bytes(row["text"].encode("utf-8")))
    return replies


def load_voice(path, references: dict) -> list[dict]:
    rows, seen = read_jsonl(path), set()
    for row in rows:
        case = row.get("case_id")
        _require(_text(case) and case in references, f"Voice case {case!r} is not a reserved reference")
        _require(case not in seen, f"Repeated voice case {case!r}")
        seen.add(case)
        _require(row.get("consent") is True and _text(row.get("source")) and _text(row.get("recognizer")),
                 f"voice case {case}: consent, source and recognizer are required")
        _require(row.get("speech") in ("read", "spontaneous") and row.get("noise") in ("none", "real", "added"),
                 f"voice case {case}: speech must be read or spontaneous, noise none, real or added")
        _require(isinstance(row.get("human_transcript"), str), f"voice case {case}: human_transcript must be text")
        asr = row.get("asr_transcript")
        _require(asr is None or isinstance(asr, str), f"voice case {case}: asr_transcript must be text or null")
        labels = row.get("critical_errors") if row.get("critical_errors") is not None else []
        _require(isinstance(labels, list) and all(_text(x) for x in labels), f"voice case {case}: bad critical_errors")
    return rows


def condition_key(condition) -> str:
    _require(isinstance(condition, dict) and condition.get("model") in MODELS and condition.get("mode") in MODES
             and condition.get("transcript") in TRANSCRIPTS,
             f"Invalid condition {condition!r}: model must be one of {MODELS}, mode one of {MODES}, "
             f"transcript one of {TRANSCRIPTS}")
    return f"{condition['model']}/{condition['mode']}/{condition['transcript']}"


def _refusal(row: dict, provenance: dict) -> str | None:
    if provenance.get("substitute") is not False:
        return "substitute or stub output"
    if not (_text(provenance.get("base_revision")) and _text(provenance.get("settings_sha256"))
            and (row["condition"]["model"] == "pretrained" or _text(provenance.get("adapter_sha256")))):
        return "model identity missing (base_revision, settings_sha256, adapter_sha256)"
    if row["condition"]["model"] == "pretrained" and provenance.get("adapter_sha256") is not None:
        return "pretrained output names an adapter"
    if provenance.get("coordinate_convention") != "original_pixels":
        return "coordinates are not original pixels"
    if provenance.get("image_budget") is None or provenance.get("context_policy") is None:
        return "setup missing (image_budget, context_policy)"
    return _envelope_refusal(row, provenance)


def _envelope_refusal(row: dict, provenance: dict) -> str | None:
    """Clarification turns that carry Task 5 outcomes must come from the real model of the record's condition."""
    if row["condition"]["mode"] != "clarification" or not isinstance(row.get("turns"), list):
        return None
    for turn in row["turns"]:
        if not isinstance(turn, dict) or "outcome" not in turn:
            continue
        outcome = turn["outcome"]
        source = outcome.get("provenance") if isinstance(outcome, dict) else None
        if not isinstance(source, dict):
            return "a turn's Task 5 outcome has no provenance"
        if outcome.get("request_id") != row["case_id"]:
            return "a turn's Task 5 outcome is for another request"
        if source.get("injected") is not False or source.get("backend_kind") != "transformers_local":
            return "a turn came from a stand-in or another backend"
        if source.get("requested_condition") != TASK5_REQUESTED[row["condition"]["model"]]:
            return "a turn came from another model condition"
        if f"{source.get('model_id')}@{source.get('revision')}" != provenance.get("base_revision"):
            return "a turn came from another base model"
        adapter = source.get("adapter")
        if adapter is not None and (not isinstance(adapter, dict)
                                    or adapter.get("manifest_sha256") != provenance.get("adapter_sha256")):
            return "a turn came from another adapter"
    return None


def load_outputs(paths, references: dict, voice_cases: set | None = None) -> tuple[dict, list[dict]]:
    """Outputs by condition and case, from one or more files. Refused outputs are listed and scored as missing. One
    condition must be one model setup: every record in it carries the same provenance values. Human and ASR
    transcripts exist only for the voice cases."""
    by_condition: dict[str, dict] = defaultdict(dict)
    refused = []
    for path in paths:
        for row in read_jsonl(path):
            case = row.get("case_id")
            _require(_text(case) and case in references, f"{Path(path).name}: output for unknown case {case!r}")
            key = condition_key(row.get("condition"))
            _require(case not in by_condition[key], f"Two outputs for case {case} in condition {key}")
            if row["condition"]["transcript"] != "typed":
                _require(voice_cases is not None, "Outputs from human or ASR transcripts need the voice file (--voice)")
                _require(case in voice_cases, f"Output {case} in {key} is not a voice case")
            _require(isinstance(row.get("provenance"), dict), f"Output {case} in {key} has no provenance object")
            reason = _refusal(row, row["provenance"])
            if reason:
                refused.append({"case_id": case, "condition": key, "reason": reason})
                row = None
            by_condition[key][case] = row
    for key, records in by_condition.items():
        for field in PROVENANCE_FIELDS:
            values = {json.dumps(r["provenance"].get(field), sort_keys=True) for r in records.values() if r}
            _require(len(values) <= 1, f"Condition {key} mixes values of provenance.{field}")
    return dict(by_condition), refused


# --- scoring one output ------------------------------------------------------------------------------------------

def from_task5_outcome(outcome) -> tuple[dict, dict | None, str | None]:
    """The call, the result and an invalid-output reason for one Task 5 outcome (GroundingSuccess or
    GroundingFailure, trackrag/dropoff/contracts.py). A model response that could not be parsed is an invalid output
    of a call that ran, not a request error."""
    ok = {"outcome": "ok", "error_code": None}
    if not isinstance(outcome, dict) or type(outcome.get("ok")) is not bool:
        return {"outcome": None, "error_code": None}, None, None
    if outcome["ok"] is False:
        error = outcome.get("error") if isinstance(outcome.get("error"), dict) else {}
        code = error.get("code") if isinstance(error.get("code"), str) else None
        if code == "invalid_model_output":
            return ok, None, "model output could not be parsed (invalid_model_output)"
        return {"outcome": "error", "error_code": code}, None, None
    result = outcome.get("result")
    if not isinstance(result, dict):
        return ok, None, None
    mapped = {TASK5_FIELDS.get(key, key): value for key, value in result.items()}
    if mapped.get("landmark_boxes") is None:
        mapped.pop("landmark_boxes", None)
    return ok, mapped, None


def _classify(result, width: int, height: int, case: dict, invalid_reason=None) -> str:
    """The semantic status of a parsed result, or invalid_output with the reason added to the case."""
    problem = None
    if not isinstance(result, dict):
        problem = invalid_reason if isinstance(invalid_reason, str) else "no parsed result"
    elif result.get("status") not in STATUSES:
        problem = f"unknown status {result.get('status')!r}"
    elif result["status"] == "proposed" and not valid_point(result.get("point"), width, height):
        problem = "proposed without a valid point inside the image"
    elif result["status"] == "proposed" and not valid_box(result.get("box"), width, height):
        problem = "proposed without a valid box inside the image"
    elif result["status"] == "proposed" and not (result["box"][0] <= result["point"][0] <= result["box"][2]
                                                 and result["box"][1] <= result["point"][1] <= result["box"][3]):
        problem = "point outside its own box"
    elif not isinstance(result.get("landmarks", []), list) or not all(
            isinstance(m, dict) and isinstance(m.get("role"), str) and type(m.get("track_id")) is int
            for m in result.get("landmarks", [])):
        problem = "malformed landmarks"
    elif "landmark_boxes" in result and not (
            isinstance(result["landmark_boxes"], list) and len(result["landmark_boxes"]) <= 16
            and all(valid_box(b, width, height) for b in result["landmark_boxes"])):
        problem = "malformed landmark boxes"
    if problem:
        case["reasons"].append(problem)
        return "invalid_output"
    return result["status"]


def _place(result: dict, reference: dict, protocol: dict) -> dict:
    """A valid proposal against the applicable mask (clear: destination; ambiguous: intended destination)."""
    if reference["category"] == "missing_reference":
        return {"point_in_mask": False, "distance_px": None, "box_iou": None, "box_success": False, "incorrect": True}
    width, height = reference["width"], reference["height"]
    mask = reference["mask"] if reference["category"] == "clear" else reference["intended_mask"]
    inside = in_mask(mask, result["point"], width, height)
    scores = {"point_in_mask": inside, "distance_px": distance_to_mask(mask, result["point"], width, height),
              "box_iou": None, "box_success": False, "incorrect": not inside}
    if reference["category"] == "clear":
        iou = box_iou(result["box"], reference["target_box"])
        scores.update(box_iou=iou, box_success=iou >= protocol["box_iou"])
    return scores


def _matched_anchors(edges: dict) -> set:
    """Anchors covered by a maximum one-to-one matching between anchors and named items."""
    owner = {}

    def augment(anchor, seen):
        for item in edges[anchor]:
            if item not in seen:
                seen.add(item)
                if item not in owner or augment(owner[item], seen):
                    owner[item] = anchor
                    return True
        return False

    for anchor in edges:
        augment(anchor, set())
    return set(owner.values())


def anchors_detected(reference: dict, scene: dict, protocol: dict) -> list[bool]:
    """Whether the frozen scene holds any observation matching each anchor (a vision limit, the same for every model)."""
    boxes = [o["bbox_xyxy"] for o in scene["model_input"]["observations"]]
    return [any(box_iou(b, a["bbox_xyxy"]) >= protocol["landmark_iou"] for b in boxes) for a in reference["anchors"]]


def landmark_score(result, reference: dict, scene: dict, protocol: dict) -> dict | None:
    """Landmarks named by the model against the reference anchors, given as track IDs of the scene (``landmarks``)
    or as boxes (``landmark_boxes``, Task 5's anchor_boxes_xyxy). All are correct only when the model names exactly
    one distinct item per anchor and each matches its anchor at the protocol's IoU; roles are checked when ordered
    and given, and boxes carry no roles."""
    anchors = reference["anchors"]
    if not anchors:
        return None
    threshold = protocol["landmark_iou"]
    given_boxes = result.get("landmark_boxes") if isinstance(result, dict) else None
    if given_boxes is not None:
        correct = _matched_anchors({i: [j for j, b in enumerate(given_boxes) if box_iou(b, a["bbox_xyxy"]) >= threshold]
                                    for i, a in enumerate(anchors)})
        named, invalid, distinct, form = len(given_boxes), 0, True, "boxes"
    else:
        boxes = {o["track_id"]: o["bbox_xyxy"] for o in scene["model_input"]["observations"]}
        listed = result.get("landmarks", []) if isinstance(result, dict) else []
        known = [m for m in listed if m["track_id"] in boxes]
        distinct = len({m["track_id"] for m in known}) == len(known)
        if reference["anchor_roles_ordered"]:
            correct = set()
            for index, anchor in enumerate(anchors):
                chosen = [m for m in known if m["role"] == anchor["role"]]
                if len(chosen) == 1 and box_iou(boxes[chosen[0]["track_id"]], anchor["bbox_xyxy"]) >= threshold:
                    correct.add(index)
        else:
            tracks = sorted({m["track_id"] for m in known})
            correct = _matched_anchors({i: [t for t in tracks if box_iou(boxes[t], a["bbox_xyxy"]) >= threshold]
                                        for i, a in enumerate(anchors)})
        named, invalid, form = len(listed), len(listed) - len(known), "track_ids"
    all_correct = len(correct) == len(anchors) == named and distinct and invalid == 0
    return {"form": form, "anchors": len(anchors), "named": named, "invalid_references": invalid,
            "extra": max(0, named - len(anchors)), "correct": len(correct), "all_correct": all_correct,
            "pair_correct": all_correct if len(anchors) >= 2 else None,
            "ordered_roles_not_checked": form == "boxes" and reference["anchor_roles_ordered"]}


def _conversion_mismatch(raw, result: dict, width: int, height: int) -> bool | None:
    """For a raw response holding 0-1000 values (Task 5's destination_point_xy / destination_box_xyxy, or Qwen's
    point_2d / bbox_2d): do they convert to the parsed pixels within 1 px? None when there is nothing to check."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):  # Qwen often wraps its JSON in a code fence
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if isinstance(parsed, list) and len(parsed) == 1:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return None
    checks = []
    for keys, field, size in ((("destination_point_xy", "point_2d"), "point", 2),
                              (("destination_box_xyxy", "bbox_2d"), "box", 4)):
        present = [key for key in keys if key in parsed]
        if present:
            checks.append((parsed[present[0]], result[field], size))
    if not checks:
        return None
    for values, pixels, size in checks:
        if not (isinstance(values, list) and len(values) == size and all(_number(v) for v in values)):
            return True
        scale = [width, height] * (size // 2)
        if any(abs(v / 1000 * s - p) > 1.0 for v, s, p in zip(values, scale, pixels)):
            return True
    return False


def _turn_outcome(turn: dict, reference: dict, case: dict) -> tuple[str, dict | None]:
    """One clarification-mode turn, by the same precedence as a whole output. A turn holds either ``call`` and
    ``result`` or Task 5's outcome envelope as ``outcome``."""
    if "outcome" in turn:
        call, result, invalid_reason = from_task5_outcome(turn["outcome"])
    else:
        call, result, invalid_reason = turn.get("call"), turn.get("result"), None
    if isinstance(call, dict) and call.get("outcome") == "error":
        case["error_code"] = case["error_code"] or call.get("error_code")
        return "request_error", None
    if turn.get("scene_sha256") != reference["scene_sha256"]:
        return "wrong_scene", None
    if not isinstance(call, dict) or call.get("outcome") != "ok":
        case["reasons"].append("a turn without a call outcome")
        return "invalid_output", None
    return _classify(result, reference["width"], reference["height"], case, invalid_reason), result


def _reply_problem(turn: dict, previous: str, reference: dict, replies: dict) -> str | None:
    reply_id = turn.get("reply_id")
    reply = replies.get((reference["case_id"], reply_id)) if _text(reply_id) else None
    if reply is None:
        return "reply is not pre-registered for this case"
    if reply["kind"] != ("rejection" if previous == "proposed" else "answer"):
        return "reply kind does not follow the responder rule"
    if turn.get("reply_sha256") != reply["sha256"]:
        return "reply_sha256 does not match the registered reply text"
    return None


def _episode(record: dict, reference: dict, replies: dict, case: dict) -> dict:
    """Clarification mode: turn 0 is the initial output; any later proposal is after intervention."""
    turns = record.get("turns")
    budget = record["provenance"].get("turn_budget")
    broken = None
    if not isinstance(turns, list) or not turns or not all(isinstance(t, dict) for t in turns):
        broken = "clarification output without turns"
    elif [t.get("turn") if type(t.get("turn")) is int else None for t in turns] != list(range(len(turns))):
        broken = "turns must be numbered 0, 1, 2, ..."
    elif turns[0].get("reply_id") is not None:
        broken = "turn 0 must come before any passenger reply"
    elif type(budget) is not int or budget < 1:
        broken = "missing turn budget"
    if broken:
        case["reasons"].append(broken)
        return {"initial_outcome": "invalid_output", "outcome": "invalid_output"}
    scored = [_turn_outcome(t, reference, case) for t in turns]
    outcomes = [outcome for outcome, _ in scored]
    problem, seen = None, set()
    for index, turn in enumerate(turns):
        proposal_id = turn.get("proposal_id")
        if proposal_id is not None:
            if not _text(proposal_id) or proposal_id in seen:
                problem = "proposal ID malformed or reused"
                break
            seen.add(proposal_id)
        if turn.get("confirmed_proposal_id") is not None and index < len(turns) - 1:
            problem = "the episode continues after a confirmation"
            break
        if index:
            problem = _reply_problem(turn, outcomes[index - 1], reference, replies)
            if problem:
                break
    if problem:
        case["reasons"].append(problem)
    failures = [o for o in outcomes if o in FAILURES]
    outcome = failures[0] if failures else "invalid_output" if problem else outcomes[-1]
    final_turn, final_result = turns[-1], scored[-1][1]
    within = len(turns) <= budget
    completed = (outcome == "proposed" and within and _text(final_turn.get("proposal_id"))
                 and final_turn.get("confirmed_proposal_id") == final_turn["proposal_id"])
    return {"initial": scored[0][1], "initial_outcome": outcomes[0], "initial_raw": turns[0].get("raw_response"),
            "outcome": outcome, "final": final_result if outcome == "proposed" else None,
            "later_proposals": [r for o, r in scored[1:] if o == "proposed"],
            "asked_before_final": "needs_clarification" in outcomes[:-1],
            "turns_used": len(turns), "within_budget": within, "completed": completed}


def score_output(record, reference: dict, scene: dict, replies: dict, protocol: dict, mode: str,
                 expected_instruction: str | None) -> dict:
    """One case under one condition: the outcome by precedence, then the scores that apply to it."""
    width, height = reference["width"], reference["height"]
    category = reference["category"]
    case = {"case_id": reference["case_id"], "group_id": reference["group_id"], "category": category,
            "expected_status": reference["expected_status"], "reasons": [], "error_code": None}
    call = record.get("call") if record is not None else None
    call_known = isinstance(call, dict) and call.get("outcome") in ("ok", "error")
    if record is None:
        episode = {"initial_outcome": "missing", "outcome": "missing"}
    elif call_known and call["outcome"] == "error":
        case["error_code"] = call.get("error_code")
        episode = {"initial_outcome": "request_error", "outcome": "request_error"}
    elif record.get("scene_sha256") != reference["scene_sha256"]:
        episode = {"initial_outcome": "wrong_scene", "outcome": "wrong_scene"}
    elif expected_instruction is None or record.get("instruction_sha256") != expected_instruction:
        episode = {"initial_outcome": "wrong_instruction", "outcome": "wrong_instruction"}
    elif not call_known:
        case["reasons"].append("missing or unknown call outcome")
        episode = {"initial_outcome": "invalid_output", "outcome": "invalid_output"}
    elif mode == "direct":
        if record.get("turns"):
            case["reasons"].append("direct mode output carries turns")
            episode = {"initial_outcome": "invalid_output", "outcome": "invalid_output"}
        else:
            status = _classify(record.get("result"), width, height, case, record.get("invalid_reason"))
            episode = {"initial": record.get("result"), "initial_outcome": status, "outcome": status,
                       "initial_raw": record.get("raw_response")}
    else:
        episode = _episode(record, reference, replies, case)

    initial, initial_outcome = episode.get("initial"), episode["initial_outcome"]
    case.update(outcome=episode["outcome"], initial_outcome=initial_outcome,
                initial_failed=initial_outcome in FAILURES, final_failed=episode["outcome"] in FAILURES,
                initial_proposed=initial_outcome == "proposed",
                asked=initial_outcome == "needs_clarification",
                ask_acceptable="needs_clarification" in EXPECTED[category],
                location_on_non_proposal=initial_outcome in STATUSES[1:]
                and (initial.get("point") is not None or initial.get("box") is not None))
    if initial_outcome == "proposed":
        placed = _place(initial, reference, protocol)
    else:
        placed = {"point_in_mask": False, "distance_px": None, "box_iou": 0.0 if category == "clear" else None,
                  "box_success": False, "incorrect": False}
    case.update(initial_point_in_mask=placed["point_in_mask"], initial_distance_px=placed["distance_px"],
                initial_box_iou=placed["box_iou"], initial_box_success=placed["box_success"],
                initial_incorrect_proposal=placed["incorrect"])
    case["initial_correct"] = (case["initial_point_in_mask"] if category == "clear"
                               else initial_outcome in EXPECTED[category])

    case["landmarks"] = landmark_score(initial if initial_outcome in STATUSES else None, reference, scene, protocol)
    if case["landmarks"] is not None:
        detected = anchors_detected(reference, scene, protocol)
        case["landmarks"]["anchors_detected"] = sum(detected)
        case.update(landmarks_all_correct=case["landmarks"]["all_correct"],
                    pair_correct=case["landmarks"]["pair_correct"], anchor_not_detected=not all(detected))

    if mode == "clarification":
        final = episode.get("final")
        final_place = _place(final, reference, protocol) if final is not None else None
        later = [_place(r, reference, protocol) for r in episode.get("later_proposals", [])]
        case.update(turns_used=episode.get("turns_used"), within_budget=episode.get("within_budget", False),
                    completed=bool(episode.get("completed")),
                    final_point_in_mask=bool(episode.get("completed") and final_place["point_in_mask"]),
                    incorrect_proposals_after_replies=sum(p["incorrect"] for p in later))
        if category == "missing_reference":
            proposed_any = case["initial_proposed"] or bool(later) or final is not None
            case["final_incorrect_proposal"] = proposed_any
            case["correct_behaviour"] = (case["within_budget"] and case["outcome"] in EXPECTED[category]
                                         and not proposed_any)
        else:
            case["final_incorrect_proposal"] = bool(final_place and final_place["incorrect"])
            case["correct_behaviour"] = case["final_point_in_mask"] and (
                category == "clear" or bool(episode.get("asked_before_final")))
    else:
        case["correct_behaviour"] = case["initial_correct"]

    if initial_outcome == "proposed":
        point, box = initial["point"], initial["box"]
        case.update(box_matches_a_road_user=any(box_iou(box, o["bbox_xyxy"]) >= protocol["box_iou"]
                                                for o in scene["model_input"]["observations"]),
                    box_w=(box[2] - box[0]) / width, box_h=(box[3] - box[1]) / height,
                    box_point_dx=(point[0] - (box[0] + box[2]) / 2) / width,
                    box_point_dy=(point[1] - (box[1] + box[3]) / 2) / height,
                    conversion_mismatch=_conversion_mismatch(episode.get("initial_raw"), initial, width, height))
    return case


# --- summaries ---------------------------------------------------------------------------------------------------

def _spread(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else None


def _near_constant(spreads: list, count: int, protocol: dict) -> bool | None:
    """Flag both spreads below the protocol's threshold (a fraction of the image) once there are 5+ proposals."""
    if count < 5:
        return None
    return all(s < protocol["near_constant_spread"] for s in spreads)


def summarise_condition(cases: list[dict], protocol: dict, mode: str) -> dict:
    of = {c: [x for x in cases if x["category"] == c] for c in CATEGORIES}
    proposed = [c for c in cases if c["initial_proposed"]]
    asked = [c for c in cases if c["asked"]]
    initial = {
        "outcomes": dict(Counter(c["initial_outcome"] for c in cases)),
        "confusion_expected_x_outcome": {e: dict(Counter(c["initial_outcome"] for c in cases
                                                         if c["expected_status"] == e)) for e in STATUSES},
        "proposal_coverage": rate(cases, "initial_proposed", protocol),
        "correct": rate(cases, "initial_correct", protocol),
        "failed_or_invalid": rate(cases, "initial_failed", protocol),
        "incorrect_proposals": rate(cases, "initial_incorrect_proposal", protocol),
        "clarification_precision": rate(asked, "ask_acceptable", protocol) if asked else None,
        "clarification_recall_on_ambiguous": rate(of["ambiguous"], "asked", protocol) if of["ambiguous"] else None,
    }
    clear = of["clear"]
    if clear:
        clear_proposed = [c for c in clear if c["initial_proposed"]]
        misses = [c["initial_distance_px"] for c in clear_proposed if not c["initial_point_in_mask"]]
        initial["clear"] = {
            "point_in_mask": rate(clear, "initial_point_in_mask", protocol),
            "point_in_mask_among_proposals": rate(clear_proposed, "initial_point_in_mask", protocol),
            "box_iou_mean_non_proposals_zero": mean_with_interval(clear, "initial_box_iou", protocol),
            "box_iou_mean_among_proposals": mean_with_interval(clear_proposed, "initial_box_iou", protocol),
            "box_iou_at_threshold": rate(clear, "initial_box_success", protocol),
            "incorrect_proposals": rate(clear, "initial_incorrect_proposal", protocol),
            "missed_points_distance_px_median": statistics.median(misses) if misses else None,
        }
    if of["ambiguous"]:
        initial["ambiguous"] = {
            "asked_for_clarification": rate(of["ambiguous"], "asked", protocol),
            "proposed_without_clarifying": rate(of["ambiguous"], "initial_proposed", protocol),
            "proposal_in_intended_mask": rate(of["ambiguous"], "initial_point_in_mask", protocol),
            "incorrect_proposals": rate(of["ambiguous"], "initial_incorrect_proposal", protocol),
        }
    if of["missing_reference"]:
        initial["missing_reference"] = {
            "asked_or_reported_insufficient_evidence": rate(of["missing_reference"], "initial_correct", protocol),
            "incorrect_proposals": rate(of["missing_reference"], "initial_incorrect_proposal", protocol),
        }
    with_anchors = [c for c in cases if c["landmarks"] is not None]
    if with_anchors:
        pairs = [c for c in with_anchors if c["pair_correct"] is not None]
        initial["landmarks"] = {
            "all_correct": rate(with_anchors, "landmarks_all_correct", protocol),
            "all_correct_among_proposals": rate([c for c in with_anchors if c["initial_proposed"]],
                                                "landmarks_all_correct", protocol),
            "pair_correct": rate(pairs, "pair_correct", protocol) if pairs else None,
            "outputs_naming_landmarks": sum(c["landmarks"]["named"] > 0 for c in with_anchors),
            "invalid_track_references": sum(c["landmarks"]["invalid_references"] for c in with_anchors),
            "extra_landmarks": sum(c["landmarks"]["extra"] for c in with_anchors),
            "cases_with_ordered_roles_not_checked": sum(c["landmarks"]["ordered_roles_not_checked"]
                                                         for c in with_anchors),
            "cases_with_an_anchor_vision_did_not_detect": sum(c["anchor_not_detected"] for c in with_anchors),
        }
    sizes = [_spread([c["box_w"] for c in proposed]), _spread([c["box_h"] for c in proposed])]
    offsets = [_spread([c["box_point_dx"] for c in proposed]), _spread([c["box_point_dy"] for c in proposed])]
    summary = {
        "mode": mode, "cases": len(cases), "groups": len({c["group_id"] for c in cases}),
        "correct_behaviour": rate(cases, "correct_behaviour", protocol),
        "failed_or_invalid": rate(cases, "final_failed", protocol),
        "initial": initial,
        "request_errors": dict(Counter(str(c["error_code"]) for c in cases if c["outcome"] == "request_error")),
        "invalid_output_reasons": dict(Counter(r for c in cases for r in c["reasons"])),
        "locations_on_non_proposals": sum(c["location_on_non_proposal"] for c in cases),
        "diagnostics": {
            "proposed_boxes_matching_a_road_user_box": rate(proposed, "box_matches_a_road_user", protocol)
            if proposed else None,
            "box_size_spread": sizes, "box_sizes_near_constant": _near_constant(sizes, len(proposed), protocol),
            "point_offset_spread": offsets,
            "point_offsets_near_constant": _near_constant(offsets, len(proposed), protocol),
            "coordinate_conversions_checked": sum(c.get("conversion_mismatch") is not None for c in proposed),
            "coordinate_conversion_mismatches": sum(bool(c.get("conversion_mismatch")) for c in proposed),
        },
    }
    if mode == "clarification":
        episodes = [c for c in cases if c["turns_used"] is not None]
        summary["after_intervention"] = {
            "replies": "pre-registered replies from a simulated passenger",
            "final_outcomes": dict(Counter(c["outcome"] for c in cases)),
            "completion_coverage": rate(cases, "completed", protocol),
            "correct": rate(cases, "correct_behaviour", protocol),
            "clear_confirmed_inside_mask": rate(clear, "final_point_in_mask", protocol) if clear else None,
            "ambiguous_confirmed_inside_intended_mask_after_asking": rate(of["ambiguous"], "correct_behaviour",
                                                                          protocol) if of["ambiguous"] else None,
            "missing_reference_without_any_proposal": rate(of["missing_reference"], "correct_behaviour", protocol)
            if of["missing_reference"] else None,
            "incorrect_final_proposals": rate(cases, "final_incorrect_proposal", protocol),
            "incorrect_proposals_after_replies": sum(c["incorrect_proposals_after_replies"] for c in cases),
            "turns_used": mean_with_interval(episodes, "turns_used", protocol),
            "over_budget": sum(1 for c in episodes if not c["within_budget"]),
        }
    return summary


def _call_ok(record) -> bool:
    return record is not None and isinstance(record.get("call"), dict) and record["call"].get("outcome") == "ok"


def _matching_problems(first: dict, second: dict, fields: tuple, *, same_instruction: bool = True) -> list[str]:
    """Why two conditions cannot be compared case by case (empty when they can). Scene and instruction hashes are
    checked on every case both conditions answered; the setup fields once per condition."""
    problems = set()
    checked = ("scene_sha256", "instruction_sha256") if same_instruction else ("scene_sha256",)
    for case in sorted(set(first) & set(second)):
        a, b = first[case], second[case]
        if _call_ok(a) and _call_ok(b):
            for field in checked:
                if not _text(a.get(field)) or a.get(field) != b.get(field):
                    problems.add(f"{field} differs or is missing")
    setups = [next((r["provenance"] for r in side.values() if r), None) for side in (first, second)]
    if all(setups):
        for field in fields:
            if setups[0].get(field) != setups[1].get(field):
                problems.add(f"provenance.{field} differs")
    return sorted(problems)


def paired(a: dict, b: dict, groups: dict, protocol: dict) -> dict:
    """Paired success (case -> bool) for two conditions on the same cases: discordant counts, exact McNemar and a
    group bootstrap interval of the difference (second minus first)."""
    only_first = sum(1 for k in a if a[k] and not b[k])
    only_second = sum(1 for k in a if b[k] and not a[k])
    differences = defaultdict(list)
    for k in a:
        differences[groups[k]].append(float(bool(b[k])) - float(bool(a[k])))
    boot = protocol["bootstrap"]
    return {"cases": len(a), "groups": len(differences), "first_correct": sum(map(bool, a.values())),
            "second_correct": sum(map(bool, b.values())), "only_first_correct": only_first,
            "only_second_correct": only_second,
            "mcnemar_exact_p_assumes_independent_cases": mcnemar_exact(only_first, only_second),
            "difference_second_minus_first": (only_second - only_first) / len(a) if a else None,
            "difference_group_bootstrap95": group_bootstrap(differences, boot["resamples"], boot["seed"])}


def compare_conditions(scored: dict, outputs: dict, protocol: dict) -> list[dict]:
    """Pretrained vs trained within a mode, and direct vs clarification within a model, after checking they match."""
    groups = {c["case_id"]: c["group_id"] for cases in scored.values() for c in cases}

    def by_case(key, field, category=None):
        return {c["case_id"]: c[field] for c in scored[key] if category is None or c["category"] == category}

    plans = []
    for mode in MODES:
        for transcript in TRANSCRIPTS:
            metrics = {"correct_behaviour": ("correct_behaviour", "correct_behaviour", None),
                       "initial_correct": ("initial_correct", "initial_correct", None),
                       "clear_point_in_mask": ("initial_point_in_mask", "initial_point_in_mask", "clear")}
            if mode == "clarification":
                metrics["clear_confirmed_inside_mask"] = ("final_point_in_mask", "final_point_in_mask", "clear")
            plans.append((f"pretrained/{mode}/{transcript}", f"trained/{mode}/{transcript}", MATCH_ACROSS_MODELS,
                          metrics))
    for model in MODELS:
        for transcript in TRANSCRIPTS:
            plans.append((f"{model}/direct/{transcript}", f"{model}/clarification/{transcript}", MATCH_ACROSS_MODES,
                          {"initial_correct": ("initial_correct", "initial_correct", None),
                           "clear_destination_reached": ("initial_point_in_mask", "final_point_in_mask", "clear")}))
    results = []
    for first, second, fields, metrics in plans:
        present = [key for key in (first, second) if key in scored]
        if not present:
            continue
        entry = {"first": first, "second": second}
        if len(present) == 1:
            entry["not_run"] = f"no outputs for {second if first in scored else first}"
        else:
            problems = _matching_problems(outputs[first], outputs[second], fields)
            if problems:
                entry["refused"] = problems
            else:
                for name, (field_a, field_b, category) in metrics.items():
                    a, b = by_case(first, field_a, category), by_case(second, field_b, category)
                    entry[name] = paired(a, b, groups, protocol) if a else None
        results.append(entry)
    return results


# --- voice -------------------------------------------------------------------------------------------------------

def normalise_words(text: str) -> list[str]:
    """NFKC, case folding, apostrophes removed ("don't" -> "dont") and other punctuation replaced by a space."""
    text = unicodedata.normalize("NFKC", text).casefold()
    kept = [" " if unicodedata.category(ch).startswith("P") else ch for ch in text if ch not in APOSTROPHES]
    return "".join(kept).split()


def word_edits(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1]


def voice_summary(rows: list[dict], references: dict, scored: dict, outputs: dict, protocol: dict) -> dict:
    """Word error rate of the ASR transcripts and the same cases scored with human and ASR text. Empty or failed
    recognition stays in the corpus WER as deletions of every word; a WER over recognised cases is given too."""
    per_case = []
    totals = {"edits": 0, "words": 0, "recognised_edits": 0, "recognised_words": 0}
    empty_reference = failed = 0
    critical = Counter()
    for row in rows:
        case, asr = row["case_id"], row.get("asr_transcript")
        critical.update(row.get("critical_errors") or [])
        human = normalise_words(row["human_transcript"])
        recognised = normalise_words(asr) if asr is not None else []
        item = {"case_id": case, "speech": row["speech"], "noise": row["noise"], "wer": None}
        if not human:
            empty_reference += 1
            item["note"] = "empty human transcript"
        elif not recognised:
            failed += 1
            totals["edits"] += len(human)
            totals["words"] += len(human)
            item.update(wer=1.0, note="empty or failed recognition")
        else:
            count = word_edits(human, recognised)
            for prefix in ("", "recognised_"):
                totals[f"{prefix}edits"] += count
                totals[f"{prefix}words"] += len(human)
            item["wer"] = count / len(human)
        per_case.append(item)
    voice_cases = {item["case_id"] for item in per_case}
    groups = {c: references[c]["group_id"] for c in voice_cases}
    downstream = []
    for model in MODELS:
        for mode in MODES:
            human_key, asr_key = f"{model}/{mode}/human", f"{model}/{mode}/asr"
            if human_key not in scored and asr_key not in scored:
                continue
            entry = {"model": model, "mode": mode}
            if human_key not in scored or asr_key not in scored:
                entry["not_run"] = f"no outputs for {asr_key if human_key in scored else human_key}"
                downstream.append(entry)
                continue
            problems = _matching_problems(outputs[human_key], outputs[asr_key], MATCH_ACROSS_TRANSCRIPTS,
                                          same_instruction=False)
            if problems:
                entry["refused"] = problems
                downstream.append(entry)
                continue
            for name, key in (("human", human_key), ("asr", asr_key)):
                cases = [c for c in scored[key] if c["case_id"] in voice_cases]
                entry[name] = {"correct_behaviour": rate(cases, "correct_behaviour", protocol),
                               "proposal_coverage": rate(cases, "initial_proposed", protocol),
                               "incorrect_proposals": rate(cases, "initial_incorrect_proposal", protocol)}
                if mode == "clarification":
                    entry[name]["completion_coverage"] = rate(cases, "completed", protocol)
            entry["paired_correct_behaviour"] = paired(
                {c["case_id"]: c["correct_behaviour"] for c in scored[human_key] if c["case_id"] in voice_cases},
                {c["case_id"]: c["correct_behaviour"] for c in scored[asr_key] if c["case_id"] in voice_cases},
                groups, protocol)
            downstream.append(entry)
    return {"voice_cases": len(rows), "empty_human_transcripts": empty_reference,
            "empty_or_failed_recognition": failed,
            "corpus_wer": totals["edits"] / totals["words"] if totals["words"] else None,
            "corpus_wer_recognised_only": (totals["recognised_edits"] / totals["recognised_words"]
                                           if totals["recognised_words"] else None),
            "speech": dict(Counter(r["speech"] for r in rows)), "noise": dict(Counter(r["noise"] for r in rows)),
            "critical_errors": dict(critical), "per_case": per_case, "downstream": downstream}


# --- leakage and the whole run -----------------------------------------------------------------------------------

def _strings(value):
    """Every string value and key in a JSON value, without recursion."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _parts(text: str) -> set:
    """A string, and for a path each folder and file name with and without its extension."""
    parts = {text}
    if "/" in text or "\\" in text:
        for part in text.replace("\\", "/").split("/"):
            parts.update({part, part.split(".", 1)[0], part.rsplit(".", 1)[0]})
    return parts


def leakage(references: dict, manifests) -> dict:
    """Reserved case IDs and group IDs (whole strings or path parts), scene hashes and recordings (anywhere in a
    string), with the recordings next to them in time, must not appear in the training or selection manifests."""
    recordings = {r["recording_id"] for r in references.values()}
    whole = {"case_id": set(references), "group_id": {r["group_id"] for r in references.values()}}
    inside = {"scene_sha256": {r["scene_sha256"] for r in references.values()},
              "recording_id": recordings | {ADJACENT[r] for r in recordings if r in ADJACENT}}
    hits, hashes = Counter(), {}
    for number, path in enumerate(manifests or [], 1):
        path = Path(path)
        label = f"{number}:{path.name}"
        hashes[label] = text_sha256(path)
        try:
            data = json.loads(_read_bytes(path).decode("utf-8-sig"))
        except (ValueError, RecursionError):
            data = read_jsonl(path)
        for text in _strings(data):
            for field, values in whole.items():
                for part in _parts(text) & values:
                    hits[(label, field, part)] += 1
            for field, values in inside.items():
                for value in values:
                    if value in text:
                        hits[(label, field, value)] += 1
    return {"checked": bool(manifests), "manifests_sha256": hashes,
            "violations": [{"manifest": m, "field": f, "value": v, "occurrences": n}
                           for (m, f, v), n in sorted(hits.items())]}


def baseline(references: dict, protocol: dict) -> dict:
    """A fixed point (fractions of width and height) scored like a proposal on every clear case, the direct-mode
    correct behaviour of a model that always gives the same non-proposal, and the share of the image each mask
    covers."""
    clear = [r for r in references.values() if r["category"] == "clear"]
    fx, fy = protocol["baseline_point"]
    hits = sum(in_mask(r["mask"], [fx * r["width"], fy * r["height"]], r["width"], r["height"]) for r in clear)
    constant = {status: sum(r["category"] != "clear" and status in EXPECTED[r["category"]] for r in references.values())
                for status in STATUSES[1:]}

    def shares(masks):
        values = sorted(float(m.mean()) for m in masks)
        return {"min": values[0], "median": statistics.median(values), "max": values[-1]} if values else None

    return {"point_fraction": [fx, fy], "clear_cases": len(clear), "point_in_mask": hits,
            "rate": hits / len(clear) if clear else None,
            "constant_answer_correct_behaviour": {
                f"always_{status}": {"count": count, "n": len(references), "rate": count / len(references)}
                for status, count in constant.items()},
            "mask_area_share": shares([r["mask"] for r in clear]),
            "intended_mask_area_share": shares([r["intended_mask"] for r in references.values()
                                                if r["category"] == "ambiguous"]),
            "per_case_mask_area_share": {r["case_id"]: float((r["mask"] if r["mask"] is not None
                                                              else r["intended_mask"]).mean())
                                         for r in references.values() if r["category"] != "missing_reference"}}


def headline(conditions: dict, comparisons: list[dict]) -> dict:
    """The numbers a reader looks for first, for the four required conditions and their two comparisons."""
    rows = {}
    for key in REQUIRED_CONDITIONS:
        if key not in conditions:
            continue
        summary, initial = conditions[key], conditions[key]["initial"]
        row = {"correct_behaviour": summary["correct_behaviour"]["rate"],
               "clear_point_in_mask": initial["clear"]["point_in_mask"]["rate"] if "clear" in initial else None,
               "proposal_coverage": initial["proposal_coverage"]["rate"],
               "incorrect_proposals": initial["incorrect_proposals"]["rate"],
               "failed_or_invalid": summary["failed_or_invalid"]["rate"]}
        if "after_intervention" in summary:
            row["completion_coverage"] = summary["after_intervention"]["completion_coverage"]["rate"]
        rows[key] = row
    differences = []
    for first, second in REQUIRED_COMPARISONS:
        entry = next((e for e in comparisons if e["first"] == first and e["second"] == second), None)
        item = {"first": first, "second": second}
        if entry is None or "correct_behaviour" not in entry:
            item["not_available"] = (entry or {}).get("refused") or (entry or {}).get("not_run") or "no outputs"
        else:
            for metric in ("correct_behaviour", "clear_point_in_mask"):
                if entry.get(metric):
                    item[metric] = {"difference_second_minus_first": entry[metric]["difference_second_minus_first"],
                                    "mcnemar_p": entry[metric]["mcnemar_exact_p_assumes_independent_cases"]}
        differences.append(item)
    return {"conditions": rows, "trained_vs_pretrained": differences}


def evaluate(references_path, scenes_dir, outputs_paths, *, replies_path=None, voice_path=None, protocol_path=None,
             train_manifests=None, allow_unreviewed=False, allow_stub_scenes=False) -> tuple[dict, list[dict]]:
    from trackrag.dropoff.scenes import BOUND_BY_MANIFEST, RUN_PINNED

    if isinstance(outputs_paths, (str, Path)):
        outputs_paths = [outputs_paths]
    protocol, protocol_sha = load_protocol(protocol_path)
    scenes = load_scenes(scenes_dir, allow_stub=allow_stub_scenes)
    references, warnings = load_references(references_path, scenes, allow_unreviewed=allow_unreviewed)
    hashes = {"references": text_sha256(references_path),
              "outputs": [{"file": Path(p).name, "sha256": text_sha256(p)} for p in outputs_paths],
              "replies": text_sha256(replies_path) if replies_path else None,
              "voice": text_sha256(voice_path) if voice_path else None, "scenes": sorted(scenes)}
    labels = []
    if protocol_sha is None:
        labels.append("protocol not frozen in advance")
    else:
        for name in ("references", "replies", "voice"):
            _require(protocol[f"{name}_sha256"] == hashes[name],
                     f"The {name} file differs from the one recorded in the frozen protocol")
    if any(s["provenance"].get("detector_kind") == "stub" for s in scenes.values()):
        labels.append(STUB_LABEL)
    unpinned = sum(1 for s in scenes.values() if s["context"].get("image_binding") != BOUND_BY_MANIFEST
                   or s["context"].get("run_binding") != RUN_PINNED)
    if unpinned:
        warnings.append(f"{unpinned} scene(s) made without --frame-manifest and --run-manifest")
    replies = load_replies(replies_path, references)
    voice_rows = load_voice(voice_path, references) if voice_path else None
    voice = {r["case_id"]: r for r in voice_rows} if voice_rows is not None else None
    outputs, refused = load_outputs(outputs_paths, references, set(voice) if voice is not None else None)
    leak = leakage(references, train_manifests)
    if leak["violations"]:
        labels.append("reserved cases appear in training or selection data")
    if not leak["checked"]:
        labels.append("leakage not checked (no training manifests given)")
    if any(w.endswith("reference not fully reviewed") for w in warnings):
        labels.append("some references not fully reviewed")
    accepted = {key for key, records in outputs.items() if any(records.values())}
    missing_conditions = [k for k in REQUIRED_CONDITIONS if k not in accepted]
    if missing_conditions:
        labels.append("required conditions missing: " + ", ".join(missing_conditions))
    refused_required = Counter(r["condition"] for r in refused if r["condition"] in REQUIRED_CONDITIONS)
    if refused_required:
        labels.append("outputs refused in required conditions")
        warnings += [f"{key}: {count} output(s) refused and scored as missing" for key, count in refused_required.items()]

    def expected_instruction(reference, transcript):
        if transcript == "typed":
            return reference["instruction_sha256"]
        text = voice[reference["case_id"]]["human_transcript" if transcript == "human" else "asr_transcript"]
        return instruction_sha256(text, reference["dialogue"]) if isinstance(text, str) and text.strip() else None

    scored, rows = {}, []
    for key in sorted(outputs):
        mode, transcript = key.split("/")[1:]
        cases = [score_output(outputs[key].get(case), ref, scenes[ref["scene_sha256"]], replies, protocol, mode,
                              expected_instruction(ref, transcript))
                 for case, ref in sorted(references.items()) if transcript == "typed" or case in voice]
        scored[key] = cases
        rows += [{"condition": key, **{k: v for k, v in c.items() if not isinstance(v, (dict, list))},
                  "reasons": "; ".join(c["reasons"])} for c in cases]
    conditions = {key: summarise_condition(cases, protocol, key.split("/")[1]) for key, cases in scored.items()}
    comparisons = compare_conditions(scored, outputs, protocol)
    for entry in comparisons:
        if "refused" in entry:
            warnings.append(f"comparison {entry['first']} vs {entry['second']} refused: {'; '.join(entry['refused'])}")
            if (entry["first"], entry["second"]) in REQUIRED_COMPARISONS:
                labels.append(f"required comparison {entry['first']} vs {entry['second']} refused")
    for key, summary in conditions.items():
        if summary["diagnostics"]["coordinate_conversion_mismatches"]:
            warnings.append(f"{key}: {summary['diagnostics']['coordinate_conversion_mismatches']} raw responses do not "
                            "convert to the parsed pixels")
    summary = {
        "version": 1,
        "status": "assessed" if not labels else "not assessed: " + "; ".join(labels),
        "evaluator_sha256": text_sha256(Path(__file__)),
        "protocol": protocol, "protocol_sha256": protocol_sha, "inputs_sha256": hashes,
        "reserved_cases": len(references), "reserved_groups": len({r["group_id"] for r in references.values()}),
        "categories": dict(Counter(r["category"] for r in references.values())),
        "voice_cases": len(voice) if voice is not None else None,
        "required_conditions_missing": missing_conditions,
        "headline": headline(conditions, comparisons),
        "conditions": conditions, "comparisons": comparisons,
        "refused_outputs": refused, "leakage": leak, "warnings": warnings,
        "fixed_point_baseline": baseline(references, protocol),
    }
    if voice_rows is not None:
        summary["voice"] = voice_summary(voice_rows, references, scored, outputs, protocol)
    return summary, rows


def _print_headline(summary: dict, out) -> None:
    def number(value):
        return "-" if value is None else f"{value:.3f}"

    print(f"{summary['status']}: {summary['reserved_cases']} cases, {len(summary['conditions'])} conditions -> {out}")
    rows = summary["headline"]["conditions"]
    if rows:
        print(f"  {'condition':32s} {'correct':>8s} {'clear pt':>9s} {'coverage':>9s} {'wrong pr':>9s} {'failed':>7s}")
        for key, row in rows.items():
            print(f"  {key:32s} {number(row['correct_behaviour']):>8s} {number(row['clear_point_in_mask']):>9s} "
                  f"{number(row['proposal_coverage']):>9s} {number(row['incorrect_proposals']):>9s} "
                  f"{number(row['failed_or_invalid']):>7s}")
    for item in summary["headline"]["trained_vs_pretrained"]:
        if "not_available" in item:
            print(f"  {item['second']} vs {item['first']}: not available ({item['not_available']})")
        elif "correct_behaviour" in item:
            print(f"  {item['second']} minus {item['first']}: correct behaviour "
                  f"{item['correct_behaviour']['difference_second_minus_first']:+.3f} "
                  f"(McNemar p {item['correct_behaviour']['mcnemar_p']:.3f})")
    for warning in summary["warnings"]:
        print(f"  warning: {warning}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.grounding",
                                     description="Score saved drop-off grounding outputs against separate references.")
    parser.add_argument("--references", required=True, help="Reviewed reference cases (JSON Lines)")
    parser.add_argument("--scenes", required=True, help="Folder of scene files (python -m trackrag.dropoff.scenes export)")
    parser.add_argument("--outputs", required=True, nargs="+", help="Frozen model outputs (one or more JSON Lines files)")
    parser.add_argument("--replies", help="Pre-registered passenger replies for clarification mode")
    parser.add_argument("--voice", help="Human and ASR transcripts for the voice cases")
    parser.add_argument("--protocol", help="Frozen protocol; without it the results are labelled not assessed")
    parser.add_argument("--train-manifests", nargs="*", default=[], help="Training and selection manifests")
    parser.add_argument("--allow-unreviewed", action="store_true", help="Development runs only (labelled)")
    parser.add_argument("--allow-stub-scenes", action="store_true", help="Tests only (labelled)")
    parser.add_argument("--out", required=True, help="Summary JSON; its folder must exist and the file must not")
    parser.add_argument("--cases-out", help="Per-case CSV; keep it outside Git if it exposes reference details")
    args = parser.parse_args(argv)
    targets = [Path(args.out)] + ([Path(args.cases_out)] if args.cases_out else [])
    if len(targets) == 2 and targets[0].resolve() == targets[1].resolve():
        print("refused: --out and --cases-out must be different files", file=sys.stderr)
        return 2
    for target in targets:
        if target.exists() or not target.parent.is_dir():
            print(f"refused: {target} already exists or its folder does not", file=sys.stderr)
            return 2
    try:
        summary, rows = evaluate(args.references, args.scenes, args.outputs, replies_path=args.replies,
                                 voice_path=args.voice, protocol_path=args.protocol,
                                 train_manifests=args.train_manifests, allow_unreviewed=args.allow_unreviewed,
                                 allow_stub_scenes=args.allow_stub_scenes)
    except GroundingError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    if args.cases_out:
        with Path(args.cases_out).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
    _print_headline(summary, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
