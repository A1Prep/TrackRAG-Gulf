"""Select original scene endpoints and observed histories for footage questions.

Untimed object searches keep Task 4's lexical ranking. Time references select
from the same index and source, up to the question's inclusive cutoff. The
normal reviewer still exports the pixels and Task 3 still answers the question.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
from functools import lru_cache
import re

from .evidence_time import referenced_track_ids, requested_frame_ids
from .retrieval import _tokens, _validate_question, build_index, rank_evidence as rank_windows


def _matches(pattern, text):
    return re.search(pattern, text, re.I) is not None


_SMALL_NUMBERS = {word: value for value, word in enumerate(
    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"), 1)}


def question_scope(text):
    """Identify evidence coverage, without assigning an answer or an object ID."""
    words = text.casefold()
    entering = _matches(r"\b(?:enter(?:s|ed|ing)?|came in(?:to)?|comes? into (?:view|frame))\b", words)
    exiting = _matches(r"\b(?:exit(?:s|ed|ing)?(?!\s+(?:sign|ramp|lane|road))|(?:mov(?:e[ds]?|ing)|went|goes?) out|leaves? the (?:frame|scene))\b", words)
    opening = _matches(r"\b(?:opening|initial(?:ly)?|beginning|at the start|first (?:image|frame|second)|early)\b", words)
    latest = _matches(r"\b(?:current(?:ly)?|now|latest|recent(?:ly)?|last (?:frame|image)|(?:towards?|near|by|at) the end)\b", words)
    history = _matches(r"\b(?:earlier observations?|recent observations?|as (?:the )?(?:clip|recording|video) progresses|over time|"
                       r"(?:larger|smaller|bigger|closer|further) (?:than|as|over)|emerges?|still visible|during|across the image)\b", words)
    scene = _matches(r"\b(?:overpass|bridge|sky|lighting|daylight|night(?:time)?|street name|road name|(?:road|exit|traffic) sign)\b", words)
    exact = requested_frame_ids(words)
    duration = re.search(r"\b(?:opening|first|initial)\s+(?:(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?seconds?\b", words)
    seconds = None
    if duration:
        value = duration.group(1) or "one"
        seconds = int(value) if value.isdigit() else _SMALL_NUMBERS[value]
        opening = history = True
    if not (entering or exiting or opening or latest or history or scene or exact):
        return None
    return {"opening": opening, "latest": latest, "history": history, "opening_seconds": seconds,
            "boundary": "enter" if entering else "exit" if exiting else None,
            "scene": scene, "exact": exact}


def _available(recording, cutoff):
    known = getattr(recording, "frame_ids", None)
    if known is not None:
        return known[:bisect_right(known, cutoff)]
    ids = []
    for frame in recording.iter_frames():
        if frame["frame_id"] > cutoff:
            break
        ids.append(frame["frame_id"])
        if frame["frame_id"] == cutoff:
            break
    return ids


_CLASSES = {
    "suv": {"car", "medium_vehicle"}, "sedan": {"car"}, "pickup": {"car", "medium_vehicle"},
    "truck": {"large_vehicle", "medium_vehicle", "truck"}, "lorry": {"large_vehicle", "truck"},
    "van": {"medium_vehicle", "car", "van"}, "coach": {"bus"}, "motorcycle": {"motorbike", "motorcycle"},
    "bicycle": {"bicycle", "cyclist"}, "person": {"pedestrian", "person"},
}


def _object_score(rows, text, dimensions, frame, direction):
    words = set(_tokens(text))
    label = Counter(row["class_name"].casefold() for row in rows).most_common(1)[0][0]
    named = set(referenced_track_ids(text))
    if named and rows[0]["track_id"] not in named:
        return 0.0
    wanted = set().union(*(_CLASSES.get(word, {word}) for word in words))
    matched = label in wanted or ("vehicle" in words and label not in {"person", "pedestrian", "cyclist"})
    if not (matched or named):
        return 0.0
    nearest = min(rows, key=lambda row: abs(row["frame_id"] - frame))
    width, height = dimensions(nearest["frame_id"])
    l, t, r, b = nearest["bbox_xyxy"]
    center = (l + r) / (2 * width)
    side = 1 - center if "left" in words else center if "right" in words else 0.5
    size = (r - l) * (b - t) / (width * height)
    # A bounded area term helps distinguish a prominent foreground object.
    prominent = bool(words & {"prominent", "large", "largest", "big", "bigger", "biggest"})
    score = 0.5 + 0.15 * side + (0.15 if prominent else 0.04) * size**0.5
    if direction:
        # Real proximity to the named border and movement into/out of the image
        # identify a useful history. They alone cannot prove a crossing time.
        outer, inner = (rows[0], rows[-1]) if direction == "enter" else (rows[-1], rows[0])
        def margins(row):
            w, h = dimensions(row["frame_id"])
            x1, y1, x2, y2 = row["bbox_xyxy"]
            return {"left": x1/w, "right": 1-x2/w, "top": y1/h, "bottom": 1-y2/h}
        border, interior = margins(outer), margins(inner)
        sides = (words & border.keys()) or border.keys()
        if any(border[s] <= 0.05 and interior[s] > border[s] + 0.02 for s in sides):
            score += 0.28
    return min(score, 1.0)


def rank_evidence(question, index, k=3):
    if type(k) is not int or not 0 <= k <= 3:
        raise ValueError("k must be an integer from 0 through 3")
    cutoff = _validate_question(question, index)
    text = question["question"]
    scope = question_scope(text)
    if scope is None:
        return rank_windows(question, index, k=k)
    if not k or index.source is None:
        return []
    allowed = _available(index.recording, cutoff)
    if (not allowed or cutoff not in allowed or scope["opening_seconds"] == 0
            or any(f not in allowed for f in scope["exact"])):
        return []
    first = allowed[0]
    horizon = cutoff
    if scope["opening_seconds"] is not None:
        start_time = index.recording.get_frame(first)["timestamp_s"]
        # Use real timestamps and a half-open time interval, even with skipped
        # frame IDs and a recording whose clock starts above zero.
        stop = start_time + scope["opening_seconds"]
        end_index = bisect_left(allowed, stop, key=lambda f: index.recording.get_frame(f)["timestamp_s"])
        horizon = allowed[max(0, end_index-1)]
    endpoints = set(scope["exact"])
    if scope["opening"]:
        endpoints.add(first)
    if scope["latest"] or not endpoints or scope["opening_seconds"] is not None:
        endpoints.add(horizon)
    if any(fid > horizon for fid in endpoints):
        return []
    ordered = sorted(endpoints)
    focus = ordered[-1]
    if scope["history"] and scope["opening"] and not scope["exact"]:
        # The query's cutoff bounds an opening-period movement question.
        focus = horizon

    @lru_cache(maxsize=None)
    def dimensions(fid):
        frame = index.recording.get_frame(fid)
        return frame["width"], frame["height"]

    candidates = []
    for key, history in index.histories.items():
        rows = [r for r in history if r["frame_id"] <= horizon]
        if not rows:
            continue
        if scope["boundary"]:
            rows_for_score = rows
        elif scope["history"] or len(ordered) > 1:
            rows_for_score = [r for r in rows if r["frame_id"] <= focus]
            if not rows_for_score:
                continue
            if scope["latest"] and not scope["opening"]:
                lower = focus - index.config.window_frames + 1
            else:
                lower = first if len(ordered) == 1 else ordered[0]
            rows_for_score = [r for r in rows_for_score if r["frame_id"] >= lower]
        else:
            rows_for_score = [r for r in rows if r["frame_id"] == focus]
        if not rows_for_score:
            continue
        score = _object_score(rows_for_score, text, dimensions, focus, scope["boundary"])
        if score > 0:
            # Prefer a track observed at both requested endpoints for comparisons.
            present = {r["frame_id"] for r in rows_for_score}
            coverage = len(present & endpoints) / len(endpoints)
            candidates.append((score, coverage, key[2], rows_for_score[0]["frame_id"], rows_for_score[-1]["frame_id"]))
    candidates.sort(key=lambda c: (-c[0], c[2]) if scope["boundary"] else (-c[1], -c[0], c[2]))

    intervals = []
    def add(track, start, end, score):
        if (len(intervals) < k and all((i["track_id"], i["start_frame"], i["end_frame"]) != (track, start, end)
                                      for i in intervals)):
            intervals.append({"recording_id": question["recording_id"], "source": index.source,
                              "track_id": track, "start_frame": start, "end_frame": end,
                              "rank": len(intervals)+1, "score": score})

    if candidates and (not scope["scene"] or scope["boundary"]):
        score, _, track, start, end = candidates[0]
        add(track, start, end, score)
    if scope["boundary"] and candidates:
        _, _, _, start, end = candidates[0]
        i = bisect_left(allowed, start)-1 if scope["boundary"] == "enter" else bisect_right(allowed, end)
        if 0 <= i < len(allowed) and allowed[i] <= horizon:
            add(None, allowed[i], allowed[i], 1.0)
        context = first if scope["boundary"] == "enter" else horizon
        add(None, context, context, 1.0)
    else:
        for f in ordered:
            add(None, f, f, 1.0)
        for score, _, track, start, end in candidates:
            add(track, start, end, score)
    return intervals
