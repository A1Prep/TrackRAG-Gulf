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


def question_scope(text, *, current_if_untimed=False):
    """Identify evidence coverage, without assigning an answer or an object ID."""
    words = text.casefold()
    entering = _matches(r"\b(?:enter(?:s|ed|ing)?|came in(?:to)?|comes? into (?:view|frame))\b", words)
    exiting = _matches(r"\b(?:exit(?:s|ed|ing)?(?!\s+(?:sign|ramp|lane|road))|(?:mov(?:e[ds]?|ing)|went|goes?) out|leaves? the (?:frame|scene))\b", words)
    opening = _matches(r"\b(?:opening|initial(?:ly)?|beginning|at the start|first (?:image|frame|second)|early)\b", words)
    latest = _matches(r"\b(?:current(?:ly)?|now|latest|recent(?:ly)?|last (?:frame|image)|"
                      r"(?:this|the (?:selected|displayed|paused)) (?:frame|image)|(?:towards?|near|by|at) the end)\b", words)
    history = _matches(r"\b(?:earlier observations?|recent observations?|as (?:the )?(?:clip|recording|video) progresses|over time|"
                       r"(?:larger|smaller|bigger|closer|further) (?:than|as|over)|emerges?|still visible|during|across the image)\b", words)
    if opening and _matches(r"\b(?:mov(?:e[ds]?|ing)|motion|movement|shift(?:s|ed|ing)?|direction)\b", words):
        history = True
    scene = _matches(r"\b(?:overpass|bridge|sky|lighting|daylight|night(?:time)?|street name|road name|(?:road|exit|traffic) sign)\b", words)
    exact = requested_frame_ids(words)
    duration = re.search(r"\b(?:opening|first|initial)\s+(?:(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?seconds?\b", words)
    seconds = None
    if duration:
        value = duration.group(1) or "one"
        seconds = int(value) if value.isdigit() else _SMALL_NUMBERS[value]
        opening = history = True
    if not (entering or exiting or opening or latest or history or scene or exact):
        if not current_if_untimed:
            return None
        latest = True
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


def _untimed_visible_object(text):
    objects = set(_tokens(text)) & (_CLASSES.keys() | {"bus", "car", "vehicle", "cyclist", "pedestrian"})
    return bool(objects) and not _matches(r"\b(?:why|will|would|later|next(?!\s+to\b)|future|tomorrow|yesterday|earlier|previous(?:ly)?|"
                               r"before|after|frames?|seconds?|minutes?|timestamps?|prefer(?:s|red)?|favou?rite|"
                               r"think(?:s|ing)?|imagined?)\b|\d+:\d{2}|\d\s*(?:ms|s|secs?)\b", text)


def _current_object_detail(text):
    """Visible appearance, cargo and attachments can have no lexical hit in the track index."""
    return _untimed_visible_object(text) and _matches(r"\b(?:colou?rs?|shape|appearance|patterns?|markings?|lettering|logos?|stripes?|"
                    r"attachments?|mounted|attached|structures?|looks? like|carrying|hauling|cargo|load on)\b", text)


def current_spatial_relation(text):
    # Buildings, kerbs and gates usually have no track ID. Include the full current image
    # for their relationship to a road user, even if a lexical vehicle match exists.
    return _untimed_visible_object(text) and _matches(
        r"\b(?:near|next to|beside|opposite|behind|under|in front of|to the (?:left|right) of)\s+"
        r"(?:(?:the|a|an|that|this|those|another)\s+)?"
        r"(?:[\w'-]+\s+){0,3}(?:buildings?|shops?|bridges?|overpasses?|gates?|kerbs?|curbs?|"
        r"sidewalks?|junctions?|signs?|barriers?|trees?|houses?|walls?|entrances?|"
        r"cars?|trucks?|vans?|buses|bus|pickups?|lanes?|pedestrians?)\b", text)


def matches_vehicle_class(noun, label):
    label = label.casefold().replace(" ", "_")
    if noun == "vehicle":
        return label in {"car", "truck", "bus", "van", "medium_vehicle", "large_vehicle"}
    return label in _CLASSES.get(noun, {noun}) or label == noun


def _object_score(rows, text, dimensions, frame, direction):
    words = set(_tokens(text))
    labels = Counter(row["class_name"].casefold() for row in rows)
    label = labels.most_common(1)[0][0]
    named = set(referenced_track_ids(text))
    if named and rows[0]["track_id"] not in named:
        return 0.0
    wanted = set().union(*(_CLASSES.get(word, {word}) for word in words))
    def class_matches(value):
        return value in wanted or ("vehicle" in words and value not in {"person", "pedestrian", "cyclist"})
    matched = class_matches(label)
    support = 1.0
    if direction and not matched:
        # A persistent track can change its per-frame class. Two corroborating
        # detections warrant showing its pixels as an alternative. Preserve every
        # original class and the whole observed history for the visual answerer.
        count = sum(n for value,n in labels.items() if class_matches(value))
        matched = count >= 2
        support = count / len(rows)
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
    return min(score, 1.0) * (support if not named else 1.0)


def rank_evidence(question, index, k=3):
    if type(k) is not int or not 0 <= k <= 3:
        raise ValueError("k must be an integer from 0 through 3")
    cutoff = _validate_question(question, index)
    text = question["question"]
    scope = question_scope(text)
    cargo = _untimed_visible_object(text) and _matches(r"\b(?:carrying|hauling|cargo|load on)\b", text)
    if scope is None and (current_spatial_relation(text) or cargo):
        # A load or an untracked landmark needs the visible scene at the paused frame.
        scope = question_scope(text, current_if_untimed=True)
    if scope is None:
        windows = rank_windows(question, index, k=k)
        if windows or not _current_object_detail(text):
            return windows
        scope = question_scope(text, current_if_untimed=True)
    # The observation source must be established by this cutoff. A detector
    # result that first appears later cannot make an earlier scene retrievable.
    if not k or index.source is None or not any(rows[0]["frame_id"] <= cutoff for rows in index.histories.values()):
        return []
    allowed = _available(index.recording, cutoff)
    if (not allowed or cutoff not in allowed or scope["opening_seconds"] == 0
            or any(f not in allowed for f in scope["exact"])):
        return []
    first = allowed[0]
    horizon = cutoff
    opening_middle = None
    if scope["opening_seconds"] is not None:
        start_time = index.recording.get_frame(first)["timestamp_s"]
        # Use real timestamps and a half-open time interval, even with skipped
        # frame IDs and a recording whose clock starts above zero.
        stop = start_time + scope["opening_seconds"]
        end_index = bisect_left(allowed, stop, key=lambda f: index.recording.get_frame(f)["timestamp_s"])
        horizon = allowed[max(0, end_index-1)]
    elif scope["opening"] and scope["history"] and not (scope["exact"] or scope["latest"] or scope["boundary"]):
        # A vague opening motion request uses the normal window of original
        # frame IDs. Keep real snapshots even when the detector missed its subject.
        end_index = bisect_right(allowed, first + index.config.window_frames - 1) - 1
        horizon = allowed[end_index]
        opening_middle = allowed[end_index // 2]
    endpoints = set(scope["exact"])
    if scope["opening"]:
        endpoints.add(first)
    if scope["latest"] or not endpoints or scope["opening_seconds"] is not None or opening_middle is not None:
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
        if opening_middle is not None and not candidates and k >= 3:
            ordered = sorted(set(ordered) | {opening_middle})
        for f in ordered:
            add(None, f, f, 1.0)
        for score, _, track, start, end in candidates:
            add(track, start, end, score)
    return intervals
