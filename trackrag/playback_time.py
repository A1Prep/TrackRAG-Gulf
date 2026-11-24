"""Resolve passenger-facing playback times against the available original frames."""
from bisect import bisect_right
import math
import re


_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_NUMBER = r"(?:\d{1,6}(?:\.\d{1,3})?|" + "|".join(_NUMBERS) + r")"
_UNIT = r"(?:milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?)"
_VALUE = rf"(?P<number>{_NUMBER})[\s-]*(?P<unit>{_UNIT})\b"
_CLOCK = re.compile(r"(?<![\d:\-])(?:(\d{1,3}):)?(\d{1,3}):([0-5]\d(?:\.\d{1,3})?)(?![\d:])")
_PAIRED_AFTER = re.compile(r"\s*[,;(–—-]?\s*(?:at\s+)?frame\s*#?\s*(\d+)\b", re.I)
_PAIRED_BEFORE = re.compile(r"\bframe\s*#?\s*(\d+)\s*[,;(–—-]?\s*(?:at\s+)?$", re.I)
_ABSOLUTE = re.compile(rf"\b(?:at|around|about|by|timestamp(?:ed)?(?: at)?|timecode)\s+(?:the\s+)?{_VALUE}|"
                       rf"\b(?P<into>{_NUMBER})\s*(?P<into_unit>{_UNIT})\s+(?:in(?:to)?\s+(?:the|this)\s+(?:clip|video|recording|footage|drive)|in\b)", re.I)
_RELATIVE = re.compile(rf"\b{_VALUE}\s+(?P<direction>ago|before\s+(?:now|this frame|the pause)|from\s+now|after\s+(?:now|this frame|the pause))\b", re.I)
_WINDOW = re.compile(rf"\b(?:last|past|previous)\s+{_VALUE}", re.I)
_LIST = re.compile(rf"\b(?:at|from|between)\s+({_NUMBER}(?:\s*{_UNIT})?"
                   rf"(?:\s*(?:,\s*(?:and\s+)?|\b(?:to|through|and)\b)\s*{_NUMBER}(?:\s*{_UNIT})?)+)\b", re.I)
_LIST_ITEM = re.compile(rf"({_NUMBER})\s*({_UNIT})?\b", re.I)
_FUTURE_INTERVAL = re.compile(r"\b(?:next\s+(?:frame|few\s+seconds?|\d+\s+seconds?)|after\s+(?:the paused frame|this point))\b", re.I)
_FUTURE_QUESTION = re.compile(r"\b(?:(?:what|which|where|when|why|how|who)\s+will\b|will\s+(?:i|we|the (?:driver|vehicle|car|truck|bus))\b)", re.I)


def _seconds(number, unit):
    value = float(_NUMBERS[number]) if number in _NUMBERS else float(number)
    if unit.startswith(("mill", "ms")):
        return value / 1000
    return 60 * value if unit.startswith("min") else value


def requested_playback_times(text):
    """Return (from paused frame, seconds) pairs, keeping durations separate from timestamps."""
    text = text.casefold()
    result = []
    for match in _CLOCK.finditer(text):
        hours, minutes, seconds = match.groups()
        if hours and int(minutes) >= 60:
            continue
        result.append((False, 3600 * int(hours or 0) + 60 * int(minutes) + float(seconds)))
    for match in _ABSOLUTE.finditer(text):
        if not re.match(r"\s+(?:ago|before|after|from now)\b", text[match.end():]):
            result.append((False, _seconds(match["number"] or match["into"], match["unit"] or match["into_unit"])))
    for match in _RELATIVE.finditer(text):
        duration = _seconds(match["number"], match["unit"])
        result.append((True, -duration if match["direction"].startswith(("ago", "before")) else duration))
    for match in _WINDOW.finditer(text):
        result.extend(((True, -_seconds(match["number"], match["unit"])), (True, 0.0)))
    for match in _LIST.finditer(text):
        items = _LIST_ITEM.findall(match[1])
        declared = [unit for _, unit in items if unit]
        if declared:
            result.extend((False, _seconds(number, unit or declared[-1])) for number, unit in items)
    return list(dict.fromkeys(result))


def inconsistent_frame_time(text, scope):
    """Check explicit adjacent clock/frame pairs for the instants named in the question."""
    expected = {moment["requested_seconds"]: moment["frame_id"] for moment in scope.get("moments", [])}
    if not expected:
        return False
    spans = []
    for clock in _CLOCK.finditer(text):
        h, m, s = clock.groups()
        spans.append((clock.start(),clock.end(),3600*int(h or 0)+60*int(m)+float(s)))
    for instant in _ABSOLUTE.finditer(text.casefold()):
        at = _seconds(instant["number"] or instant["into"], instant["unit"] or instant["into_unit"])
        spans.append((instant.start(),instant.end(),at))
    for start,end,at in spans:
        pair = _PAIRED_AFTER.match(text[end:]) or _PAIRED_BEFORE.search(text[:start])
        if pair and int(pair[1]) != expected.get(at):
            return True
    return False


def available_frame_ids(recording, cutoff):
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


def playback_scope(text, recording, cutoff):
    """Resolve exact source timestamps, refusing absent samples and times beyond the pause."""
    requested = requested_playback_times(text)
    seeking = requested and re.search(r"\bwill\s+(?:I|we)\s+see\s+at\b", text, re.I)
    future = _FUTURE_INTERVAL.search(text) or (_FUTURE_QUESTION.search(text) and not seeking)
    if future:
        return {"status": "outside_review_window", "moments": []}
    if not requested:
        return None
    ids = available_frame_ids(recording, cutoff)
    if not ids or ids[-1] != cutoff:
        return {"status": "unavailable", "moments": []}
    def stamp(fid):
        return recording.get_frame(fid)["timestamp_s"]
    opening = stamp(ids[0])
    paused = stamp(cutoff) - opening
    moments = []
    for relative, seconds in requested:
        at = paused + seconds if relative else seconds
        if at < -1e-7 or at > paused + 1e-7:
            outside = {"status": "outside_review_window", "paused_seconds": round(paused, 6), "moments": []}
            if at > paused and all(not from_pause and time >= 0 for from_pause, time in requested):
                outside["later_absolute_seconds"] = max(time for _, time in requested)
            return outside
        index = bisect_right(ids, opening + at + 1e-7, key=stamp) - 1
        fid = ids[max(index, 0)]
        if not math.isclose(stamp(fid) - opening, at, rel_tol=0, abs_tol=1e-7):
            return {"status": "unavailable", "paused_seconds": round(paused, 6), "moments": []}
        moments.append({"requested_seconds": round(at, 6), "frame_id": fid,
                        "frame_seconds": round(stamp(fid) - opening, 6)})
    scope = {"status": "available", "paused_seconds": round(paused, 6), "moments": moments}
    if inconsistent_frame_time(text, scope):
        return {"status": "conflicting_frame_and_time", "paused_seconds": round(paused, 6), "moments": []}
    return scope
