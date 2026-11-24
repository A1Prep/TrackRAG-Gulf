"""Answer single-frame questions that are directly determined by saved predicted tracks."""
from collections import Counter
import math
import re

from .evidence_time import requested_frame_ids
from .playback_time import requested_playback_times
from .questions import validate_evidence
from .vision.classes import EMT_CLASSES


_LABELS = {label.casefold().replace("_", " "): label for label in EMT_CLASSES}
_MOTOR = set(EMT_CLASSES) - {"Pedestrian", "Cyclist"}
_GROUPS = {"vehicle": _MOTOR, "motor vehicle": _MOTOR, "road user": set(EMT_CLASSES),
           "track": set(EMT_CLASSES), "motorcycle": {"Motorbike"}}
_GROUPS.update({name: {label} for name, label in _LABELS.items()})
_CLOCK = r"(?:\d{1,3}:)?\d{1,3}:[0-5]\d(?:\.\d{1,3})?"
_NUMBER = r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
_TIME = rf"(?:{_CLOCK}|{_NUMBER}\s*(?:milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?))"
_TIME_TOKENS = re.compile(rf"\b{_TIME}\b", re.I)
_FRAME = r"frame(?:\s+(?:id|number))?\s*#?\s*\d+"
_WHEN = re.compile(rf"\b(?:(?:at|around|about|in|on|by)\s+)?(?:{_TIME}(?:\s+ago)?|{_FRAME})\b|"
                   r"\b(?:(?:in|on|at)\s+)?(?:this|the (?:current|paused|selected|displayed))\s+(?:frame|image)\b|"
                   r"\b(?:right now|currently|now)\b", re.I)
_CURRENT = re.compile(r"\b(?:(?:this|the (?:current|paused|selected|displayed)) (?:frame|image)|right now|currently|now)\b", re.I)
_OPENING_FRAME = re.compile(r"\b(?:at the (?:start|beginning) of (?:the|this) (?:clip|video|recording|footage)|"
                            r"in the (?:first|opening) (?:frame|image))\b", re.I)
_CLASS = r"(?:road users?|(?:motor |small motorised |medium |large |emergency )?vehicles?|cars?|buses|bus|pedestrians?|cyclists?|motorbikes?|motorcycles?|tracks?)"
_PERSON = r"(?:people|persons?|anyone|anybody)(?: on foot)?"
_SUBJECT = rf"(?:{_CLASS}|{_PERSON})"
_TRACK = r"(?:track(?:ed object)?(?: id| number)?\s*#?\s*(\d+))"
_COUNT = re.compile(rf"how many (?:tracked |detected )?({_SUBJECT})(?: (?:"
                    r"(?:are|were) (?:there|here|visible|present|detected|tracked)|"
                    r"can you (?:see|count)|do you (?:see|count)|does the tracker (?:show|detect)))?(?: here)?")
_PRESENT = re.compile(rf"(?:(?:are|is|were|was) there|(?:can|do) you (?:see|spot)) (?:any |an? )?({_SUBJECT})(?: (?:here|visible))?")
_OVERVIEW = re.compile(r"(?:what(?:'s| is) (?:happening|going on)|what (?:can|do) you see|"
                       r"what(?:'s| is) in the (?:frame|image|scene))(?: here| in the (?:frame|image|scene))?")
_CLASSES = re.compile(r"(?:what|which) (?:types?|classes|kinds?)(?: of (?:tracked )?(?:road users|vehicles|objects))? "
                      r"(?:are (?:there|visible|present|tracked|detected)|can you see|does the tracker show)")
_IDENTITY = re.compile(rf"(?:what|which) (?:(?:type|kind|class)(?: of (?:vehicle|road user|object))?|vehicle|road user) "
                       rf"(?:is|was) {_TRACK}|(?:what|which) (?:is|was) the (?:class|type) of {_TRACK}")
_SIDE = re.compile(rf"(?:which|what) side(?: of the (?:image|frame|screen))? (?:is|was) {_TRACK}(?: on)?|"
                   rf"where(?: (?:in|on) the (?:image|frame|screen))? (?:is|was) {_TRACK}(?: located)?")


def _parse(text):
    if not isinstance(text, str) or len(requested_frame_ids(text)) > 1:
        return None
    times = requested_playback_times(text)
    time_tokens = _TIME_TOKENS.findall(text)
    if len(times) > 1 or (time_tokens and (len(time_tokens) != 1 or not times)):
        return None
    original = text.casefold().replace("’", "'")
    words = _OPENING_FRAME.sub(" ", original)
    if words != original and _WHEN.search(words):
        return None
    words = " ".join(re.sub(r"[,?!().:;]", " ", _WHEN.sub(" ", words)).split())
    counts, classes, identity, side = (_COUNT.fullmatch(words) or _PRESENT.fullmatch(words),
                                      _CLASSES.fullmatch(words) or _OVERVIEW.fullmatch(words),
                                      _IDENTITY.fullmatch(words), _SIDE.fullmatch(words))
    if not (counts or classes or identity or side):
        return None
    if side and words.startswith("where") and not re.search(r"\b(?:image|frame|screen)\b", original):
        return None
    return words, counts, classes, identity, side


def supports_track_question(text):
    """Whether these words ask only for a supported, single-frame track fact."""
    return _parse(text) is not None


def _frame(question, scope, first_frame):
    words, cutoff = question["question"], question["query_frame"]
    frames = requested_frame_ids(words)
    times = requested_playback_times(words)
    if len(frames) > 1 or len(times) > 1:
        return None
    if scope is not None or times:
        if not times or not isinstance(scope, dict) or scope.get("status") != "available" or len(scope.get("moments", [])) != 1:
            return None
        moment = scope["moments"][0]
        paused, requested, displayed = scope["paused_seconds"], moment["requested_seconds"], moment["frame_seconds"]
        if not all(type(s) in (float, int) and math.isfinite(s) for s in (paused, requested, displayed)):
            return None
        relative, seconds = times[0]
        expected = paused + seconds if relative else seconds
        if (requested < 0 or requested > paused + 1e-6
                or not math.isclose(displayed, requested, rel_tol=0, abs_tol=1e-6)
                or not math.isclose(expected, requested, rel_tol=0, abs_tol=1e-6)):
            return None
        frame = moment["frame_id"]
        if frames and frames != [frame]:
            return None
    elif _OPENING_FRAME.search(words):
        frame = first_frame
    else:
        frame = frames[0] if frames else cutoff
    if _CURRENT.search(words) and frame != cutoff:
        return None
    return frame if type(frame) is int and 0 <= frame <= cutoff else None


def _original_box(box, size):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(type(n) in (float, int) and math.isfinite(n) for n in box):
        return False
    left, top, right, bottom = box
    return 0 <= left < right and 0 <= top < bottom and (size is None or right <= size[0] and bottom <= size[1])


def answer_from_tracks(question, evidence_rows, tracks, *, playback_scope=None, frame_size=None, first_frame=None):
    """Return an ordinary cited answer or None when the exact displayed frame cannot establish it."""
    try:
        validate_evidence(question, evidence_rows)
        frame = _frame(question, playback_scope, first_frame)
    except (KeyError, TypeError, ValueError):
        return None
    if frame is None:
        return None
    if frame_size is not None and (not isinstance(frame_size, (tuple, list)) or len(frame_size) != 2
            or not all(type(n) is int and n > 0 for n in frame_size)):
        return None
    parsed = _parse(question["question"])
    if parsed is None:
        return None
    words, counts, classes, identity, side = parsed
    if side and frame_size is None:
        return None
    current = {}
    for row in tracks:
        if (row.get("recording_id"), row.get("frame_id")) != (question["recording_id"], frame):
            continue
        tid, label, box = row.get("track_id"), row.get("class_name"), row.get("bbox_xyxy")
        if (row.get("source") != "predicted" or type(tid) is not int or tid < 0 or tid in current
                or label not in EMT_CLASSES or not _original_box(box, frame_size)):
            return None
        current[tid] = (label, box)
    highlighted, scene = {}, None
    for record in evidence_rows:
        interval, cid = record["interval"], record["evidence_id"]
        if interval.get("source") != "predicted" or not isinstance(cid, str) or not cid:
            return None
        for image in record["images"]:
            if image["frame_id"] != frame:
                continue
            tid = interval["track_id"]
            if tid is None:
                scene = scene or cid
            elif tid in current and list(current[tid][1]) == list(image["bbox_xyxy"]):
                highlighted.setdefault(tid, cid)
            else:
                return None
    wanted = set(current)
    if counts:
        noun = counts[1]
        if noun in {"people", "person", "persons", "anyone", "anybody"}:
            wanted = {tid for tid, (label, _) in current.items() if label in {"Pedestrian", "Cyclist"}}
            people = Counter(current[tid][0] for tid in wanted)
            if len(people) < 2 and scene is None:
                return None
            groups = [f"{people[label]} {label.lower()}{'' if people[label] == 1 else 's'}"
                      for label in ("Pedestrian", "Cyclist")]
            answer = f"In frame {frame}, the tracker recorded {' and '.join(groups)}. It does not count people inside vehicles."
        else:
            singular = "bus" if noun in {"bus", "buses"} else noun.removesuffix("s")
            if noun.endswith(" on foot"):
                singular = "pedestrian"
            wanted = {tid for tid, (label, _) in current.items() if label in _GROUPS[singular]}
            plural = "buses" if singular == "bus" else singular + "s"
            answer = f"The tracker detected {len(wanted)} {singular if len(wanted) == 1 else plural} in frame {frame}."
    elif classes:
        if "vehicles" in words:
            wanted = {tid for tid, (label, _) in current.items() if label in _MOTOR}
        inventory = Counter(current[tid][0].lower().replace("_", " ") for tid in wanted)
        names = ", ".join(f"{label} ({n})" for label, n in sorted(inventory.items()))
        answer = (f"In frame {frame}, the tracker reported {names}." if names else f"The tracker recorded no matching road users in frame {frame}.")
    else:
        tid = int(next(g for g in (identity or side).groups() if g is not None))
        if tid not in current:
            return None
        wanted = {tid}
        label, (left, _, right, _) = current[tid]
        if identity:
            answer = f"In frame {frame}, the tracker labels track {tid} as {label.lower().replace('_',' ')}."
        else:
            midpoint = (left + right) / (2 * frame_size[0])
            position = "on the left side" if midpoint < 1/3 else "on the right side" if midpoint > 2/3 else "near the centre"
            answer = f"In frame {frame}, the centre of track {tid}'s box is {position} of the image."
    if scene:
        citations = [scene]
    elif wanted and wanted <= highlighted.keys():
        citations = list(dict.fromkeys(highlighted[tid] for tid in sorted(wanted)))
    else:
        return None
    return {"answer_status": "answered", "answer": answer, "cited_evidence_ids": citations}
