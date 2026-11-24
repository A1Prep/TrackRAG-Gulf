"""Cite the actual first and last detector boxes seen by the footage playhead."""
import re
from collections import defaultdict

from .questions import validate_evidence


_TRACK = r"(?:(?:track|tracked (?:vehicle|object))(?: id| number)?\s*#?\s*(\d{1,9})|the (bus|car|motorbike|pedestrian|cyclist))"
_PATTERNS = {
    "first": [rf"when (?:was|is) {_TRACK} first (?:seen|detected|tracked)",
              rf"when (?:did|does) {_TRACK} (?:first appear|enter(?: the (?:frame|scene|picture))?)"],
    "last": [rf"when (?:was|is) {_TRACK} last (?:seen|detected|tracked)",
             rf"when (?:did|does) {_TRACK} (?:disappear|(?:exit|leave)(?: the (?:frame|scene|picture))?)"],
    "motion": [rf"how (?:has|did) {_TRACK} (?:moved?|shifted?)(?: across (?:the|this) (?:image|frame))?",
               rf"which (?:way|direction) did {_TRACK} move(?: (?:in|across) (?:the|this) (?:image|frame))?"],
}


def intent(text):
    if not isinstance(text, str):
        return None
    words = " ".join(text.strip().rstrip("?.!").lower().split())
    for name, patterns in _PATTERNS.items():
        for pattern in patterns:
            match = re.fullmatch(pattern, words)
            if match:
                return name, int(match[1]) if match[1] is not None else match[2]
    return None


def timeline_context(clock, tracks):
    """Make each frame's suggestions using only what was observed by that frame."""
    by_frame = defaultdict(list)
    for t in tracks:
        by_frame[t["frame_id"]].append(t)
    last, first, result = {}, {}, {}
    recent = None
    previous = {}
    for frame in clock:
        fid = frame["id"]
        rows = by_frame[fid]
        current = {t["track_id"]:t for t in rows}
        def area(row):
            x0, y0, x1, y1 = row["bbox_xyxy"]
            return (x1-x0)*(y1-y0)
        ended = [r for tid,r in previous.items() if tid not in current]
        if ended:
            recent = max(ended, key=area)
        if recent and recent["track_id"] in current:
            recent = None
        for tid, row in current.items():
            first.setdefault(tid, row)
            last[tid] = row["frame_id"]
        def subject(row):
            tid = row["track_id"]
            return {"id":tid, "class_name":row["class_name"], "first_frame":first[tid]["frame_id"],
                    "first_box":list(first[tid]["bbox_xyxy"]),
                    "last_frame":last[tid], "box":list(row["bbox_xyxy"])}
        lead = max(rows,key=area) if rows else None
        result[fid] = {"count":len(rows), "current":subject(lead) if lead else None,
                       "recent":subject(recent) if recent else None}
        previous = current
    return result


def allowed_rows(question, tracks, track_id):
    return sorted((t for t in tracks if t["source"] == "predicted" and
                   t["recording_id"] == question["recording_id"] and t["track_id"] == track_id and
                   t["frame_id"] <= question["query_frame"]), key=lambda t: t["frame_id"])


def _ids(question, target, tracks):
    if isinstance(target, int):
        return [target]
    return sorted({r["track_id"] for r in tracks if r["frame_id"] <= question["query_frame"] and
                   r["recording_id"] == question["recording_id"] and r["source"] == "predicted" and
                   r["class_name"].casefold() == target})


def intervals(question, index, k=3):
    parsed = intent(question["question"])
    if not parsed or not k:
        return []
    kind, target = parsed
    if index.recording.recording_id != question["recording_id"]:
        return []
    result = []
    for tid in _ids(question, target, (r for series in index.histories.values() for r in series))[:k]:
        rows = allowed_rows(question, index.histories.get((question["recording_id"], "predicted", tid), []), tid)
        if rows:
            first, last = rows[0]["frame_id"], rows[-1]["frame_id"]
            result.append({"recording_id":question["recording_id"],"source":"predicted","track_id":tid,
                           "start_frame":last if kind == "last" else first,"end_frame":first if kind == "first" else last,
                           "rank":len(result)+1,"score":1.0})
    return result


def answer(question, tracks, evidence, recording):
    parsed = intent(question["question"])
    if not parsed:
        return None
    kind, target = parsed
    ids = _ids(question,target,tracks)
    if len(ids) > 1:
        choices = ", ".join("track "+str(t) for t in ids[:3])
        return {"answer_status":"insufficient_evidence","answer":f"Several tracks match that description. Which did you mean? The frames below show {choices}.","cited_evidence_ids":[]}
    if not ids:
        return None
    tid = ids[0]
    rows = allowed_rows(question, tracks, tid)
    if not rows or not evidence:
        return None
    validate_evidence(question, evidence)
    ends = [rows[0]] if kind == "first" else [rows[-1]] if kind == "last" else [rows[0], rows[-1]]
    citations = []
    for row in ends:
        found = next((e["evidence_id"] for e in evidence if e["interval"]["source"] == "predicted"
                      and e["interval"]["track_id"] == tid and any(
                          im["frame_id"] == row["frame_id"] and list(im["bbox_xyxy"]) == list(row["bbox_xyxy"])
                          for im in e["images"])), None)
        if not found:
            return None
        if found not in citations:
            citations.append(found)
    first_frame = recording.frame_ids[0]
    origin = recording.get_frame(first_frame)["timestamp_s"]
    def moment(fid):
        seconds = recording.get_frame(fid)["timestamp_s"] - origin
        return f"{int(seconds//60)}:{seconds%60:04.1f} (frame {fid})"
    a,b = rows[0], rows[-1]
    names = {r["class_name"].lower().replace("_", " ") for r in rows}
    subject = f"The tracked {next(iter(names))} (track {tid})" if len(names) == 1 else f"Track {tid}"
    if len(names) > 1 and b["frame_id"] == question["query_frame"]:
        current_name = b["class_name"].lower().replace("_", " ")
        subject = f"Track {tid}, labelled {current_name} in the paused frame,"
    if kind == "first":
        message = f"{subject} was first detected at {moment(a['frame_id'])}."
        message += (" It was already tracked at the start of this clip." if a["frame_id"] == first_frame
                    else " It may have been visible before the tracker picked it up.")
    elif kind == "last":
        message = f"{subject} was last detected at {moment(b['frame_id'])}."
        message += (" It is still tracked at the selected moment." if b["frame_id"] == question["query_frame"]
                    else " The tracker then lost it. A missed detection can also end a track.")
    elif a["frame_id"] == b["frame_id"]:
        return {"answer_status":"insufficient_evidence","answer":"There is only one recorded position for this track up to the playhead. Move forward to compare two positions.","cited_evidence_ids":[]}
    else:
        def centre(r):
            f = recording.get_frame(r["frame_id"])
            x0,y0,x1,y1 = r["bbox_xyxy"]
            return (x0+x1)/(2*f["width"]), (y0+y1)/(2*f["height"])
        ax,ay = centre(a)
        bx,by = centre(b)
        directions = []
        if abs(bx-ax) > .02:
            directions.append("right" if bx>ax else "left")
        if abs(by-ay) > .02:
            directions.append("down" if by>ay else "up")
        change = "shifted " + " and ".join(directions) if directions else "stayed near the same position"
        message = (f"{subject} {change} in the image between {moment(a['frame_id'])} and {moment(b['frame_id'])}. "
                   "This compares the box centres as seen by the moving camera.")
    return {"answer_status":"answered","answer":message,"cited_evidence_ids":citations}
