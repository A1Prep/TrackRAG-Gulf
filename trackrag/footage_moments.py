"""Find inspectable moments in saved tracks from attached recordings."""
from collections import Counter, defaultdict
from collections.abc import Iterable
import re

from .contracts import TrackObservation
from .vision.classes import EMT_CLASSES


BUSY = "busy"
PAGE_SIZE = 6
_TERMS = {name.lower().replace("_", " "): name for name in EMT_CLASSES}
_TERMS.update({term + "s": name for term, name in tuple(_TERMS.items()) if term != "bus"})
_TERMS.update(buses="Bus", motorcycle="Motorbike", motorcycles="Motorbike", bicycle="Cyclist",
              bicycles="Cyclist", bicyclist="Cyclist", bicyclists="Cyclist", people="Pedestrian",
              person="Pedestrian", **{"people on foot": "Pedestrian", "busy traffic": BUSY,
                                      "busiest moments": BUSY, "most road users": BUSY})


def search_type(query: str) -> str | None:
    """Recognize complete type requests without dropping unsupported qualifiers."""
    if not isinstance(query, str) or len(query) > 100:
        return None
    words = " ".join(query.casefold().replace("-", " ").strip(" .?!").split())
    words = re.sub(r"^(?:find|show(?: me)?|look for|search for) ", "", words)
    words = re.sub(r"^(?:a|an|the|any) ", "", words)
    return _TERMS.get(words)


def _area(row: TrackObservation) -> float:
    left, top, right, bottom = row["bbox_xyxy"]
    return (right - left) * (bottom - top)


def recording_moments(recording_id: str, clock: list[dict], observations: Iterable[TrackObservation]) -> dict:
    """Choose one legible frame per repeatedly labelled track and one busy frame per recording."""
    times = {frame["id"]: frame["time"] for frame in clock}
    tracks = defaultdict(dict)
    for row in observations:
        if (row["recording_id"] == recording_id and row["source"] == "predicted"
                and row["frame_id"] in times and row["class_name"] in EMT_CLASSES):
            tracks[row["track_id"]][row["frame_id"]] = row

    moments = defaultdict(list)
    counts = Counter()
    for track_id, frames in tracks.items():
        counts.update(frames.keys())
        labels = Counter(row["class_name"] for row in frames.values())
        name, sightings = labels.most_common(1)[0]
        if sightings < 2 or sightings * 2 <= len(frames):
            continue
        first_frame = min(frames)
        representative = max((row for row in frames.values() if row["class_name"] == name),
                             key=lambda row: (_area(row), -row["frame_id"]))
        frame = representative["frame_id"]
        moments[name].append({"frame_id": frame, "timestamp_s": times[frame], "track_id": track_id,
                              "class_name": name, "bbox_xyxy": list(representative["bbox_xyxy"]),
                              "first_frame": first_frame, "first_box": list(frames[first_frame]["bbox_xyxy"]),
                              "observations": sightings})
    if counts:
        peak = max(counts, key=lambda frame: (counts[frame], -frame))
        moments[BUSY].append({"frame_id": peak, "timestamp_s": times[peak], "track_id": None,
                              "observations": counts[peak]})
    return dict(moments)
