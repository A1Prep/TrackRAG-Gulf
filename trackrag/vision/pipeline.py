"""Frame loop for ``predict_tracks``: decode -> resize -> detect -> map back -> post-process -> track.

Strictly online: the observation exported for frame t depends only on frames <= t. A
background thread may decode upcoming frames early (decode only), which does not change any
output. No backfill, interpolation, minimum-hit filters or whole-track votes are applied.
"""

from __future__ import annotations

import math
import platform
import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict

from trackrag.contracts import TrackObservation

from .classes import RunningClassVote
from .config import VisionConfig
from .detections import Detection, coco_to_emt, drop_ego_vehicle, keep_emt, nms
from .geometry import ResizeTransform, clip_box, resize_image

ANNOTATION_ACCESS = "predict_tracks reads frame images through the Recording only; no label file is opened"
_END = object()


def postprocess(native: list[Detection], label_space: str, nms_iou: float, merge_iou: float):
    """Map detector outputs to EMT names and remove duplicates. Returns (mapped, dropped counts).

    COCO: NMS on the raw detections first (a duplicate low-score "bicycle" on a motorcycle's box
    must not steal its rider), then the rider merge / name mapping, then NMS again.
    EMT: drop non-EMT names (e.g. ``__background__``), then NMS.
    """
    if label_space == "coco":
        mapped, dropped = coco_to_emt(nms(native, nms_iou), merge_iou)
    elif label_space == "emt":
        mapped, dropped = keep_emt(native)
    else:
        raise ValueError(f"Unknown label space {label_space!r}")
    return nms(mapped, nms_iou), dropped


def _frames_with_pixels(recording, stop: threading.Event):
    """Yield (frame, rgb PIL image, bgr ndarray) using a small background decode queue."""
    buffer: queue.Queue = queue.Queue(maxsize=4)

    def put(item) -> bool:
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        try:
            import numpy as np
            from PIL import Image

            for frame in recording.iter_frames():
                if stop.is_set():
                    return
                # PIL decoding, as in tools/vision_prepare_dataset.py, so training and inference
                # see identical pixels. The BGR copy is only for the tracker's motion compensation.
                with Image.open(frame["image_path"]) as opened:
                    rgb = opened.convert("RGB")
                bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
                if not put((frame, rgb, bgr)):
                    return
        except BaseException as error:  # forwarded to the consumer
            put(error)
            return
        put(_END)

    thread = threading.Thread(target=producer, name="trackrag-vision-decode", daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def _check_observations(observations, recording_id, sizes):
    seen = set()
    for row in observations:
        key = (row["frame_id"], row["track_id"])
        if key in seen:
            raise RuntimeError(f"Duplicate observation for track {key[1]} in frame {key[0]}")
        seen.add(key)
        width, height = sizes[row["frame_id"]]
        left, top, right, bottom = row["bbox_xyxy"]
        if (row["recording_id"] != recording_id or row["source"] != "predicted"
                or type(row["frame_id"]) is not int or type(row["track_id"]) is not int
                or not all(type(v) is float and math.isfinite(v) for v in row["bbox_xyxy"])
                or not (0 <= left < right <= width and 0 <= top < bottom <= height)):
            raise RuntimeError(f"Invalid observation produced: {row}")


def run_recording(recording, config: VisionConfig, detector, tracker_factory, *, sink=None, log=None) -> list[TrackObservation]:
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    started = time.perf_counter()
    settings = config.detector
    tracker = tracker_factory(config.tracker)
    vote = RunningClassVote()
    empty: list[Detection] = []
    observations: list[TrackObservation] = []
    dropped: Counter = Counter()
    sizes: dict[int, tuple[int, int]] = {}
    frame_ids: list[int] = []
    missing = 0
    previous_id = previous_bgr = None
    stop = threading.Event()
    for frame, rgb, bgr in _frames_with_pixels(recording, stop):
        frame_id, width, height = frame["frame_id"], frame["width"], frame["height"]
        if frame["recording_id"] != recording.recording_id:
            raise RuntimeError("Recording returned a frame from another recording")
        if type(frame_id) is not int or (previous_id is not None and frame_id <= previous_id):
            raise RuntimeError("Frames must arrive with strictly increasing integer frame IDs")
        if rgb.size != (width, height) or bgr.shape[:2] != (height, width):
            raise RuntimeError(f"Frame {frame_id} image is {rgb.size}, metadata says {width}x{height}")
        if previous_id is not None:
            for _ in range(previous_id + 1, frame_id):  # advance time over missing frames, export nothing
                tracker.update(empty, previous_bgr)
                missing += 1
        transform = ResizeTransform.for_size(width, height, settings.inference_long_side)
        small = resize_image(rgb, transform)
        native: list[Detection] = []
        replay = getattr(detector, "detect_frame", None)
        if callable(replay):
            # Saved detections (tracker tuning): already in original pixels, same post-processing below.
            for detection in replay(frame_id, width, height):
                box = clip_box(detection.box, width, height)
                if box is not None:
                    native.append(Detection(box, detection.score, detection.name, detection.native))
        else:
            for detection in detector.detect(small):
                box = clip_box(transform.to_original(detection.box), width, height)
                if box is not None:
                    native.append(Detection(box, detection.score, detection.name, detection.native))
        mapped, frame_dropped = postprocess(native, settings.label_space, settings.nms_iou, settings.rider_merge_iou)
        dropped.update(frame_dropped)
        mapped, ego = drop_ego_vehicle(mapped, width, height, settings.ego_bottom_fraction,
                                       settings.ego_min_width_fraction)
        if ego:
            dropped["ego vehicle (camera car bonnet)"] += ego
        for track_id, index in tracker.update(mapped, bgr):
            detection = mapped[index]
            observations.append(TrackObservation(
                recording_id=recording.recording_id, frame_id=frame_id, track_id=int(track_id),
                class_name=str(vote.update(int(track_id), detection.name)),
                bbox_xyxy=tuple(float(value) for value in detection.box), source="predicted",
            ))
        if sink is not None:
            sink.frame({
                "recording_id": recording.recording_id, "frame_id": frame_id, "width": width, "height": height,
                "native": [{"bbox_xyxy": list(d.box), "score": d.score, "class_name": d.name} for d in native],
                "mapped": [{"bbox_xyxy": list(d.box), "score": d.score, "class_name": d.name, "native": d.native}
                           for d in mapped],
            })
        sizes[frame_id] = (width, height)
        frame_ids.append(frame_id)
        previous_id, previous_bgr = frame_id, bgr
    observations.sort(key=lambda row: (row["frame_id"], row["track_id"]))
    _check_observations(observations, recording.recording_id, sizes)
    seconds = time.perf_counter() - started
    identities = len({row["track_id"] for row in observations})
    describe = getattr(detector, "describe", None)
    provenance = {
        "version": 1, "recording_id": recording.recording_id, "config_name": config.name,
        "config_source": config.source_path, "identity_hash": config.identity_hash(), "identity": config.identity(),
        "detector_kind": getattr(detector, "kind", "stub"),
        "detector": describe() if callable(describe) else {"kind": getattr(detector, "kind", "stub")},
        "tracker": asdict(config.tracker), "frames": len(frame_ids),
        "first_frame": frame_ids[0] if frame_ids else None, "last_frame": frame_ids[-1] if frame_ids else None,
        "frame_ids": frame_ids, "first_frame_size": list(sizes[frame_ids[0]]) if frame_ids else None,
        "missing_frame_ids_advanced": missing, "observations": len(observations), "identities": identities,
        "dropped_detections": dict(sorted(dropped.items())), "seconds": round(seconds, 2),
        "seconds_per_frame": round(seconds / max(1, len(frame_ids)), 4), "annotation_access": ANNOTATION_ACCESS,
        "python": platform.python_version(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if sink is not None:
        sink.finish(provenance, observations)
    log(f"[vision] {recording.recording_id}: config={config.name} identity={provenance['identity_hash'][:12]} "
        f"detector={provenance['detector_kind']} frames={len(frame_ids)} observations={len(observations)} "
        f"identities={identities} seconds={seconds:.1f}")
    return observations
