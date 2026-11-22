"""BoT-SORT (ultralytics 8.4.215) used detector-agnostically.

The tracker receives detections in original-image pixels plus the original BGR frame (for
camera-motion compensation). It returns ``(track_id, detection_index)`` pairs so the caller
exports the detector's own box, never the tracker's Kalman-smoothed box.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Sequence

from .classes import EMT_INDEX
from .config import TrackerSettings
from .detections import Detection

REQUIRED_ULTRALYTICS = "8.4.215"

class BotSortTracker:
    def __init__(self, settings: TrackerSettings):
        os.environ["YOLO_AUTOINSTALL"] = "false"  # never pip-install at runtime (e.g. lap)
        import numpy as np
        import ultralytics
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.bot_sort import BOTSORT

        # Detection-index mapping (row[7]) and per-tracker ID counters were verified on this exact version.
        if ultralytics.__version__ != REQUIRED_ULTRALYTICS:
            raise RuntimeError(f"Task 1 requires ultralytics=={REQUIRED_ULTRALYTICS}, found {ultralytics.__version__}")

        settings.validate()
        self._np, self._boxes = np, Boxes
        self._tracker = BOTSORT(SimpleNamespace(**settings.as_namespace_dict()))
        self._tracker.reset()  # IDs restart at 1, Kalman/GMC state cleared
        self.updates = 0

    def update(self, detections: Sequence[Detection], image_bgr) -> list[tuple[int, int]]:
        np = self._np
        data = np.asarray(
            [[*d.box, d.score, EMT_INDEX[d.name]] for d in detections], dtype=np.float32,
        ).reshape(-1, 6)
        output = self._tracker.update(self._boxes(data, image_bgr.shape[:2]).numpy(), image_bgr)
        self.updates += 1
        rows = np.asarray(output, dtype=np.float64).reshape(-1, 8)
        pairs = [(int(row[4]), int(row[7])) for row in rows]
        indices = [index for _, index in pairs]
        if any(not 0 <= index < len(detections) for index in indices) or len(set(indices)) != len(indices):
            raise RuntimeError(f"Tracker returned invalid detection indices {indices} for {len(detections)} inputs")
        if len({track for track, _ in pairs}) != len(pairs):
            raise RuntimeError("Tracker returned a track ID twice in one frame")
        return pairs
