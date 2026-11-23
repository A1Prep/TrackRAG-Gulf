"""Causal reassociation of a new BoT-SORT ID with a recently lost track.

The input boxes are detector measurements in original pixels. Only the identity is
rewritten. Existing IDs observed in this frame always have priority over handoffs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from trackrag.contracts import TrackObservation

from .classes import GROUP_OF
from .geometry import Box, iou


@dataclass
class _LastSeen:
    raw_id: int
    frame: int
    group: str
    box: Box
    # Smoothed corner displacement in pixels per frame, including apparent scale changes.
    velocity: tuple[float, ...]

    def overlap(self, frame: int, box: Box) -> float:
        horizon = min(frame - self.frame, 5)
        predicted = tuple(p + v * horizon for p, v in zip(self.box, self.velocity))
        return max(iou(self.box, box), iou(predicted, box))


def _clear_winner(scores: list[tuple[float, int]], margin: float) -> int | None:
    if not scores:
        return None
    scores.sort(reverse=True)
    if len(scores) > 1 and scores[0][0] - scores[1][0] < margin:
        return None
    return scores[0][1]


class TrackContinuity:
    """Keep a short-lived list of missing tracks and a persistent raw-ID ownership map."""

    def __init__(self, max_gap: int, min_iou: float, margin: float = 0.08):
        if max_gap < 1 or not 0 < min_iou <= 1 or not 0 < margin <= 1:
            raise ValueError("Invalid track continuity settings")
        self.max_gap, self.min_iou, self.margin = max_gap, min_iou, margin
        self._owner: dict[int, int] = {}
        self._recent: dict[int, _LastSeen] = {}
        self._next = 1
        self._frame: int | None = None
        self.handoffs = 0

    @classmethod
    def frozen_profile(cls) -> "TrackContinuity":
        """V1 development-selected operating point, included in the tracker's versioned kind."""
        return cls(max_gap=25, min_iou=0.15, margin=0.08)

    def relabel_observations(self, frame: int, rows: Sequence[TrackObservation]) -> list[TrackObservation]:
        ids = self.update(frame, [(row["track_id"], row["class_name"], row["bbox_xyxy"]) for row in rows])
        return [{**row, "track_id": canonical} for row, canonical in zip(rows, ids)]

    def update(self, frame: int, tracks: Sequence[tuple[int, str, Box]]) -> list[int]:
        """Return one canonical ID per input, using only this and earlier frames."""
        if self._frame is not None and frame <= self._frame:
            raise ValueError("Track continuity requires strictly increasing frame IDs")
        self._frame = frame
        if len({raw for raw, _, _ in tracks}) != len(tracks):
            raise ValueError("A raw track ID occurs twice in one frame")
        self._recent = {key: value for key, value in self._recent.items() if frame - value.frame <= self.max_gap}
        occupied = {self._owner[raw] for raw, _, _ in tracks if raw in self._owner}
        by_raw = defaultdict(list)
        by_canonical = defaultdict(list)
        for raw, label, box in tracks:
            if raw in self._owner:
                continue
            for canonical, last in self._recent.items():
                if canonical in occupied or last.group != GROUP_OF[label]:
                    continue
                score = last.overlap(frame, box)
                if score >= self.min_iou:
                    by_raw[raw].append((score, canonical))
                    by_canonical[canonical].append((score, raw))

        best_raw = {key: _clear_winner(scores, self.margin) for key, scores in by_canonical.items()}
        for raw, scores in by_raw.items():
            canonical = _clear_winner(scores, self.margin)
            if canonical is not None and best_raw[canonical] == raw:
                del self._owner[self._recent[canonical].raw_id]
                self._owner[raw] = canonical
                self.handoffs += 1

        # Allocate in raw-ID order to give saved rows and live backend output identical identities.
        for raw in sorted(raw for raw, _, _ in tracks if raw not in self._owner):
            self._owner[raw] = self._next
            self._next += 1

        ids = []
        for raw, label, box in tracks:
            canonical = self._owner[raw]
            last = self._recent.get(canonical)
            velocity = (0.0,) * 4
            if last is not None:
                elapsed = frame - last.frame
                velocity = tuple(0.5 * (current - previous) / elapsed + 0.5 * speed
                                 for current, previous, speed in zip(box, last.box, last.velocity))
            self._recent[canonical] = _LastSeen(raw, frame, GROUP_OF[label], box, velocity)
            ids.append(canonical)
        return ids
