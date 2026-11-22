"""Image resizing and box geometry shared by data preparation, inference and tests.

All boxes are ``(left, top, right, bottom)`` tuples of Python floats. ``ResizeTransform``
maps boxes between an original frame and the pre-resized copy given to the detector, so
the exported boxes are always in original-image pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Box = tuple[float, float, float, float]

DEFAULT_LONG_SIDE = 1280


def _box(values: Sequence[float]) -> Box:
    if len(values) != 4:
        raise ValueError("A box needs four coordinates")
    box = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in box):
        raise ValueError("Box coordinates must be finite")
    return box  # type: ignore[return-value]


@dataclass(frozen=True)
class ResizeTransform:
    """Downscale-only resize so the longer side is at most ``long_side`` pixels.

    Separate x/y factors absorb rounding of the resized size, so ``to_original`` is the
    exact inverse of ``to_resized``. 2560x1440 and 3840x2160 both map to 1280x720.
    """

    width: int
    height: int
    resized_width: int
    resized_height: int

    @classmethod
    def for_size(cls, width: int, height: int, long_side: int = DEFAULT_LONG_SIDE) -> "ResizeTransform":
        for name, value in (("width", width), ("height", height), ("long_side", long_side)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        scale = min(1.0, long_side / max(width, height))
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        return cls(width, height, resized_width, resized_height)

    @property
    def scale_x(self) -> float:
        return self.width / self.resized_width

    @property
    def scale_y(self) -> float:
        return self.height / self.resized_height

    @property
    def is_identity(self) -> bool:
        return (self.resized_width, self.resized_height) == (self.width, self.height)

    @property
    def tolerance_px(self) -> float:
        """One resized pixel expressed in original pixels: the stated box-check tolerance."""
        return max(self.scale_x, self.scale_y)

    def to_resized(self, box: Sequence[float]) -> Box:
        left, top, right, bottom = _box(box)
        return (left / self.scale_x, top / self.scale_y, right / self.scale_x, bottom / self.scale_y)

    def to_original(self, box: Sequence[float]) -> Box:
        left, top, right, bottom = _box(box)
        return (left * self.scale_x, top * self.scale_y, right * self.scale_x, bottom * self.scale_y)


def resize_image(image, transform: ResizeTransform):
    """Return an RGB PIL image of the transform's resized size.

    Pillow's BILINEAR resize is antialiased when downscaling. The same function is used to
    build training images and to prepare inference frames, so both see identical pixels.
    """
    from PIL import Image

    if image.size != (transform.width, transform.height):
        raise ValueError(
            f"Image is {image.size[0]}x{image.size[1]}, expected {transform.width}x{transform.height}"
        )
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    if transform.is_identity:
        return rgb.copy()
    return rgb.resize(
        (transform.resized_width, transform.resized_height),
        resample=Image.Resampling.BILINEAR,
        reducing_gap=None,
    )


def clip_box(box: Sequence[float], width: float, height: float) -> Box | None:
    """Clip to ``[0, width] x [0, height]``; return None when no positive area remains."""
    left, top, right, bottom = _box(box)
    left, right = min(max(left, 0.0), float(width)), min(max(right, 0.0), float(width))
    top, bottom = min(max(top, 0.0), float(height)), min(max(bottom, 0.0), float(height))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def iou(first: Sequence[float], second: Sequence[float]) -> float:
    a_left, a_top, a_right, a_bottom = _box(first)
    b_left, b_top, b_right, b_bottom = _box(second)
    inter_w = min(a_right, b_right) - max(a_left, b_left)
    inter_h = min(a_bottom, b_bottom) - max(a_top, b_top)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    area_a = max(0.0, a_right - a_left) * max(0.0, a_bottom - a_top)
    area_b = max(0.0, b_right - b_left) * max(0.0, b_bottom - b_top)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def union_box(first: Sequence[float], second: Sequence[float]) -> Box:
    a, b = _box(first), _box(second)
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
