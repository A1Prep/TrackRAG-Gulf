"""Checks shared by passenger vehicle marks and existing-model point reconciliation."""
import re

import cv2
import numpy as np

from . import curbside_bays, painted_bays, parking_row_bays


_RESTRICTED = re.compile(
    r"\b(?:gravel|unpaved|work\s*(?:site|zone|area)|construction|coned|closed|"
    r"crosswalk|crossing|lanes?|footpath|sidewalk|reserved|free|paid|unless|without|except|only|"
    r"blocked|barrier|cycleway|bike\s+path|sand|dirt|grass|permit|illegal|legal|safe)\b",
    re.IGNORECASE,
)


def restricted_instruction(text):
    """Keep a requested surface, access restriction or condition attached to a passenger mark."""
    return not isinstance(text, str) or bool(_RESTRICTED.search(text))


def is_complete(option):
    """Recognize an existing measured stall whose visible outline covers both observed dividers."""
    evidence = option.get("evidence", {})
    recipe = evidence.get("recipe")
    coverage = evidence.get("coverage")
    if coverage not in (None, "measured_stall"):
        return False
    outline = np.asarray(option.get("polygon", []), dtype=np.float32)
    if (outline.ndim != 2 or outline.shape[1] != 2 or len(outline) < 3
            or not np.isfinite(outline).all() or cv2.contourArea(outline) <= 0):
        return False
    if recipe == parking_row_bays.RECIPE:
        return coverage == "measured_stall" and evidence.get("boundary_kind") in {
            "aisle_to_curb_parking_row", "curb_facing_capped_row",
        }
    if recipe not in (painted_bays.RECIPE, painted_bays.ROW_RECIPE, curbside_bays.RECIPE):
        return False

    dividers = np.asarray(evidence.get("painted_boundaries", []), dtype=np.float32)
    if dividers.shape != (2, 2, 2) or not np.isfinite(dividers).all():
        return False
    # Optical dividers are recorded before small occlusions are trimmed from a displayed outline.
    # Comparing their four actual endpoints keeps a clipped visible portion out of reconciliation.
    return all(cv2.pointPolygonTest(outline, (float(x), float(y)), True) >= -0.01
               for x, y in dividers.reshape(4, 2))
