"""Select and reproduce the image-space roadside geometry permitted by the app configuration."""

from . import between_road
from .projected_pair import Ineligible


EDGE = "edge"
EDGE_OR_BAY = "edge-or-bay"
MODES = (EDGE, EDGE_OR_BAY)
BAY_GUARD = "side_facing_roadside_gap_v1"
# Duplicate/overlapping identities, invalid data and model errors are never new orientation evidence.
ORIENTATION_REFUSALS = {"indistinguishable_ground_contact_depth", "no_visible_shared_road_edge",
                        "no_supported_neighborhood"}
LATERAL_GEOMETRIC_REFUSALS = ORIENTATION_REFUSALS | {"no_visible_between_gap", "two_distinct_current_vehicles_required",
                                                     "selected_vehicles_substantially_occlude_each_other"}
BAY_GEOMETRIC_REFUSALS = {"two_distinct_current_vehicles_required", "incompatible_anchor_scales",
                          "selected_vehicles_substantially_occlude_each_other", "no_visible_between_gap",
                          "no_supported_neighborhood", "no_unoccupied_between_ground", "no_independent_roadside_evidence",
                          "no_ground_adjacent_to_independent_roadside"}


def check_mode(value):
    if value not in MODES:
        raise ValueError(f"Unknown between-road interpretation {value!r}; choose from {MODES}")
    return value


def mode(model):
    return check_mode(getattr(model, "between_road_mode", EDGE))


def evidence(model):
    selected = mode(model)
    return {"between_road_mode": selected} if selected != EDGE else {}


def preflight(model, observations, size):
    """Avoid loading the road mask for a geometry that every configured method rejects."""
    try:
        return between_road.contact_geometry(observations, size)
    except Ineligible as error:
        if mode(model) == EDGE_OR_BAY and str(error) in ORIENTATION_REFUSALS:
            return {}
        raise


class UnsupportedOrientations(Ineligible):
    def __init__(self, primary, alternative):
        super().__init__(str(primary))
        self.bay_reason = str(alternative)


def refusal(error):
    details = {"withheld_reason": str(error)}
    if isinstance(error, UnsupportedOrientations):
        details["bay_withheld_reason"] = error.bay_reason
    return details


def estimate(model, mask, image, observations, tracks, text, *, image_sha):
    """Keep a successful lateral estimate. A bay requires its own positive image evidence."""
    selected = mode(model)
    current_source = getattr(model.road,"image_hash",None)
    if current_source is not None and current_source != image_sha:
        raise ValueError("Road evidence belongs to a different original photograph")
    try:
        return between_road.estimate(model.geometry, mask, image.size, observations, tracks, text)
    except Ineligible as error:
        if str(error) not in LATERAL_GEOMETRIC_REFUSALS:
            raise ValueError(f"Invalid evidence during lateral roadside estimation: {error}") from error
        if selected != EDGE_OR_BAY or str(error) not in ORIENTATION_REFUSALS:
            raise
        from . import curb_facing
        from .sidewalk import mask_for
        sidewalk = mask_for(model.road,image,image_sha)
        try:
            patch = curb_facing.estimate(model.geometry, mask, image.size, observations, tracks, text,
                                         image=image,sidewalk_mask=sidewalk)
        except Ineligible as alternate:
            # An unexamined competitor cannot be ruled out to make another pair appear unique.
            if str(alternate) not in BAY_GEOMETRIC_REFUSALS:
                raise ValueError(f"Invalid evidence during parking-row estimation: {alternate}") from alternate
            raise UnsupportedOrientations(error, alternate) from alternate
        if patch.get("guard") != BAY_GUARD:
            raise ValueError("Bay estimator returned an unrecognized road-evidence recipe")
        return {**patch, "lateral_withheld_reason": str(error)}


def validate_patch(patch, observations, tracks, size, point, box, selected_mode):
    """Check static geometry against the two exact track observations for the configured method."""
    selected_mode = check_mode(selected_mode)
    if patch.get("guard") == between_road.VERSION:
        return between_road.validate_patch(patch, observations, tracks, size, point, box)
    if (selected_mode != EDGE_OR_BAY or patch.get("guard") != BAY_GUARD
            or patch.get("lateral_withheld_reason") not in ORIENTATION_REFUSALS):
        raise ValueError("Between-vehicle proposal does not match its configured geometry")
    from . import curb_facing
    return curb_facing.validate_patch(patch, observations, tracks, size, point, box)
