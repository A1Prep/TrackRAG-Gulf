import pytest

from trackrag.dropoff import curbside_bays, painted_bays, parking_row_bays, surface_stalls
from trackrag.dropoff.vehicle_relative_bays import is_complete, restricted_instruction


@pytest.mark.parametrize("phrase", [
    "Pull up in the traffic lane", "Put me on the gravel", "Inside the coned work zone",
    "Only if parking is free", "By the pedestrian crossing", "On the sidewalk",
])
def test_destination_conditions_survive_a_passenger_mark(phrase):
    assert restricted_instruction(phrase)


def test_plain_vehicle_relative_request_allows_a_checked_mark():
    assert not restricted_instruction("Beside the red car on its left")
    assert restricted_instruction(None)


@pytest.mark.parametrize("recipe", [painted_bays.RECIPE, painted_bays.ROW_RECIPE, curbside_bays.RECIPE])
def test_measured_paint_must_cover_both_dividers(recipe):
    left = [[10, 12], [8, 31]]
    right = [[18, 12], [23, 31]]
    evidence = {"recipe": recipe, "painted_boundaries": [left, right]}
    outline = [left[0], right[0], right[1], left[1]]
    assert is_complete({"polygon": outline, "evidence": evidence})
    assert not is_complete({"polygon": outline, "evidence": {**evidence, "painted_boundaries": [left]}})
    assert not is_complete({"polygon": outline, "evidence": {**evidence, "coverage": "visible_portion"}})
    assert not is_complete({"polygon": [*outline[:3], [11, 31]], "evidence": evidence})


def test_measured_row_uses_its_native_coverage_verdict():
    outline = [[8, 31], [10, 12], [18, 12], [23, 31]]
    proof = {"recipe": parking_row_bays.RECIPE, "coverage": "measured_stall",
             "boundary_kind": "aisle_to_curb_parking_row"}
    assert is_complete({"polygon": outline, "evidence": proof})
    assert not is_complete({"polygon": outline, "evidence": {**proof, "coverage": "visible_portion"}})
    assert not is_complete({"polygon": outline, "evidence": {**proof, "boundary_kind": "repeated_pavement_studs"}})
    assert not is_complete({"polygon": outline, "evidence": {**proof, "recipe": surface_stalls.RECIPE}})
    assert not is_complete({})
