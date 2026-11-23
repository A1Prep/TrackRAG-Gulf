import pytest

from trackrag.vision.config import PROJECT_ROOT, load_config
from trackrag.vision.continuity import TrackContinuity


def box(raw, x, name="Car"):
    return (raw, name, (x, 10.0, x + 20.0, 30.0))


def test_handoff_keeps_all_observations_and_reappearing_ids_remain_distinct():
    model = TrackContinuity.frozen_profile()
    assert model.update(1, [box(10, 2), box(11, 60)]) == [1, 2]
    assert model.update(2, [box(11, 61)]) == [2]
    assert model.update(4, [box(14, 4, "Medium_vehicle"), box(11, 60)]) == [1, 2]
    # The original raw ID returns while its previous canonical identity is still occupied.
    assert model.update(5, [box(14, 5), box(10, 6), box(11, 60)]) == [1, 3, 2]
    assert model.handoffs == 1


def test_refuses_ambiguous_competitors_in_both_directions():
    two_new = TrackContinuity.frozen_profile()
    assert two_new.update(1, [box(1, 10)]) == [1]
    assert two_new.update(2, [box(3, 11), box(2, 9)]) == [3, 2]
    two_old = TrackContinuity.frozen_profile()
    assert two_old.update(1, [box(7, 9), box(6, 11)]) == [2, 1]
    assert two_old.update(2, [box(8, 10)]) == [3]
    assert two_new.handoffs == two_old.handoffs == 0


def test_uses_real_frame_gap_and_does_not_join_different_road_user_groups():
    model = TrackContinuity.frozen_profile()
    assert model.update(1, [box(4, 10)]) == [1]
    assert model.update(2, [box(5, 11, "Pedestrian")]) == [2]
    assert model.update(27, [box(6, 11)]) == [3]
    # Native BoT-SORT can retain an unchanged ID beyond the custom geometric handoff window.
    assert model.update(100, [box(4, 18), box(5, 20, "Pedestrian")]) == [1, 2]
    assert model.handoffs == 0
    with pytest.raises(ValueError, match="strictly increasing"):
        model.update(100, [])


def test_published_identity_is_compatible_and_new_method_has_its_own_identity():
    original = load_config(PROJECT_ROOT / "configs/vision/finetuned.json")
    updated = load_config(PROJECT_ROOT / "configs/vision/finetuned-continuity.json")
    assert original.identity_hash() == "acf21b53525d5729541aaedbf961a9c634399fa881fc7976909d48575c5e870d"
    assert updated.identity_hash() != original.identity_hash()
    assert updated.with_overrides(tracker={"kind": "ultralytics_botsort"}).identity() == original.identity()
