"""Offline research-export checks using generated pixels, not real-model accuracy."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
from PIL import Image
import pytest

from tools import passenger_road_points as points
from trackrag.dropoff.road_cache import mask_shape, save_mask


SIZE = (512, 288)


def request(image_path="input.png", identity="fixture-request"):
    scene = {"frame": {"recording_id": "software-fixture", "frame_id": 4, "timestamp_s": .3,
                       "width": SIZE[0], "height": SIZE[1], "image_path": str(image_path)},
             "source": "predicted", "tracks": [], "provenance": {"software_fixture": True}}
    for identity_number, box in ((1, [80, 100, 140, 160]), (2, [220, 100, 280, 160])):
        scene["tracks"].append({"recording_id": "software-fixture", "frame_id": 4,
                                "track_id": identity_number, "class_name": "Car",
                                "bbox_xyxy": box, "source": "predicted"})
    return {"request_id": identity, "text": "The ground between those two cars, please.",
            "input_mode": "typed", "dialogue": [], "scene": scene}


def test_symmetric_free_gap_uses_a_real_cell_center_not_a_floating_centroid():
    row = request()
    before = deepcopy(row)
    options = points.generate_options(row, np.ones(mask_shape(SIZE), dtype=bool))
    assert options == [{"point": [179.5, 159.5], "tracks": [1, 2],
                        "anchor_boxes": [[80, 100, 140, 160], [220, 100, 280, 160]], "choice": "A"}]
    assert row == before


def test_original_image_scale_does_not_change_projected_cell_choice():
    row = request()
    row["scene"]["frame"]["width"], row["scene"]["frame"]["height"] = 1024, 576
    for track in row["scene"]["tracks"]:
        track["bbox_xyxy"] = [2 * x for x in track["bbox_xyxy"]]
    result = points.generate_options(row, np.ones(mask_shape((1024, 576)), dtype=bool))
    assert result[0]["point"] == [359.0, 319.0]


def test_intervening_vehicle_prevents_connecting_nonadjacent_anchors():
    row = request()
    row["scene"]["tracks"].append({**row["scene"]["tracks"][0], "track_id": 3,
                                    "bbox_xyxy": [155, 100, 195, 160]})
    options = points.generate_options(row, np.ones(mask_shape(SIZE), dtype=bool))
    assert options
    assert all(set(option["tracks"]) != {1, 2} for option in options)
    assert all(3 in option["tracks"] for option in options)


def test_no_three_by_three_free_road_or_occupied_gap_gives_no_point():
    row = request()
    assert points.generate_options(row, np.zeros(mask_shape(SIZE), dtype=bool)) == []
    obstacle = {**row["scene"]["tracks"][0], "track_id": 3, "class_name": "Pedestrian",
                "bbox_xyxy": [140, 140, 220, 180]}
    row["scene"]["tracks"].append(obstacle)
    assert points.generate_options(row, np.ones(mask_shape(SIZE), dtype=bool)) == []


def test_output_is_bounded_to_eight_stable_letter_ids():
    row = request()
    prototype = row["scene"]["tracks"][0]
    row["scene"]["tracks"] = [
        {**prototype, "track_id": i + 1, "bbox_xyxy": [8 + 30*i, 100, 22 + 30*i, 160]}
        for i in range(16)
    ]
    options = points.generate_options(row, np.ones(mask_shape(SIZE), dtype=bool))
    assert len(options) == 8
    assert [option["choice"] for option in options] == list("ABCDEFGH")
    assert all(set(option["tracks"]) <= set(range(1, 15)) for option in options)
    assert options == points.generate_options(row, np.ones(mask_shape(SIZE), dtype=bool))


@pytest.mark.parametrize("change", ["gold", "historical", "future", "annotated"])
def test_noncurrent_or_oracle_inputs_cannot_enter_the_research_generator(change):
    row = request()
    if change == "gold":
        row["target"] = {"destination_point_xy": [180, 160]}
    elif change == "annotated":
        row["scene"]["source"] = "annotated"
    else:
        row["scene"]["tracks"][0]["frame_id"] = 3 if change == "historical" else 5
    with pytest.raises(ValueError):
        points.generate_options(row, np.ones(mask_shape(SIZE), dtype=bool))


def fixture_files(tmp_path, *, identity="fixture-request"):
    image = tmp_path / "input.png"
    Image.new("RGB", SIZE, (65, 65, 65)).save(image)
    row = request(image, identity)
    row["scene"]["provenance"]["image_sha256"] = sha256(image.read_bytes()).hexdigest()
    source = tmp_path / "requests.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cache = tmp_path / "cache"
    return source, cache, row


def test_export_saves_bound_images_exact_prompt_and_scaled_points(tmp_path):
    source, cache, row = fixture_files(tmp_path, identity="../../untrusted-id")
    save_mask(cache, row["scene"]["provenance"]["image_sha256"], SIZE,
              np.ones(mask_shape(SIZE), dtype=bool))
    out = tmp_path / "export"
    manifest = points.export(source, cache, out)
    record, = json.loads((out / "road-choices.json").read_text())
    marked = next(out.glob("*-road-points.jpg"))
    assert record["id"] == "../../untrusted-id"
    assert record["marked_sha256"] == sha256(marked.read_bytes()).hexdigest()
    assert record["sha256"] == row["scene"]["provenance"]["image_sha256"]
    assert record["options"][0]["point"] == record["scaled_points"][0]["point_marked_xy"]
    assert record["system"] == points.MARKED + " Offered letters: A"
    assert (out / "selection-system-prompt.txt").read_text() == points.MARKED + "\n"
    assert (out / "input-snapshot.jsonl").read_bytes() == source.read_bytes()
    assert manifest["models_run"] is False and manifest["labels_read"] is False
    assert manifest["candidate_count"] == 1
    assert not (tmp_path.parent / "untrusted-id-road-points.jpg").exists()
    with pytest.raises(FileExistsError):
        points.export(source, cache, out)


def test_missing_cache_never_starts_downloads_or_creates_partial_output(tmp_path):
    source, cache, row = fixture_files(tmp_path)
    with pytest.raises(FileNotFoundError, match="road cache"):
        points.export(source, cache, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_changed_image_or_declared_dimensions_rejected_before_export(tmp_path):
    source, cache, row = fixture_files(tmp_path)
    row["scene"]["frame"]["width"] += 1
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions"):
        points.export(source, cache, tmp_path / "wrong-size")
    row["scene"]["frame"]["width"] -= 1
    row["scene"]["provenance"]["image_sha256"] = "0" * 64
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        points.export(source, cache, tmp_path / "wrong-hash")
