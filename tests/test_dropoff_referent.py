import copy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff.referent import (
    ALIGN_NAMES, KIND, REGION_NAMES, ReferentRanker, category, ordered_alternatives, overlap,
)


def checkpoint(tmp_path):
    base = [0.] * len(REGION_NAMES)
    aligned = [0.] * len(ALIGN_NAMES)
    aligned[ALIGN_NAMES.index("noun_agreement")] = 3.
    aligned[ALIGN_NAMES.index("colour_agreement")] = 3.
    aligned[ALIGN_NAMES.index("right_cue")] = 4.
    aligned[ALIGN_NAMES.index("left_cue")] = 4.
    payload = {"kind": KIND, "base": base, "aligned": aligned, "vocabulary": [],
               "interactions": [], "region_names": list(REGION_NAMES), "align_names": list(ALIGN_NAMES)}
    raw = json.dumps(payload).encode()
    path = tmp_path / "fixture.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def setup_scene(tmp_path):
    path, digest = checkpoint(tmp_path)
    model = ReferentRanker.load(path, expected_sha256=digest)
    image = Image.new("RGB", (100, 50), (0, 0, 0))
    image.paste((220, 30, 30), (10, 15, 35, 40))
    image.paste((30, 40, 230), (65, 15, 90, 40))
    frame = {"width": 100, "height": 50, "frame_id": 6, "recording_id":"v", "timestamp_s": .6,
             "image_path": "frame.jpg"}
    tracks = [{"source":"predicted", "recording_id":"v", "frame_id":6, "track_id":i,
               "class_name":"Car", "bbox_xyxy":b}
              for i,b in ((10, [10.,15.,35.,40.]), (11,[65.,15.,90.,40.]))]
    return model, image, frame, tracks


def test_rank_uses_current_pixels_language_and_position(tmp_path):
    model, image, frame, tracks = setup_scene(tmp_path)
    for text, intended in (("The red car",10),("The blue car",11),("The car on the right",11),
                           ("The left car",10)):
        result = model.rank(image, text, list(reversed(tracks)), frame=frame)
        assert result["alternatives"][0]["track_id"] == intended
        assert result["candidate_count"] == result["distinct_candidate_count"] == 2
        assert np.isclose(sum(v["score"] for v in result["alternatives"])+result["null_score"], 1.)
    past = dict(tracks[1],frame_id=5,track_id=99)
    assert model.rank(image, "left car", tracks+[past], frame=frame)["candidate_count"] == 2


@pytest.mark.parametrize("invalid", [{"source":"annotated"},{"frame_id":7},{"recording_id":"wrong"},
                                     {"bbox_xyxy":[1,1,400,401]},{"ground_truth":True}])
def test_reject_incompatible_observation(tmp_path,invalid):
    model,image,frame,tracks=setup_scene(tmp_path)
    bad = copy.deepcopy(tracks)
    bad[0].update(invalid)
    with pytest.raises(ValueError):
        model.rank(image,"stop near the car",bad,frame=frame)


def test_empty_frame_duplicate_suppression_and_checkpoint_integrity(tmp_path):
    model, image, frame, tracks = setup_scene(tmp_path)
    result = model.rank(image, "car", [], frame=frame)
    assert result["alternatives"] == [] and result["null_score"] == 1.
    duplicate = dict(tracks[0], track_id=20)
    duplicates = model.rank(image, "red car", tracks+[duplicate], frame=frame)
    assert duplicates["candidate_count"] == 3 and duplicates["distinct_candidate_count"] == 2
    assert overlap([[0,0,10,10],[2,2,8,8]], [0,0,10,10]).tolist() == [1.,.36]
    assert ordered_alternatives([], [], []) == []
    with pytest.raises(ValueError,match="size"):
        model.rank(image.resize((50,50)), "car", tracks, frame=frame)
    path,digest=checkpoint(tmp_path)
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match="SHA256"):
        ReferentRanker.load(path,expected_sha256=digest)
    assert category("human.pedestrian.construction_worker")=="person" and category("Motorbike")=="motorcycle"
