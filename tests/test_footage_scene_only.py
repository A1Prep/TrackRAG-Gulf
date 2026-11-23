"""Current road-scene questions need the selected original, regardless of lexical track hits."""
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from trackrag.retrieval import build_index, rank_evidence as rank_windows
from trackrag.retrieval_temporal import current_scene_layout, rank_evidence


def source(frames=(1, 11, 21), *, record_source="predicted"):
    originals = {n: {"recording_id": "road", "frame_id": n, "timestamp_s": (n-1)/10,
                     "width": 160, "height": 90, "image_path": f"frame-{n}.jpg"}
                 for n in (1, 11, 21)}
    recording = SimpleNamespace(recording_id="road", frame_ids=sorted(originals),
                                get_frame=originals.__getitem__, iter_frames=lambda: iter(originals.values()))
    observations = [{"recording_id": "road", "frame_id": n, "track_id": 3, "source": record_source,
                     "class_name": "Car", "bbox_xyxy": [10, 20, 25, 36]} for n in frames]
    return build_index(recording, observations)


def question(words, cutoff=11):
    return {"recording_id": "road", "question_id": "q", "query_frame": cutoff, "question": words}


@pytest.mark.parametrize("words", [
    "Does the lane ahead bend to the left or to the right?",
    "Is the sandy shoulder on the same side as the tall pointed tower?",
    "Is the brightest spot above the horizon or down on the pavement?",
    "Which side of the picture has the blue overhead direction boards?",
    "Are the cars currently underneath the elevated structure or out in the open?",
    "Where is the viaduct relative to the road?",
])
def test_present_layout_always_cites_only_the_full_selected_original(words):
    assert current_scene_layout(words)
    result = rank_evidence(question(words), source((1, 21)))
    assert [(r["track_id"], r["source"], r["start_frame"], r["end_frame"]) for r in result] == [
        (None, "predicted", 11, 11)]
    assert rank_evidence(question(words), source((1,))) == result


def test_lexical_directions_can_select_an_old_car_and_untracked_landmarks_can_select_nothing():
    index = source((1,))
    bend = question("Does the lane ahead bend left or right?")
    shoulder = question("Is the sandy shoulder on the same side as the tower?")
    assert all(row["end_frame"] == 1 for row in rank_windows(bend, index))
    assert rank_windows(bend, index) and not rank_windows(shoulder, index)
    assert rank_evidence(bend, index)[0]["start_frame"] == rank_evidence(shoulder, index)[0]["start_frame"] == 11


def test_empty_known_source_and_future_rows_give_the_same_current_original():
    q = question("Which side of the road has the towers?")
    empty, future = source(()), source((21,))
    assert rank_evidence(q, empty) == rank_evidence(q, future) == []
    result = rank_evidence(q, empty, observation_source="predicted")
    assert rank_evidence(q, future, observation_source="predicted") == result
    assert result[0]["track_id"] is None and result[0]["start_frame"] == 11
    with pytest.raises(ValueError, match="different sources"):
        rank_evidence(q, source(record_source="annotated"), observation_source="predicted")
    assert rank_evidence(question(q["question"], 9), empty, observation_source="predicted") == []


@pytest.mark.parametrize("words", [
    "Which side of the road had a tower at 0:00?", "Which side of the road has towers in frame 1?",
    "Which side of the road will have a tower in the next frame?", "Why is the road on the left?",
    "How many metres wide is the lane on the right?", "What is the exact height of the tower?",
    "Which side of the road is track 999 on?", "Was the building on the left earlier?",
    "Compare the right shoulder before and after the bend.", "Which side of the road is the bus beside the gate?",
    "Which direction does the lorry move relative to the road in its recent observations?",
    "What structure is mounted above the pickup's bed?",
])
def test_scene_fallback_does_not_override_times_identity_or_unobservable_requests(words):
    assert not current_scene_layout(words)


def test_explicit_sample_and_future_and_physical_measurement_preserve_their_own_results():
    index = source()
    earlier = rank_evidence(question("Which side of the road has a tower at 0:00?"), index)
    assert earlier and all(r["end_frame"] == 1 for r in earlier)
    opening = rank_evidence(question("At the start, does the road bend left or right?"), index)
    assert opening and all(r["end_frame"] == 1 for r in opening)
    for words in ("Which side of the road has a tower at 0:02?", "Which side of the road has a tower at 0:00.5?",
                  "Which side of the road will have a tower in the next frame?",
                  "How many metres wide is the lane on the right?", "What is the exact height of the tower?"):
        assert rank_evidence(question(words), index) == []


def test_normal_reviewer_cites_the_exact_empty_detector_frame(tmp_path):
    from trackrag.review import review_recording

    image = tmp_path / "original.png"
    Image.new("RGB", (160, 90), "gray").save(image)
    recording = source(()).recording
    get_frame = recording.get_frame
    recording.get_frame = lambda fid: {**get_frame(fid), "image_path": str(image)}
    recording.iter_frames = lambda: (recording.get_frame(fid) for fid in recording.frame_ids)
    result = review_recording(recording, [question("Does the road bend to the right?")], source="predicted",
                              load_tracks=lambda _: [], build_index=build_index, rank_evidence=rank_evidence,
                              run_metadata={"retrieval": "trackrag.retrieval_temporal"}, output_dir=tmp_path / "run")
    evidence = json.loads(result.read_bytes())["evidence"]
    assert len(evidence) == 1 and evidence[0]["interval"]["track_id"] is None
    assert evidence[0]["images"][0]["frame_id"] == 11
    assert (result.parent / evidence[0]["images"][0]["image_path"]).is_file()
