from types import SimpleNamespace

import pytest

from trackrag.retrieval import build_index, rank_evidence as lexical
from trackrag.retrieval_temporal import rank_evidence


def make_index(observed_frames):
    ids = (10, 13, 18, 24, 31, 42, 64)
    frames = {fid: dict(recording_id="street", frame_id=fid, timestamp_s=(fid-10)/10,
                        width=100, height=60, image_path=f"{fid}.png") for fid in ids}
    recording = SimpleNamespace(recording_id="street", frame_ids=ids, get_frame=frames.__getitem__,
                                iter_frames=lambda: iter(frames.values()))
    observations = [dict(recording_id="street", frame_id=fid, track_id=7, class_name="Medium_vehicle",
                         bbox_xyxy=(20, 20, 40, 45), source="predicted") for fid in observed_frames]
    return build_index(recording, observations)


def question(text, cutoff=42):
    return dict(question_id="q", recording_id="street", query_frame=cutoff, question=text)


def test_empty_visual_object_search_uses_actual_selected_scene_and_candidate():
    index = make_index([31, 42, 64])
    q = question("What is mounted on top of the pickup?")
    assert lexical(q, index) == []
    assert [(i["track_id"], i["start_frame"], i["end_frame"]) for i in rank_evidence(q, index)] == [
        (7, 42, 42), (None, 42, 42)]
    assert rank_evidence(q, make_index([31, 42])) == rank_evidence(q, index)
    assert rank_evidence(q, make_index([64])) == rank_evidence(q, make_index([])) == []


@pytest.mark.parametrize("text", [
    "What does the pickup weigh?", "Why was the rack attached to the pickup?",
    "What colour will the pickup be next week?", "What was mounted on the pickup at 00:01?",
    "What was mounted on the pickup at 1.5 seconds?", "What was mounted on the pickup at frame #64?",
    "Which structure was mounted on the pickup at frame #33?",
])
def test_visual_fallback_preserves_nonvisual_untimed_and_unavailable_time_results(text):
    assert rank_evidence(question(text), make_index([31, 42, 64])) == []


def test_opening_movement_without_a_track_gets_bounded_scene_comparisons():
    index = make_index([10, 42, 64])
    q = question("Which direction does the early cyclist move?")
    ranked = rank_evidence(q, index)
    # The 30-frame opening window starts at original frame 10 and ends before 40.
    assert [i["start_frame"] for i in ranked] == [10, 18, 31]
    assert all(i["track_id"] is None and i["start_frame"] == i["end_frame"] for i in ranked)
    assert [i["start_frame"] for i in rank_evidence(q, index, k=2)] == [10, 31]
    earlier = question(q["question"], 13)
    assert [i["start_frame"] for i in rank_evidence(earlier, index)] == [10, 13]
    explicit = question("Which direction does the early cyclist move through frame 64?", 64)
    assert [i["start_frame"] for i in rank_evidence(explicit, index)] == [10, 64]
