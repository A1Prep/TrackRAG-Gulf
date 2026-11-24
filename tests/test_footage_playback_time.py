import json

import pytest

from trackrag.answering import _temporal_context, validate_answer
from trackrag.evidence_time import referenced_track_ids
from trackrag.playback_time import playback_scope, requested_playback_times
from trackrag.retrieval_temporal import build_index, rank_evidence


class OriginalFrames:
    recording_id = "uneven"
    frame_ids = [4, 15, 39, 62]
    times = {4: 10.0, 15: 10.4, 39: 12.1, 62: 13.0}

    def get_frame(self, fid):
        return {"recording_id": self.recording_id, "frame_id": fid, "width": 100, "height": 70,
                "timestamp_s": self.times[fid], "image_path": f"frame-{fid}.jpg"}

    def iter_frames(self):
        return (self.get_frame(fid) for fid in self.frame_ids)


def query(words, stop=62):
    return {"question_id": "public", "question": words, "query_frame": stop, "recording_id": "uneven"}


@pytest.mark.parametrize("text, times", [
    ("Compare 0:00.4 and 00:02.1.", [(False, .4), (False, 2.1)]),
    ("At two seconds, what's overhead?", [(False, 2.0)]),
    ("At the 3-second mark, what's overhead?", [(False, 3.0)]),
    ("What was there two seconds ago?", [(True, -2.0)]),
    ("What changed over the last two seconds?", [(True, -2.0), (True, 0)]),
    ("What happened between 300 ms and 2 seconds?", [(False, .3), (False, 2)]),
    ("Compare the bus at 7 seconds and 14 seconds.", [(False, 7), (False, 14)]),
    ("Compare the bus at 7 and 14 seconds.", [(False, 7), (False, 14)]),
    ("At 1:02:03.5 what was visible?", [(False, 3723.5)]),
    ("Did the driver stop for two seconds?", []),
    ("A pickup next to a blue bus", []),
])
def test_real_clock_language_has_distinct_times_and_durations(text, times):
    assert requested_playback_times(text) == times


def test_uses_exact_original_timestamps_and_declines_a_missing_sample():
    source = OriginalFrames()
    assert playback_scope("2.6 seconds ago", source, 62)["moments"] == [
        {"requested_seconds": .4, "frame_id": 15, "frame_seconds": .4}]
    assert playback_scope("Two seconds ago", source, 62)["status"] == "unavailable"
    assert playback_scope("What will I see at 0:00.4?", source, 62)["moments"][0]["frame_id"] == 15
    assert playback_scope("What changed between 0:00.4 and 0:02.1?", source, 62)["moments"][-1]["frame_id"] == 39
    for words in ("At 00:04 what happened?", "Four seconds ago", "What will I see next?"):
        assert playback_scope(words, source, 62)["status"] == "outside_review_window"
    assert playback_scope("What will I see at 0:00.4 and in the next frame?", source, 62)["status"] == "outside_review_window"
    assert playback_scope("At 0:00.4 (frame 39), is it visible?", source, 62)["status"] == "conflicting_frame_and_time"


def test_later_search_hint_only_identifies_absolute_timestamps():
    original = OriginalFrames()
    absolute = playback_scope("How many buses were there at 12 seconds?", original, 62)
    assert absolute["status"] == "outside_review_window"
    assert absolute["later_absolute_seconds"] == 12
    assert absolute["moments"] == []
    for words in ("How many cars were there 12 seconds ago?", "How many cars are there 12 seconds from now?"):
        result = playback_scope(words, original, 62)
        assert result["status"] == "outside_review_window"
        assert "later_absolute_seconds" not in result


def test_clock_comparison_supplies_both_originals_and_future_request_cannot_use_past_frames():
    source = OriginalFrames()
    observations = [{"recording_id": "uneven", "track_id": 7, "frame_id": f, "source":"predicted",
                     "class_name":"Bus", "bbox_xyxy":[3, 10, 13, 20]} for f in source.frame_ids]
    idx = build_index(source, observations)
    result = rank_evidence(query("Is the bus visible at 0:00.4 and 0:02.1?"), idx)
    assert {r["start_frame"] for r in result if r["track_id"] is None} == {15, 39}
    assert all(r["end_frame"]<=39 for r in result)
    assert rank_evidence(query("Is the bus visible at 0:02.1?", 15), idx) == []
    assert rank_evidence(query("What will I see three seconds from now?", 15), idx) == []


def test_hosted_answer_must_cite_both_exact_clock_frames_and_cannot_swap_them(tmp_path):
    q = query("Is the bus there at 0:00.4 and 0:02.1?")
    resolved = playback_scope(q["question"], OriginalFrames(), q["query_frame"])
    (tmp_path/"run.json").write_text(json.dumps({"first_frame_id":4, "configuration":{"retrieval":"trackrag.retrieval_temporal"},
                                         "playback_scope":{"public":resolved}}))
    scope = _temporal_context(tmp_path,q)
    assert scope["required_cited_frames"] == [15,39]
    views = [{"evidence_id":e,"interval":{"track_id":None,"source":"predicted"},"images":[{"frame_id":f}]}
             for e,f in [("a",15),("b",39)]]
    views[0]["temporal_scope"]=scope
    def checked(text, ids):
        return validate_answer({"answer_status":"answered","answer":text,"cited_evidence_ids":ids},views,q)["answer_status"]
    assert checked("Only at 0:02.1 (frame 39).",["a","b"]) == "answered"
    assert checked("Only at 0:02.1 (frame 15).",["a","b"]) == "insufficient_evidence"
    assert checked("At 0:03 (frame 39) there is a bus.",["a","b"]) == "insufficient_evidence"
    assert checked("Frame 39 at 0:00.4 shows a bus.",["a","b"]) == "insufficient_evidence"
    assert checked("It appears in the later image.",["b"]) == "insufficient_evidence"


def test_track_identity_at_requested_time_cannot_be_borrowed_from_another_frame():
    q=query("Is track 5 a bus at 0:00.4?")
    moment=playback_scope(q["question"], OriginalFrames(), q["query_frame"])
    scene={"evidence_id":"p","interval":{"track_id":None,"source":"predicted"},
           "images":[{"frame_id":15,"observations":[]}],
           "temporal_scope":{"required_cited_frames":[15], "boundary":None, "playback_scope":moment}}
    other={"evidence_id":"t","interval":{"track_id":5,"source":"predicted"},
           "images":[{"frame_id":39,"observations":[{"track_id":5}]}]}
    def assessed(words):
        response={"answer_status":"answered","answer":words,"cited_evidence_ids":["p","t"]}
        return validate_answer(response,[scene,other],q)["answer_status"]
    assert assessed("Track 5 is a bus at 0:00.4.") == "insufficient_evidence"
    assert assessed("Track 5 is not observed at 0:00.4.") == "answered"
    scene["images"][0]["observations"].append({"track_id":5})
    assert assessed("Track 5 is a bus at 0:00.4.") == "answered"


def test_missing_tracker_alias_is_parsed_instead_of_using_an_unrelated_box():
    assert referenced_track_ids("Is tracker ID 9999 this pickup?") == [9999]
    assert referenced_track_ids("What are the tracker's IDs 14 and 23?") == [14, 23]


def test_untimed_local_track_hint_uses_current_identity_and_current_frame_for_counts():
    source=OriginalFrames()
    observations=[{"recording_id":"uneven","track_id":tid,"frame_id":fid,"source":"predicted",
                   "class_name":"Bus","bbox_xyxy":[9,12,20,24]} for tid,fid in [(17,15),(23,39),(23,62)]]
    index=build_index(source,observations)
    side=rank_evidence(query("Which side of the frame is track 23 on?",39),index)
    assert {(r["track_id"],r["start_frame"],r["end_frame"]) for r in side}=={(23,39,39),(None,39,39)}
    counts=rank_evidence(query("How many buses are detected?",39),index)
    assert any(r["track_id"] is None for r in counts)
    assert all(r["start_frame"]==r["end_frame"]==39 for r in counts)
    timed=rank_evidence(query("Which side of the frame is track 23 on at 0:02.1?"),index)
    assert {(r["track_id"],r["end_frame"]) for r in timed}=={(None,39),(23,39)}
