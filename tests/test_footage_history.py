from types import SimpleNamespace

import pytest

from trackrag import footage_history as history


def row(fid, x, *, tid=7, name="Bus", y=30, rec="drive", source="predicted"):
    return {"recording_id": rec, "source": source, "frame_id": fid, "track_id": tid,
            "class_name": name, "bbox_xyxy": [x, y, x + 20, y + 12]}


def question(words, cutoff=5):
    return {"question_id": "test", "recording_id": "drive", "query_frame": cutoff, "question": words}


def clock():
    return SimpleNamespace(frame_ids=[1, 3, 5, 7], recording_id="drive", get_frame=lambda f:
                           {"timestamp_s": 60 + (f - 1) * .5, "width": 100, "height": 80})


def evidence(rows):
    return [{"evidence_id": "E1", "interval":{"recording_id": "drive", "source": "predicted", "track_id": 7,
             "start_frame": rows[0]["frame_id"], "end_frame": rows[-1]["frame_id"], "rank": 1, "score": 1.},
             "images": [{"frame_id": r["frame_id"], "bbox_xyxy": r["bbox_xyxy"], "image_path": "evidence/7.png"} for r in rows]}]


@pytest.mark.parametrize(("text","expected"), [
    ("When does the bus first appear?", ("first", "bus")),
    ("When did track #7 enter the frame?", ("first",7)),
    ("When was track 7 last seen?", ("last",7)),
    ("When did track 7 leave the scene?", ("last",7)),
    ("Which direction did track 7 move across the image?", ("motion",7)),
    ("How did track 7 move across the image?", ("motion",7)),
    ("When does the red bus first appear?", None),
    ("How fast does track 7 move?", None),
    ("When does the bus first appear tomorrow?", None),
])
def test_only_explicit_history_questions_use_detector_endpoints(text,expected):
    assert history.intent(text) == expected


def test_suggestions_at_each_observation_cannot_depend_on_future_observations():
    frames = [{"id": f, "time": f / 10} for f in (1,3,5,7)]
    rows = [row(1,5),row(3,20),row(5,45,tid=9),row(7,30)]
    full = history.timeline_context(frames,rows)
    assert full[1]["current"]["first_frame"] == full[1]["current"]["last_frame"] == 1
    assert full[5]["recent"]["id"] == 7 and full[5]["recent"]["last_frame"] == 3
    assert full[7]["recent"]["id"]==9
    for r in frames:
        cutoff = r["id"]
        assert history.timeline_context([f for f in frames if f["id"]<=cutoff],
                                        [t for t in rows if t["frame_id"]<=cutoff])[cutoff] == full[cutoff]


def test_retrieval_includes_only_chosen_recording_predicted_boxes_and_original_cutoff():
    rows = [row(1,5),row(3,20),row(7,30),row(3,20,rec="other",tid=9),row(3,26,source="human",tid=10)]
    groups = {}
    for r in rows:
        groups.setdefault((r["recording_id"],r["source"],r["track_id"]),[]).append(r)
    index = SimpleNamespace(histories=groups,recording=clock())
    q = question("When does the bus first appear?")
    assert history.intervals(q,index)[0]["end_frame"]==1
    last = history.intervals(question("When was track 7 last detected?"),index)
    assert [(g["track_id"],g["start_frame"],g["end_frame"]) for g in last] == [(7,3,3)]
    assert history.intervals(question("How did track 7 move across the image?"),index)[0]["end_frame"]==3
    assert history.intervals(question("When was track 9 last seen?"),index)==[]
    assert history.intervals({**q,"recording_id":"different"},index)==[]


def test_first_already_present_and_last_lost_are_distinguished_from_physical_entry_or_exit():
    rows = [row(1,4),row(3,14),row(7,50)]
    first = history.answer(question("When did track 7 enter the scene?"),rows,evidence(rows[:1]),clock())
    last = history.answer(question("When did track 7 exit the scene?"),rows,evidence(rows[1:2]),clock())
    assert "already tracked at the start" in first["answer"] and "0:00.0 (frame 1)" in first["answer"]
    assert "missed detection" in last["answer"] and last["cited_evidence_ids"] == ["E1"]
    still = history.answer(question("When was track 7 last seen?",3),rows,evidence(rows[1:2]),clock())
    assert "still tracked at the selected moment" in still["answer"]


def test_box_motion_requires_the_two_authentic_endpoint_boxes_and_two_separate_observations():
    rows=[row(1,45,y=45),row(5,24,y=13),row(7,72,y=4)]
    q=question("How did track 7 move across the image?")
    answer=history.answer(q,rows,evidence(rows[:2]),clock())
    assert answer["cited_evidence_ids"]==["E1"]
    assert "shifted left and up" in answer["answer"] and "moving camera" in answer["answer"]
    assert history.answer(q,rows,evidence(rows[:1]),clock()) is None
    bad=evidence(rows[:2]);bad[0]["images"][-1]["bbox_xyxy"]=[2,2,8,8]
    assert history.answer(q,rows,bad,clock()) is None
    single=history.answer(q,rows[:1],evidence(rows[:1]),clock())
    assert single["answer_status"]=="insufficient_evidence"


def test_multiple_bus_tracks_request_clarification_and_an_unknown_track_never_invents_an_observation():
    rows=[row(1,4),row(3,15,tid=10)]
    answer=history.answer(question("When does the bus first appear?"),rows,evidence(rows[:1]),clock())
    assert answer["answer_status"]=="insufficient_evidence" and "track 7, track 10" in answer["answer"]
    assert history.answer(question("When was track 999 first detected?"),rows,[],clock()) is None


def test_changed_class_applies_only_to_the_displayed_observation():
    rows = [row(1,4,name="Car"),row(5,50,name="Bus")]
    answer = history.answer(question("How did track 7 move across the image?"),rows,evidence(rows),clock())
    assert "labelled bus in the paused frame" in answer["answer"]
    assert "tracked bus" not in answer["answer"]

