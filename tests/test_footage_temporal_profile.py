"""Regressions for real-time bounds, boundary citations and unchanged lexical queries."""
import pytest

from trackrag.answering import validate_answer
from trackrag.answering import _temporal_context
from trackrag.retrieval import rank_evidence as lexical
from trackrag.retrieval_temporal import build_index, question_scope, rank_evidence


class Frames:
    recording_id = "street"
    frame_ids = [5, 7, 12, 22, 29, 38, 44]

    def get_frame(self, fid):
        if fid not in self.frame_ids:
            raise KeyError(fid)
        return dict(recording_id="street", frame_id=fid, timestamp_s=3 + (fid-5)/10,
                    width=100, height=80, image_path=f"{fid}.png")

    def iter_frames(self):
        return (self.get_frame(f) for f in self.frame_ids)


def track(fid, tid, label, x, *, source="predicted"):
    return dict(recording_id="street", frame_id=fid, track_id=tid, class_name=label,
                bbox_xyxy=(x, 20, x+10, 40), source=source)


def query(text, cutoff=38):
    return dict(question_id="never-read", recording_id="street", query_frame=cutoff, question=text)


def intervals(result):
    return [(r["track_id"], r["start_frame"], r["end_frame"]) for r in result]


@pytest.mark.parametrize("source", ["predicted", "annotated"])
def test_late_object_uses_the_requested_frame_and_source(source):
    rows = [track(5, 6, "Car", 75, source=source), track(38, 7, "Car", 70, source=source),
            track(38, 9, "Car", 5, source=source), track(44, 11, "Car", 80, source=source)]
    result = rank_evidence(query("What colour is the prominent SUV on the right near the end?"), build_index(Frames(), rows))
    assert intervals(result) == [(7,38,38), (None,38,38), (9,38,38)]
    assert {r["source"] for r in result} == {source}


def test_entry_uses_actual_birth_context_and_gap_before_observation():
    rows = [track(5, 2, "Bus", 28), track(7, 2, "Bus", 18),
            track(12, 3, "Bus", 0), track(22, 3, "Bus", 12), track(38, 3, "Bus", 22)]
    q = query("Which coach entered from the left after the filming started?")
    assert intervals(rank_evidence(q, build_index(Frames(), rows))) == [(3,12,38), (None,7,7), (None,5,5)]
    assert intervals(rank_evidence(q, build_index(Frames(), rows+[track(44, 14, "Bus", 1)]))) == [(3,12,38), (None,7,7), (None,5,5)]


def test_opening_interval_uses_timestamps_and_skips_future_objects():
    rows = [track(5, 5, "Bus", 3),track(12, 5, "Bus", 0),track(29, 6, "Bus", 80)]
    result = rank_evidence(query("Which bus moves out on the left during the opening second?"), build_index(Frames(), rows))
    assert all(r["end_frame"] <= 12 for r in result)
    assert result[0]["track_id"] == 5


def test_comparison_keeps_two_original_times_and_one_track():
    rows = [track(fid, 4, "Car", 22) for fid in Frames.frame_ids]
    q = query("Does the pickup appear larger at frame 29 than at the beginning?")
    assert intervals(rank_evidence(q, build_index(Frames(), rows))) == [(4,5,29), (None,5,5), (None,29,29)]
    assert rank_evidence(query("What was visible at frame 44?"), build_index(Frames(), rows)) == []
    assert rank_evidence(query("What was visible at frame 14?"), build_index(Frames(), rows)) == []


def test_untimed_search_preserves_task4_and_exit_sign_is_a_scene():
    idx = build_index(Frames(), [track(5,2,"Bus",32),track(7,2,"Bus",41)])
    q = query("What colour is the bus?")
    assert rank_evidence(q, idx) == lexical(q, idx)
    assert question_scope("Can you read the exit sign?")["boundary"] is None
    assert intervals(rank_evidence(query("What road name is on the exit sign?"), idx)) == [(None,38,38)]


def test_answer_requires_actual_cited_time_and_boundary_context():
    def shown(fid, x):
        return {"frame_id":fid,"bbox_xyxy":[x,12,x+10,25],"privacy":{"original_size_wh":[100,80]}}
    evidence = [{"evidence_id":"E1","interval":{"track_id":7},"images":[shown(12,0),shown(38,20)],
                 "temporal_scope":{"required_cited_frames":[],"boundary":"enter","tracked_extents":{"7":[12,38]}}},
                {"evidence_id":"E2","interval":{"track_id":None},"images":[{"frame_id":7}]}]
    q = query("Which van entered from the left?")
    def check(text, cites):
        return validate_answer({"answer_status":"answered","answer":text,"cited_evidence_ids":cites},evidence,q)["answer_status"]
    assert check("It is the blue van.", ["E1"]) == "insufficient_evidence"
    assert check("It is the blue van.", ["E1","E2"]) == "answered"
    assert check("The blue van entered at frame 12.", ["E1","E2"]) == "insufficient_evidence"
    assert check("It entered exactly two seconds in.", ["E1","E2"]) == "insufficient_evidence"
    assert check("The blue van was first detected in frame 38.", ["E1","E2"]) == "insufficient_evidence"
    assert check("Track 7 was last detected in frame 12.", ["E1","E2"]) == "insufficient_evidence"
    assert check("Track 4 was first detected in frame 12.", ["E1","E2"]) == "insufficient_evidence"
    assert check("The blue van was first detected in frame 12.", ["E1","E2"]) == "answered"
    evidence[0]["images"][0] = shown(12,45)
    assert check("The blue van entered from the left.", ["E1","E2"]) == "insufficient_evidence"
    evidence[0]["temporal_scope"].update(required_cited_frames=[38],boundary=None)
    assert check("The van at the end is blue.", ["E2"]) == "insufficient_evidence"


def test_opening_sky_and_seconds_never_include_the_later_cutoff(tmp_path):
    import json
    (tmp_path/"run.json").write_text(json.dumps({"first_frame_id":5,"configuration":{"retrieval":"trackrag.retrieval_temporal"}}))
    opening = query("What is the sky like in the opening frame?")
    assert _temporal_context(tmp_path, opening)["required_cited_frames"] == [5]
    idx = build_index(Frames(), [track(5,9,"Car",4),track(38,10,"Car",10)])
    for q in ("What do the first two seconds show?", "Which cars appear during the first two seconds?"):
        assert {r['end_frame'] for r in rank_evidence(query(q), idx)} <= {5,7,12,22}
    assert rank_evidence(query("What crossed the street in the first 0 seconds?"), idx) == []


def test_bridge_landmark_does_not_erase_track_for_boundary_question():
    rows = [track(12,11,"Bus",0),track(22,11,"Bus",15)]
    q = query("Which bus entered on the left under the bridge?")
    assert rank_evidence(q,build_index(Frames(),rows))[0]["track_id"] == 11


def test_boundary_can_inspect_a_repeated_minority_class_without_relabelling_history():
    rows=[track(f,11,"Motorbike" if f in {5,7} else "Car",x)
          for f,x in [(5,30),(7,26),(12,20),(22,10),(29,0)]]
    q=query("Which motorbike exited the left edge?")
    idx=build_index(Frames(),rows)
    assert intervals(rank_evidence(q,idx))[:2] == [(11,5,29),(None,38,38)]
    assert [r["class_name"] for r in idx.histories[("street","predicted",11)]] == [r["class_name"] for r in rows]
    false_positive=[dict(row, class_name="Car") if row["frame_id"]==7 else row for row in rows]
    assert all(r["track_id"] is None for r in rank_evidence(q,build_index(Frames(),false_positive)))


@pytest.mark.parametrize("text, source, expected", [
    ("Which type is annotated as track 19?", "predicted", "insufficient_evidence"),
    ("Which ground-truth ID belongs to this van?", "predicted", "insufficient_evidence"),
    ("What are the tracker’s IDs for these two vans?", "annotated", "insufficient_evidence"),
    ("Which class did the human labels assign?", "annotated", "answered"),
    ("What do the predicted tracks show?", "predicted", "answered"),
    ("Which van is visible?", "predicted", "answered"),
])
def test_source_specific_question_cannot_borrow_an_independently_assigned_id(text, source, expected):
    evidence=[{"evidence_id":"E1","interval":{"track_id":19,"source":source},
               "images":[{"frame_id":38,"observations":[{"track_id":19,"source":source}]}]}]
    raw={"answer_status":"answered","answer":"The van.","cited_evidence_ids":["E1"]}
    answer=validate_answer(raw,evidence,query(text))
    assert answer["answer_status"]==expected
    assert bool(answer["cited_evidence_ids"]) == (expected=="answered")


@pytest.mark.parametrize("verb,span,scene_frame", [("enter",[12,38],7),("exit",[7,22],38)])
def test_visible_boundary_identification_requires_distinct_past_or_later_pixel_evidence(verb,span,scene_frame):
    q=query("Which vehicle moved out through the right side?")
    context={"required_cited_frames":[],"boundary":verb,"identification_only":True,"tracked_extents":{"4":span}}
    evidence=[{"evidence_id":"E1","interval":{"track_id":4},"temporal_scope":context,
               "images":[{"frame_id":f,"bbox_xyxy":[15,12,35,32],"observations":[{"track_id":4}]} for f in span]},
              {"evidence_id":"E2","interval":{"track_id":None},"images":[{"frame_id":scene_frame}]}]
    def answer(text, ids=("E1","E2")):
        raw={"answer_status":"answered","answer":text,"cited_evidence_ids":list(ids)}
        return validate_answer(raw,evidence,q)["answer_status"]
    qualifier=" The footage does not establish a boundary crossing."
    assert answer("The visible candidate is a red delivery van on the right."+qualifier) == "answered"
    assert answer("The visible candidate is a red delivery van on the right."+qualifier,["E1"]) == "insufficient_evidence"
    assert answer("The visible candidate is a red van that entered at frame 12."+qualifier) == "insufficient_evidence"
    assert answer("It definitely crossed.") == "insufficient_evidence"
    evidence[1]["images"][0]["frame_id"] = span[0]
    assert answer("The visible candidate is a red van."+qualifier) == "insufficient_evidence"
