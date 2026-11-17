"""Exercise the model-independent scoring paths most likely to overstate retrieval."""
from collections import Counter

import pytest

from evaluation.pretrained_retrieval import coverage, identity_matcher, interval_id, inventory, ranked_rows


def candidate(track, start, end, frames, source="annotated", kind="short observed window"):
    row = dict(recording_id="clip", source=source, track_id=track, start_frame=start, end_frame=end,
               observed_frames=frames, kind=kind)
    row["candidate_id"] = interval_id(row)
    return row


def reference(track, start, end):
    return dict(track_id=track, start_frame=start, end_frame=end)


def test_association_uses_competing_geometry_and_does_not_compare_numeric_ids():
    left, right = [0,0,20,20], [30,30,50,50]
    annotations = {("clip",1):{1:left}, ("clip",2):{1:right}}
    predictions = {("clip",1):{1:right}, ("clip",9):{1:left}}
    match = identity_matcher(predictions,annotations,"predicted")
    assert not match("clip",1,1,1,1)
    assert match("clip",9,1,1,1)
    # A globally tied prediction is not assigned an arbitrary gold identity.
    ambiguous = identity_matcher({("clip",9):{1:left}}, {("clip",1):{1:left},("clip",2):{1:left}}, "predicted")
    assert not ambiguous("clip",9,1,1,1)


def test_scene_alternative_does_not_grant_highlighted_identity_hit():
    refs = {"acceptable_evidence":[reference(9,1,10), reference(None,10,10)]}
    rows = [candidate(7,1,10,[1,10]), candidate(None,10,10,[10])]
    flags = coverage(rows,refs,{},identity_matcher({},{}, "annotated"))
    assert flags == dict(primary=True,identity=False,scene=True,bundle=True)


def test_bundle_requires_actual_observed_endpoints_and_explicit_scene():
    key = {"acceptable_evidence":[reference(9,1,10)]}
    rules = {"full_evidence_requires_observed_track_endpoints":[1,10],
             "full_evidence_requires_additional_exact_scene":12}
    match = identity_matcher({},{},"annotated")
    # Half of the inclusive gold span is sufficient for primary retrieval.
    first = candidate(9,1,5,[1,2,3,4,5])
    last = candidate(9,10,10,[10])
    scene = candidate(None,12,12,[])
    assert coverage([first],key,rules,match)["primary"]
    assert not coverage([first,scene],key,rules,match)["bundle"]
    assert not coverage([first,last],key,rules,match)["bundle"]
    assert coverage([first,last,scene],key,rules,match)["bundle"]
    assert not coverage([candidate(9,1,4,[1,4])],key,{},match)["primary"]


def test_inventory_and_rankings_are_bound_to_frames_cutoff_and_published_pool():
    q = dict(question_id="q",recording_id="clip",query_frame=3,question="Which vehicle?")
    short = candidate(9,1,3,[1,3])
    scene = candidate(None,3,3,[3,3],kind="full scene")
    raw = {"question":q,"candidates":[short,scene],"published_pool_candidate_ids":[short["candidate_id"]]}
    boxes = {("clip",9):{1:[],3:[]}}
    args = (q,"annotated",boxes,Counter({3:2}),{1,2,3})
    expanded,published = inventory(raw,*args)
    scene_rank = {**scene,"rank":1,"score":0.2}
    assert ranked_rows([scene_rank],expanded) == [scene]
    with pytest.raises(ValueError,match="unknown or duplicate"):
        ranked_rows([scene_rank],published)
    with pytest.raises(ValueError,match="unknown or duplicate"):
        ranked_rows([scene_rank,{**scene_rank,"rank":2}],expanded)
    for invalid,message in [
        ({**short,"observed_frames":[1,2,3]}, "observations"),
        ({**short,"end_frame":4}, "cutoff"),
        ({**short,"source":"predicted"}, "source"),
        ({**short,"candidate_id":"fake"}, "ID differs"),
    ]:
        with pytest.raises(ValueError,match=message):
            inventory({**raw,"candidates":[invalid,scene]},*args)


def test_byte_bound_saved_run_scores_only_frozen_ids_and_rejects_replaced_rankings(tmp_path,monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace
    from evaluation import pretrained_retrieval as scorer

    def write(name, payload, lines=False):
        path = tmp_path/name
        data = "".join(json.dumps(r)+"\n" for r in payload) if lines else json.dumps(payload)
        path.write_text(data,encoding="utf-8")
        return path

    def binding(path):
        return {"path":str(path),"sha256":scorer.fingerprint(path)}

    qs = [dict(question_id=i,recording_id="clip",query_frame=1,question="Which car?") for i in ["scored","manual"]]
    qp = write("example-questions.jsonl",qs,True)
    gt = [dict(recording_id="clip",source="annotated",track_id=1,frame_id=1,bbox_xyxy=[0,0,10,10])]
    labels = write("labels.txt",gt,True)
    monkeypatch.setattr(scorer,"open_recording",lambda root,name:SimpleNamespace(annotation_path=labels,frame_ids=[1]))
    monkeypatch.setattr(scorer,"load_annotated_tracks",lambda rec:gt)
    key = write("key.jsonl",[dict(question_id="scored",recording_id="clip",query_frame=1,answerable=True,
                                   acceptable_evidence=[reference(1,1,1)])],True)
    review = write("review.json",{"key":binding(key),"counts":{"all_questions":2,"scored_primary":1},
                 "evidence_inputs":[binding(qp),binding(labels)],"logic_inputs":[],
                 "dispositions":[{"question_id":q["question_id"],"disposition":q["question_id"]} for q in qs],
                 "target_kind_counts":{"identity_only":1}})
    vision = tmp_path/"vision"
    vision.mkdir()
    vision_summary = write("vision/summary.json",{"recordings":[{"recording_id":"clip","trackers":[]}]})
    c = candidate(1,1,1,[1])
    inv = write("candidates.jsonl",[{"condition":"annotated","question":q,"candidates":[c],
                               "published_pool_candidate_ids":[c["candidate_id"]]} for q in qs],True)
    ranked = [{**{f:c[f] for f in scorer.FIELDS},"rank":1,"score":1.0}]
    ranking = write("rankings.jsonl",[dict(condition="annotated",variant="published-task4",question=q,
                                    source="annotated",raw=ranked,reserved_current_scene=ranked) for q in qs],True)
    canon = hashlib.sha256(json.dumps(gt,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    write("summary.json",dict(inputs=[binding(qp),binding(vision_summary)],question_count=2,rankers=["lexical"],
          conditions=["annotated"],ranking_sha256=scorer.fingerprint(ranking),candidate_sha256=scorer.fingerprint(inv),
          track_inputs=[dict(condition="annotated",recording_id="clip",source="annotated",observation_count=1,
                             canonical_rows_sha256=canon)],
          candidate_counts=[dict(condition="annotated",question_id=q["question_id"],expanded_candidates=1,
                                 published_candidates=1) for q in qs]))
    report = scorer.score_run(tmp_path,vision,tmp_path,key,review)
    assert report["primary_denominator"] == 1 and len(report["per_case"]) == 2
    assert all(r["primary"] == r["identity"] == r["bundle"] == 1 and r["scene"] == 0 for r in report["scores"])
    assert "rankings" not in report["per_case"][1]
    with ranking.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError,match="Hash mismatch"):
        scorer.score_run(tmp_path,vision,tmp_path,key,review)


def test_switch_on_endpoint_cannot_certify_both_target_observations():
    a, b = [0,0,20,20], [40,40,60,60]
    gt = {("clip",3):{f:a for f in range(1,11)}, ("clip",8):{f:b for f in range(1,11)}}
    predicted = {("clip",4):{f:(a if f<10 else b) for f in range(1,11)}}
    scores = coverage(
        [candidate(4,1,10,list(range(1,11)),"predicted")],
        {"acceptable_evidence":[reference(3,1,10)]},
        {"full_evidence_requires_observed_track_endpoints":[1,10]},
        identity_matcher(predicted,gt,"predicted"),
    )
    assert scores["primary"] and not scores["bundle"]
