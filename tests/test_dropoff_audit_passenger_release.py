from types import SimpleNamespace

from PIL import Image
import pytest

from tools.dropoff_audit_passenger_release import compare_scorers, night_endpoints, public_endpoints, published_summary, run


def test_incomplete_run_refuses_before_opening_protocol_or_reference_material(tmp_path):
    ext=tmp_path/"batches"
    (ext/"trained-night").mkdir(parents=True)
    (ext/"trained-night"/"run.json").write_text("{}")
    args=SimpleNamespace(out=tmp_path/"audit.json",external=ext)
    with pytest.raises(ValueError,match="before reading references"):
        run(args)
    assert not args.out.exists()


def result(status,p=None,b=None):
    return {"status":status,"destination_point_xy":p,"destination_box_xyxy":b}


def outcome(status=None,p=None,b=None):
    return {"ok":True,"result":result(status,p,b)} if status else {"ok":False,"error":{"code":"timeout"}}


def test_publication_keeps_exact_outcome_hashes_without_private_case_records():
    row={"metrics":{"rows":2},"counts":{"initial_instruction":2},"outcomes_sha256":"exact-saved-output",
         "requests_sha256":"original-requests","primary_scorer_cross_check":{"case_count":2},
         "worker_pids":[333],"rows":[{"id":"private-case"}]}
    audit={"protocol_sha256":"protocol","runtime_freeze_sha256":"runtime","inference_completed_at":"finished",
           "all_four_outcome_files_sealed_and_verified_before_reference_read":True,"conditions":{"trained-night":row}}
    receipt=published_summary(audit,"original-audit")
    assert receipt["conditions"]["trained-night"]=={k:v for k,v in row.items() if k not in {"rows","worker_pids"}}
    assert receipt["audit_sha256"]=="original-audit"
    assert "Zero human reviews" in receipt["design"]["night"] and "No status" in receipt["design"]["public"]


def test_independent_metrics_keep_missed_cases_and_scores_outside_half_open_box():
    destination=result("proposed",[20,20],[10,10,30,30])
    refs={k:{"target":destination} for k in ("a","b","c")}
    refs.update(d={"target":result("needs_clarification")},e={"target":result("insufficient_visual_evidence")})
    inputs={k:{"scene":{"frame":{"width":100,"height":100}}} for k in refs}
    preds={"a":outcome("proposed",[20,20],[10,10,30,30]),
           "b":outcome("proposed",[30,20],[20,10,40,30]),
           "c":outcome(),"d":outcome("proposed",[20,20],[10,10,30,30]),"e":outcome("insufficient_visual_evidence")}
    m,rows=night_endpoints(refs,inputs,preds)
    assert m["rows"]==5 and m["clear"]==3 and m["unresolved"]==2
    assert m["status_matches"]==3 and m["false_proposals_on_unresolved"]==1
    assert m["strict_hits_all_clear"]==m["within_5pct_all_clear"]==1
    assert m["mean_coarse_iou_all_clear_missing_zero"]==pytest.approx((1+1/3)/3)
    assert m["operational_failures"]==1
    assert [r["strict_hit"] for r in rows]==[True,False,False,False,False]


def test_public_masks_use_original_coordinates_and_count_abstentions(tmp_path):
    path=tmp_path/"mask.png"
    im=Image.new("L",(10,10))
    im.putpixel((2,2),255)
    im.save(path)
    cases=[{"case_id":k,"source":{"mask_path":str(path)},
            "target":{"destination_box_xyxy":[20,20,30,30]},"request":{"scene":{"frame":{"width":100,"height":100}}}}
            for k in ("a","b","c")]
    preds={"a":outcome("proposed",[20,20],[20,20,30,30]),
           "b":outcome("needs_clarification"),"c":outcome()}
    m,rows=public_endpoints(cases,preds)
    assert m["rows"]==3 and m["proposals"]==1 and m["mask_hits_all"]==1 and m["operational_failures"]==1
    assert m["mean_coarse_iou_all_missing_zero"]==pytest.approx(1/3)
    assert "status_matches" not in m
    reference={"cases":3,"destinations_provided":1,"point_hits_all_requests":1,
               "mean_coarse_box_iou_all_requests_missing_as_zero":1/3,"unexpected_output_ids":[],
               "failures":{"timeout":1},"rows":[{"request_id":r["id"],"point_in_public_mask":r["point_in_original_mask"],
                                                  "coarse_box_iou":r["coarse_iou"]} for r in rows]}
    assert compare_scorers("trained-public",m,rows,reference)["case_count"]==3
    reference["rows"][0]["point_in_public_mask"]=False
    with pytest.raises(ValueError,match="original public mask"):
        compare_scorers("trained-public",m,rows,reference)
