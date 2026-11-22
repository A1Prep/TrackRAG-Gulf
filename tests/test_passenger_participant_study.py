"""Artificial photographs, artificial intentions and explicitly marked software participants only."""
from copy import deepcopy
from pathlib import Path
import json

from PIL import Image
import pytest

from evaluation.grounding import _pinned_mask, load_replies
from tools import passenger_participant_study as study


FIXTURE_ID = "AUTOMATED SOFTWARE FIXTURE"
WHEN = "2026-01-01T10:00:00Z"


def fixture_bank(folder):
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(3):
        photo=folder/f"fixture-{i}.jpg"
        image=Image.new("RGB",(96,60),(45,60+40*i,82))
        image.save(photo)
        rows.append({"request_id":f"fixture-{i}", "text":"Fixture <marker> only; identify the designated patch.",
                     "dialogue":[] if i!=2 else [{"role":"user","text":"Clarify the mocked location.",
                                                 "recording_id":"fixture-recording","frame_id":i}],
                     "input_mode":"typed","scene":{"frame":{"recording_id":"fixture-recording","frame_id":i,
                     "width":96,"height":60,"timestamp_s":i/10,"image_path":photo.name},
                     "tracks":[],"source":"predicted","provenance":{"image_sha256":study.digest(photo.read_bytes())}}})
    bank=folder/"requests.jsonl"
    study.write_lines(bank,rows)
    return bank


def mock_export(bank, *, indices=(0,1,2)):
    bank_hash,rows,_=study.plain_source(bank,_fixture=True)
    out={"kind":"passenger_preproposal_return","version":1,"input_bank_sha256":bank_hash,"participant_id":FIXTURE_ID,
         "exported_utc":WHEN,"source":"software_fixture","software_fixture":True,"human_attested":False,
         "declaration":study.ATTEST,"completed":[]}
    for i in indices:
        case=rows[f"fixture-{i}"]
        missing=i==1
        out["completed"].append({**{k:case[k] for k in study.BINDINGS},"participant_id":FIXTURE_ID,
                "reviewed_utc":WHEN,"input_bank_sha256":bank_hash,"choice":"landmark_missing" if missing else "drawn_area",
                "polygon_xy":None if missing else [[7,9],[60,9],[60,20],[24,20],[24,47],[7,47]],
                "point_xy":None if missing else [12,36],"answer":"Artificial clarification for the software fixture.",
                "rejection":"Artificial correction for the software fixture."})
    return out


def test_blank_pack_has_only_originals_and_input_binding(tmp_path):
    bank=fixture_bank(tmp_path/"source"); package=tmp_path/"pack"
    with pytest.raises(ValueError,match="44-case"):
        study.prepare(bank,package)
    result=study.prepare(bank,package,_fixture=True)
    assert result["original_jpegs"]==3 and result["participant_returns"]==0 and result["software_fixture"]
    verified=study.verify_pack(package,_fixture=True)
    assert verified["verified_original_photographs"]==3 and verified["form_sha256"]==result["form_sha256"]
    with pytest.raises(ValueError,match="version, source"):
        study.verify_pack(package)
    page=(package/"review.html").read_text()
    registry=json.loads(page.split('id="study-data">')[1].split("</script>")[0])
    assert registry["rows"][0]["text"].startswith("Fixture <marker>") and "<marker>" not in page
    allowed=set(study.BINDINGS)|{"text","dialogue","width","height"}
    assert all(set(row)==allowed for row in registry["rows"])
    assert "tracks" not in json.dumps(registry) and "target" not in json.dumps(registry) and FIXTURE_ID not in page
    with pytest.raises(ValueError,match="previous forms"):
        study.prepare(bank,package,_fixture=True)
    extra=package/"unexpected-review.json";extra.write_text('{"artificial":true}')
    with pytest.raises(ValueError,match="must contain only"):study.verify_pack(package,_fixture=True)
    extra.unlink()
    record_file=package/"blank-receipt.json";record=record_file.read_bytes()
    tampered=json.loads(record);tampered["model_proposals_shown"]=True;study.json_file(record_file,tampered)
    with pytest.raises(ValueError,match="construction receipt"):study.verify_pack(package,_fixture=True)
    record_file.write_bytes(record)
    next((package/"images").glob("*.jpg")).write_bytes(b"different")
    with pytest.raises(ValueError,match="photograph changed"):
        study.verify_pack(package,_fixture=True)


def test_real_evaluator_reads_rasterized_concave_region_and_both_reply_kinds(tmp_path):
    bank=fixture_bank(tmp_path/"source")
    file=tmp_path/"artificial.json"; study.json_file(file,mock_export(bank))
    ids=tmp_path/"ids.jsonl";study.write_lines(ids,[{"case_id":"fixture-0"},{"case_id":"fixture-1"}])
    out=tmp_path/"review"
    got=study.intake(bank,[file,file],out,unresolved_ids=ids,_fixture=True)
    assert got["returned_cases"]==3 and got["genuine_human_cases"]==0 and got["participants"]==1
    assert got["registered_replies"]==6 and got["drawn_pngs"]==2 and got["approved_task2_references"]==0
    p="software-fixture-"
    intended=list(map(json.loads,(out/(p+"intended.jsonl")).read_bytes().splitlines()))
    mask=_pinned_mask(out,intended[0],"intended_mask","fixture")
    assert mask.shape==(60,96) and mask[36,12] and not mask[35,40]
    assert intended[1]=={"case_id":"fixture-1","category":"missing_reference","intended_mask":None,"intended_mask_sha256":None}
    region,=map(json.loads,(out/(p+"task2-region-candidates.jsonl")).read_bytes().splitlines())
    assert region["case_id"]=="fixture-2" and region["destination_box_xyxy"]==[7,9,61,48]
    assert region["destination_point_xy"]==[12,36] and region["independent_reviewer_ids"]==[] and not region["review_complete"]
    assert region["mask_sha256"]==study.digest((out/region["mask_path"]).read_bytes())
    replies=load_replies(out/(p+"replies.jsonl"),{f"fixture-{i}":True for i in range(3)})
    assert {(k[0],r["kind"]) for k,r in replies.items()}=={(f"fixture-{i}",kind) for i in range(3) for kind in ("answer","rejection")}
    attest,=map(json.loads,(out/(p+"participant-attestations.jsonl")).read_bytes().splitlines())
    assert attest["human_attested"] is False and attest["export_sha256"]==study.digest(file.read_bytes())


@pytest.mark.parametrize("change,message",[
    ({"point_xy":[40,35]},"outside the actual drawing"), ({"point_xy":[10000,1]},"within bounds"),
    ({"polygon_xy":[[3,3],[3,3],[7,7]]},"repeats"), ({"polygon_xy":[[1,1],[3,3],[5,5]]},"no area"),
    ({"polygon_xy":[[0,0],[16,7],[2,16],[12,0],[12,14]]},"crosses itself"),
    ({"choice":"landmark_missing"},"missing landmark cannot carry"), ({"answer":""},"clarification answer"),
    ({"image_sha256":"X"},"binding"), ({"request_sha256":"X"},"binding"),
    ({"reviewed_utc":"2026-01-01T12:00:00Z"},"dated after"), ({"model_proposal":{}},"Unexpected fields"),
])
def test_invalid_geometry_attribution_and_time_leave_no_outputs(tmp_path,change,message):
    bank=fixture_bank(tmp_path/"source");fake=mock_export(bank,indices=(0,))
    fake["completed"][0].update(change)
    ret=tmp_path/"explicit-software-fixture.json";study.json_file(ret,fake)
    with pytest.raises(ValueError,match=message):study.intake(bank,[ret],tmp_path/"out",_fixture=True)
    assert not (tmp_path/"out").exists()


def test_merge_retains_conflicts_and_never_guesses_unresolved_ids(tmp_path):
    bank=fixture_bank(tmp_path/"source");uploaded=tmp_path/"software.json";fake=mock_export(bank,indices=(0,))
    study.json_file(uploaded,fake)
    bank_hash,rows,_=study.plain_source(bank,_fixture=True)
    with pytest.raises(ValueError,match="human declaration"):study.load_return(uploaded,rows,bank_hash,fixture=False)
    assert study.load_return(uploaded,rows,bank_hash,fixture=True)[1]["software_fixture"]
    seen=deepcopy(fake); seen["completed"][0]["answer"]="A different software answer."
    different=tmp_path/"software-second.json";study.json_file(different,seen)
    with pytest.raises(ValueError,match="Overlapping"):study.intake(bank,[uploaded,different],tmp_path/"conflict",_fixture=True)
    out=tmp_path/"no-classification"
    got=study.intake(bank,[uploaded],out,_fixture=True)
    assert got["returned_cases"]==1 and got["task2_candidate_regions"]==1
    assert not list(out.glob("*intended.jsonl")) and got["independent_unresolved_ids_supplied"] is None
    ids=tmp_path/"ids.jsonl";study.write_lines(ids,[{"case_id":"fixture-2"}])
    got=study.intake(bank,[uploaded],tmp_path/"missing",unresolved_ids=ids,_fixture=True)
    assert got["intended_rows"]==0 and got["unresolved_awaiting_human"]==["fixture-2"]
    ids.write_text('{"case_id":"fixture-0","target":"hidden labels are forbidden"}')
    with pytest.raises(ValueError,match="IDs only"):
        study.unresolved_cases(ids,rows)


def test_original_source_replacement_refused_before_intake(tmp_path):
    bank=fixture_bank(tmp_path/"source")
    (bank.parent/"fixture-0.jpg").write_bytes(b"modified bytes")
    with pytest.raises(ValueError,match="Original image differs"):study.intake(bank,[],tmp_path/"output",_fixture=True)
