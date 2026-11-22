import json

import numpy as np
import pytest

from trackrag.dropoff.spatial_confidence import HEADS, choose, features, fit
from tools.dropoff_pack_matched import json_archive
from tools.dropoff_spatial_confidence import score


def candidate(x,peak=.08):
    return {"status":"geometry","destination_point_xy":[x,60],"destination_box_xyxy":[x-8,52,x+8,68],
            "peak_mass":peak,"point_road_probability":.95,"eligible_grid_cells":800}


def test_calibrator_returns_only_a_supplied_geometry_and_reloads_exactly():
    examples=[{HEADS[0]:candidate(30+i,.04+i*.009),HEADS[1]:candidate(66-i,.12-i*.008)} for i in range(8)]
    design=[row for ex in examples for row in features(ex,(100,100))]
    trained=fit(design,[i%3==1 for i in range(len(design))])
    new={HEADS[0]:candidate(40),HEADS[1]:candidate(57)}
    result=choose(trained,new,(100,100))
    assert result==choose(json.loads(json.dumps(trained)),new,(100,100))
    assert result==choose(trained,{k:{**v,"target":{"destination_point_xy":[3,3]}} for k,v in new.items()},(100,100))
    assert result["head"] in new and set(result["mask_hit_probabilities"])==set(HEADS)
    assert all(0<value<1 for value in result["mask_hit_probabilities"].values())
    empty=choose(trained,{name:{"status":"no_visible_road"} for name in HEADS},(100,100))
    assert empty=={"head":None,"mask_hit_probabilities":{},"meets_selected_threshold":False}
    with pytest.raises(ValueError,match="original|candidate"):
        choose(trained,{HEADS[0]:candidate(102)},(100,100))
    with pytest.raises(ValueError,match="Calibration"):
        fit(np.ones((3,27)),[1,1,1])


def test_matched_evaluation_refuses_unsealed_qwen_before_opening_participant_annotations(tmp_path):
    from types import SimpleNamespace
    audit=tmp_path/"metadata.json"
    audit.write_text("{}")
    args=SimpleNamespace(bank=audit,predictions=audit,deployed_run=tmp_path/"unsealed",
                         references=tmp_path/"must-never-be-opened.jsonl")
    with pytest.raises(ValueError,match="sealed Qwen"):
        score(args)


def test_publication_preserves_paired_hits_and_full_geometry_denominator(tmp_path,monkeypatch):
    from PIL import Image
    from tools import dropoff_publish_spatial_confidence as publish_study
    from tools.dropoff_spatial_confidence import PROTOCOL_HASH
    from trackrag.dropoff.data import file_sha256
    from zipfile import ZipFile
    def put(path,contents):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(contents),encoding="utf8")
    fresh=tmp_path/"fresh"
    cal=tmp_path/"cal"
    deployed=fresh/"deployed-trained-12"
    request=fresh/"inputs"/"requests.jsonl"
    put(request,{"id":"example"})
    refs=fresh/"corpus"/"references.jsonl"
    put(refs,{"fixture":True})
    put(cal/"calibrator.json",{"example":True})
    mask=fresh/"mask.png"
    Image.new("L",(100,100),255).save(mask)
    digest=file_sha256(mask)
    size={"width":100,"height":100}
    case={"case_id":"example","request":{"text":"use the painted lane","scene":{"frame":size}},
          "target":{"destination_box_xyxy":[30,52,46,68]},"source":{"scene_token":"group1","source_index":1,"mask_path":str(mask),
          "image_sha256":digest,"mask_sha256":digest}}
    for fname in ("run.json","settings.json"):
        put(deployed/fname,{"fixture":True})
    put(deployed/"outcomes.jsonl",{"request_id":"example","ok":True,"result":{"destination_point_xy":None}})
    for name in HEADS:
        for fname in ("run.json","outcomes.jsonl"):
            put(fresh/("normal-reload-"+name)/fname,{"fixture":True})
    req_digest=file_sha256(request)
    cand={HEADS[0]:candidate(38),HEADS[1]:candidate(80)}
    cpu={"requests_sha256":req_digest,"calibrator_sha256":file_sha256(cal/"calibrator.json"),"conditions":dict.fromkeys(HEADS,"fixture"),
         "rows":[{"request_id":"example","candidates":cand,"choice":{"head":HEADS[0],"meets_selected_threshold":False}}]}
    cpu_path=fresh/"cpu-predictions.json"
    put(cpu_path,cpu)
    meta={"protocol_sha256":PROTOCOL_HASH,"request_file_sha256":req_digest,"reference_file_sha256":file_sha256(refs),
          "source_scenes":["group1"],"overlap_counts":{"scene":0}}
    put(fresh/"manifest.json",meta)
    put(fresh/"normal-reload-agreement.json",{"all_choice_and_selective_decisions_reproduced":True,"models":{},
         "cpu_prediction_sha256":file_sha256(cpu_path)})
    put(fresh/"road-support.json",{})
    (fresh/"acquire.py").write_text("# software fixture",encoding="utf8")
    put(cal/"cross-fitted-scores.json",{"metrics":{},"selected_head_counts":{}})
    q_digest=file_sha256(deployed/"outcomes.jsonl")
    queue={"state":"complete","end_utc":"done","outcomes_sha256":q_digest,"requests_sha256":req_digest,
           "frozen_comparison_sha256":PROTOCOL_HASH,"run_sha256":file_sha256(deployed/"run.json")}
    put(fresh/"deployed-comparison.json",queue)
    from trackrag.vision.geometry import iou
    result={name:{"hit":True,"box_iou":iou(cand[source]["destination_box_xyxy"],case["target"]["destination_box_xyxy"])}
            for name,source in ((HEADS[0],HEADS[0]),(HEADS[1],HEADS[1]),("selected",HEADS[0]))}
    result["selected_confidence_030"]={"hit":False,"box_iou":0.}
    def expected_metrics(value):
        return {"cases":1,"mask_hits_all":int(value["hit"]),"proposals":int(value["hit"]),
                "operational_failures":0,"mean_coarse_iou_all_missing_zero":value["box_iou"]}
    scored={"protocol_sha256":PROTOCOL_HASH,"predictions_sha256":file_sha256(cpu_path),"calibrator_sha256":cpu["calibrator_sha256"],
            "deployed_comparison":{"outcomes_sha256":q_digest,"requests_sha256":req_digest,
                "primary_rows":[{"request_id":"example","point_in_public_mask":False,"destination_provided":False,"coarse_box_iou":0.}],
                "metrics":expected_metrics({"hit":False,"box_iou":0.})},
            "per_case":[{"request_id":"example","result":result}],"metrics":{k:expected_metrics(v) for k,v in result.items()},"selected_head_counts":{}}
    score_file=fresh/"scores.json"
    put(score_file,scored)
    monkeypatch.setattr(publish_study,"load_cases",lambda *a,**kw:[case])
    out=tmp_path/"summary.json";bundle=tmp_path/"evidence.zip"
    receipt=publish_study.publish(fresh,cal,score_file,out,bundle)
    paired=receipt["independent"]["paired_native_mask_hits_versus_deployed"]
    assert paired["selected"]["only_dense_hit"]==1 and paired["selected_confidence_030"]["neither_hit"]==1
    assert receipt["evidence_zip"]["sha256"]==file_sha256(bundle)
    with ZipFile(bundle) as z:
        assert all(n.endswith((".json",".jsonl")) for n in z.namelist())


def test_deterministic_hash_inventory_has_no_photos_and_blocks_home_paths(tmp_path):
    from zipfile import ZipFile
    entries={"inputs/test.jsonl":b'{"id":"a"}\n',"eval.json":b'{"count":1}'}
    first=tmp_path/"a.zip"
    second=tmp_path/"b.zip"
    assert json_archive(first,entries)==json_archive(second,dict(reversed(list(entries.items()))))
    with ZipFile(first) as z:
        items=json.loads(z.read("inventory.json"))
        assert set(items)==set(entries)
        assert z.read("inputs/test.jsonl")==entries["inputs/test.jsonl"]
    with pytest.raises(FileExistsError):
        json_archive(first,entries)
    with pytest.raises(ValueError,match="home"):
        json_archive(tmp_path/"home.zip",{"x.json":b'{"path":"C:/Users/Example/private"}'})
    with pytest.raises(ValueError,match="relative JSON"):
        json_archive(tmp_path/"image.zip",{"original.jpg":b"image"})
