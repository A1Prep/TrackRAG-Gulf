from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from test_passenger_manual_offlane import provide_map
from test_passenger_roadside_evidence import reviewed
from trackrag.dropoff import curb_point_language,roadside_context,whole_bay


def bind_gravel(monkeypatch, packet):
    monkeypatch.setattr(roadside_context,"get_packet",lambda key,gravel_only=False:packet if gravel_only else None)


def test_normal_two_sentence_request_is_checkable_and_bound_to_its_own_words(reviewed,monkeypatch):
    request,photo,surface,_,packet,_=reviewed
    bind_gravel(monkeypatch,packet)
    request["text"]="Corner shop is my stop. I'd like to get off between the red minivan and the black hatchback."
    out=whole_bay.explicit_request(request)
    assert out["result"]["status"]=="proposed"
    assert out["provenance"]["language_policy"]==whole_bay.SPATIAL_DIALOGUE_POLICY
    assert whole_bay.verify_current(request,out)["archived_gravel_context_records"]==1
    with pytest.raises(ValueError):
        whole_bay.verify({**request,"text":request["text"].replace("red","blue")},out)
    old=deepcopy(out)
    old["provenance"]["language_policy"]=whole_bay.LANGUAGE_POLICY
    with pytest.raises(ValueError):
        whole_bay.verify(request,old)


def test_wrong_anchored_colour_yields_fast_source_verified_clarification(reviewed,monkeypatch):
    request,photo,surface,_,packet,_=reviewed
    bind_gravel(monkeypatch,packet)
    request["text"]="Corner shop is my stop. Could you drop me in the gap between the blue van and the black hatchback?"
    out=whole_bay.explicit_request(request)
    assert out["provenance"]["reason"]=="indexed_reference_unmatched"
    assert out["result"]["status"]=="needs_clarification" and not out["provenance"]["bay_choices"]
    assert whole_bay.verify_current(request,out)["archived_gravel_context_records"]==1
    changed=deepcopy(out)
    changed["provenance"]["gravel_evidence"]["signature"]="00"*64
    with pytest.raises(ValueError):
        whole_bay.verify_current(request,changed)


def test_historical_scoped_receipt_replays_under_its_own_language_policy(reviewed):
    request,photo,surface,_,packet,_=reviewed
    request["text"]="Let me out in the open gap between the red minivan and the black hatchback near the corner shop."
    older=whole_bay.roadside_outcome(request,photo,surface,None,None,gravel_packet=packet,strict=True)
    assert older["result"]["status"]=="proposed" and older["provenance"]["language_policy"]==whole_bay.LANGUAGE_POLICY
    assert whole_bay.verify(request,older)["decision_reproduced"]
    with pytest.raises(ValueError):
        whole_bay.verify_current(request,older)
    second=whole_bay.roadside_outcome(request,photo,surface,None,None,gravel_packet=packet,conversational=True)
    assert second["provenance"]["language_policy"]==whole_bay.INDEXED_DIALOGUE_POLICY
    assert whole_bay.verify(request,second)["decision_reproduced"]
    with pytest.raises(ValueError):
        whole_bay.verify_current(request,second)


def test_absent_geometry_is_replayed_from_the_authentic_recorded_surface(tmp_path,monkeypatch):
    path=tmp_path/"through-road.png"
    Image.new("RGB",(320,192),(100,100,100)).save(path)
    key=sha256(path.read_bytes()).hexdigest()
    scene={"source":"predicted","frame":{"recording_id":"empty-road","frame_id":1,"timestamp_s":0,
           "width":320,"height":192,"image_path":str(path)},"tracks":[],"provenance":{"image_sha256":key}}
    provide_map(monkeypatch,scene,np.full((192,320),13,dtype=np.uint8))
    request={"request_id":"none","dialogue":[],"input_mode":"typed","scene":scene,
             "text":"See the blue bin? That gravel area beside it is where I'd like to get out."}
    out=whole_bay.explicit_request(request)
    assert out["result"]["status"]=="needs_clarification" and not out["provenance"]["bay_choices"]
    assert out["provenance"]["index_geometry_checked"] is True
    assert whole_bay.verify_current(request,out)["offlane_mask_replayed"]
    changed=deepcopy(out)
    changed["provenance"]["surface_evidence"]["signature"]="00"*64
    with pytest.raises(ValueError):
        whole_bay.verify(request,changed)


def test_fragment_requires_two_vehicle_heads_and_checks_all_modifiers_against_the_source(reviewed,monkeypatch):
    request,photo,surface,pairs,packet,_=reviewed
    # Feed this independently signed fixture board through the usual roadside reader.
    monkeypatch.setattr(roadside_context,"get_packet",lambda key,gravel_only=False:None if gravel_only else packet)
    monkeypatch.setattr(roadside_context,"candidates",lambda *a,**kw:pairs)
    request["text"]="Between the red minivan and the black hatchback just here, please."
    out=whole_bay.explicit_request(request)
    assert out["provenance"]["language_policy"]==whole_bay.ABBREVIATED_PAIR_POLICY
    assert out["result"]["status"]=="needs_clarification" and len(out["provenance"]["bay_choices"])==1
    assert not out["provenance"]["independent_model_proposal"]
    assert whole_bay.verify_current(request,out)["archived_roadside_context_records"]==1
    changed={**request,"text":request["text"].replace("red","blue")}
    with pytest.raises(ValueError):
        whole_bay.verify(changed,out)
    withheld=whole_bay.explicit_request(changed)
    assert not withheld["provenance"]["bay_choices"]
    assert withheld["provenance"]["reason"]=="pictured_pair_fragment_unmatched"
    assert whole_bay.verify_current(changed,withheld)["decision_reproduced"]
    forged=deepcopy(out)
    forged["provenance"]["roadside_evidence"]["signature"]="00"*64
    with pytest.raises(ValueError):
        whole_bay.verify_current(request,forged)


@pytest.mark.parametrize("words",[
    "Between the white sedan and the blue van, which is moving?",
    "Between the car and the minivan on the left or on the right, please.",
    "Between the blue car and a child on the right.",
    "Between the trucks in the active travel lane.",
    "Before the blue van and after the red bus, please.",
    "I don't want to stop between the blue minivan and the red car.",
])
def test_abbreviated_pair_cannot_convert_a_question_or_partial_clause_into_a_location(words):
    assert curb_point_language.abbreviated_pair(words) is None
