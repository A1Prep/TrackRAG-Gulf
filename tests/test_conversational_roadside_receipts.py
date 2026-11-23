from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from test_passenger_manual_offlane import provide_map
from test_passenger_roadside_evidence import reviewed
from trackrag.dropoff import roadside_context,whole_bay


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
