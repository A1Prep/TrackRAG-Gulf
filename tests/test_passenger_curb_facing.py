"""Software-only facing-row geometry and refusal checks, not real scene accuracy."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from trackrag.dropoff import curb_facing, pair_geometry
from trackrag.dropoff.projected_pair import Ineligible


SIZE=(512,288)


def track(i,box,kind="Car"):
    return {"track_id":i,"bbox_xyxy":box,"class_name":kind}


@pytest.fixture
def scene():
    anchors=[track(1,[80,70,170,150]),track(2,[260,70,350,150])]
    mask=np.zeros((288,512),bool);mask[145:]=True
    sidewalk=np.zeros_like(mask);sidewalk[:145]=True
    examples=[{"anchors":[[80,70,170+i%3,150],[260+i%2,70,350,150]],"size":SIZE,
               "point":[214+i%3,159],"box":[195,148,238,176]} for i in range(30)]
    model=SimpleNamespace(pair_regression=pair_geometry.fit(examples))
    return anchors,mask,sidewalk,model


def estimate(scene,**kwargs):
    anchors,mask,sidewalk,model=scene
    return curb_facing.estimate(model,mask,SIZE,anchors,anchors,"between the two vehicles",**kwargs)


def test_equal_ground_contact_heights_allow_real_road_with_independent_sidewalk(scene):
    anchors,mask,sidewalk,model=scene
    patch=estimate(scene,sidewalk_mask=sidewalk)
    assert patch["orientation_geometry"]=="inner_sides_and_contact_band"
    assert patch["witness_kinds"]==["independent_sidewalk_semantics"]
    assert 170 < patch["point"][0] < 260
    assert patch["point"][1] >= 145
    curb_facing.validate_mask(patch,mask,SIZE,anchors,anchors,sidewalk_mask=sidewalk)
    assert curb_facing.reproduce(model,patch,mask,SIZE,anchors,anchors,"between the two vehicles",sidewalk_mask=sidewalk)


def test_road_and_generic_exterior_alone_never_certify_a_roadside(scene):
    with pytest.raises(Ineligible,match="no_independent_roadside_evidence"):
        estimate(scene)


def test_closed_raised_island_semantics_are_not_a_shared_curb(scene):
    anchors,mask,sidewalk,model=scene
    mask[:]=True;mask[125:145,180:250]=False
    sidewalk[:]=False;sidewalk[125:145,180:250]=True
    with pytest.raises(Ineligible,match="no_independent_roadside_evidence"):
        estimate(scene,sidewalk_mask=sidewalk)


def test_boundary_parking_paint_l_shape_is_independent_of_semantic_sidewalk(scene):
    image=Image.new("RGB",SIZE,(60,60,60))
    draw=ImageDraw.Draw(image)
    draw.line((190,146,240,146),fill="white",width=2)
    draw.line((190,146,190,157),fill="white",width=2)
    patch=estimate(scene,image=image)
    assert "visible_nonparallel_boundary_paint" in patch["witness_kinds"]


def test_parallel_highway_lane_paint_does_not_certify_parking(scene):
    anchors,mask,sidewalk,model=scene
    mask[:]=True
    image=Image.new("RGB",SIZE,(60,60,60))
    paint=ImageDraw.Draw(image)
    paint.line((180,130,180,190),fill="white",width=3)
    paint.line((245,130,245,190),fill="white",width=3)
    with pytest.raises(Ineligible,match="no_independent_roadside_evidence"):
        estimate(scene,image=image)


@pytest.mark.parametrize("colour",[(246,246,246),(120,120,120),(230,200,10)])
def test_uniform_pale_gray_or_yellow_pavement_cannot_invent_painted_lines(scene,colour):
    anchors,mask,sidewalk,model=scene
    mask[:]=False;mask[130:]=True
    with pytest.raises(Ineligible,match="no_independent_roadside_evidence"):
        estimate(scene,image=Image.new("RGB",SIZE,colour))


def test_two_tone_unmarked_pavement_is_not_a_parking_bay(scene):
    anchors,mask,sidewalk,model=scene
    image=Image.new("RGB",SIZE,(120,120,120))
    ImageDraw.Draw(image).rectangle((200,0,511,287),fill=(240,240,240))
    with pytest.raises(Ineligible,match="no_independent_roadside_evidence"):
        estimate(scene,image=image)


def test_visible_bay_front_paint_needs_a_distinct_vehicle_row_context(scene):
    anchors,mask,sidewalk,model=scene
    mask[:]=True
    image=Image.new("RGB",SIZE,(60,60,60));paint=ImageDraw.Draw(image)
    paint.line((185,176,240,176),fill="white",width=2)
    paint.line((185,160,185,176),fill="white",width=2)
    with pytest.raises(Ineligible,match="no_independent_roadside_evidence"):
        estimate(scene,image=image)
    tracks=[*anchors,track(3,[380,70,470,150])]
    patch=curb_facing.estimate(model,mask,SIZE,anchors,tracks,"between the two vehicles",image=image)
    assert patch["witness_kinds"]==["visible_nonparallel_parking_paint_with_row"]
    assert patch["aligned_row_context"]["additional_track_id"]==3
    curb_facing.validate_mask(patch,mask,SIZE,anchors,tracks,image=image)


def test_duplicate_detection_is_not_an_additional_parking_row_vehicle(scene):
    anchors,mask,sidewalk,model=scene
    assert curb_facing._row_context(anchors,[*anchors,track(3,[82,72,168,149])]) is None


@pytest.mark.parametrize("kind",["Pedestrian","Car","Motorbike","Unknown"])
def test_any_observed_object_can_block_a_proposed_patch(scene,kind):
    anchors,mask,sidewalk,model=scene
    occupied=[*anchors,track(7,[170,130,260,190],kind)]
    with pytest.raises(Ineligible,match="no_unoccupied_between_ground"):
        curb_facing.estimate(model,mask,SIZE,anchors,occupied,"between",sidewalk_mask=sidewalk)


def test_untracked_nonroad_sign_is_not_independent_roadside_evidence(scene):
    anchors,mask,sidewalk,model=scene
    image=Image.new("RGB",SIZE,(60,60,60));paint=ImageDraw.Draw(image)
    mask[135:150,200:225]=False
    paint.rectangle((200,135,224,149),fill="white")
    with pytest.raises(Ineligible,match="no_independent_roadside_evidence"):
        estimate(scene,image=image)


def test_occupancy_and_source_sidewalk_changes_invalidate_the_proposal(scene):
    anchors,mask,sidewalk,model=scene
    patch=estimate(scene,sidewalk_mask=sidewalk)
    altered=sidewalk.copy();altered[0,0]=False
    with pytest.raises(Ineligible,match="surface_evidence_changed"):
        curb_facing.validate_mask(patch,mask,SIZE,anchors,anchors,sidewalk_mask=altered)
    changed=[*anchors,track(9,patch["box"],"Pedestrian")]
    with pytest.raises(Ineligible,match="patch_occupied"):
        curb_facing.validate_mask(patch,mask,SIZE,anchors,changed,sidewalk_mask=sidewalk)


def test_another_admissible_road_point_cannot_replace_the_fitted_winner(scene):
    anchors,mask,sidewalk,model=scene
    patch=estimate(scene,sidewalk_mask=sidewalk)
    other=deepcopy(patch)
    other["point"][0]+=1
    with pytest.raises(Ineligible,match="fitted_winner_changed"):
        curb_facing.reproduce(model,other,mask,SIZE,anchors,anchors,"between the two vehicles",sidewalk_mask=sidewalk)


def test_changed_original_pixels_invalidate_visible_paint_evidence(scene):
    anchors,mask,sidewalk,model=scene
    image=Image.new("RGB",SIZE,(60,60,60));draw=ImageDraw.Draw(image)
    draw.line((190,146,240,146),fill="white",width=2);draw.line((190,146,190,157),fill="white",width=2)
    patch=estimate(scene,image=image)
    changed=image.copy();changed.putpixel((1,1),(0,0,0))
    with pytest.raises(Ineligible,match="surface_evidence_changed"):
        curb_facing.validate_mask(patch,mask,SIZE,anchors,anchors,image=changed)


def test_oblique_overlapping_x_ranges_keep_a_swept_contact_polygon():
    anchors=[track(1,[80,50,250,160]),track(2,[210,80,410,220])]
    geometry=curb_facing._geometry(anchors,SIZE)
    assert geometry["orientation_geometry"]=="oblique_contact_sweep"
    assert geometry["center_corridor_xyxy"]==[80.0,160.0,410.0,220.0]


def test_contradictory_road_and_sidewalk_predictions_are_rejected(scene):
    anchors,mask,sidewalk,model=scene
    sidewalk[150,200]=True
    with pytest.raises(Ineligible,match="contradictory_semantic_surfaces"):
        estimate(scene,sidewalk_mask=sidewalk)
