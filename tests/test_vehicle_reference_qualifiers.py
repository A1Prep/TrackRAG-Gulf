from copy import deepcopy
from hashlib import sha256
from io import BytesIO

import pytest
from PIL import Image

from trackrag.dropoff import bay_anchors, parking_evidence, vehicle_reference_qualifiers as qual


def parse(words, **kwargs):
    return qual.parse({"text":words}, **kwargs)


def car(track_id, box):
    return {"track_id":track_id,"box":box}


LEFT=car(1,[100,150,240,260])
RIGHT=car(2,[620,145,760,262])
WHITE=["white","suv","REAR","CURBSIDE"]
BLACK=["black","sedan","SIDE","NONE"]


def road_user(identity,kind,box,frame=1):
    return {"frame_id":frame,"track_id":identity,"class_name":kind,"bbox_xyxy":box}


def solve(words,specs,answers,photos=(),*,other=(),expanded=True):
    tracks=[road_user(s["track_id"],"Car",s["box"]) for s in specs]+list(other)
    request={"text":words,"scene":{"frame":{"width":1000,"height":600,"frame_id":1},"tracks":tracks}}
    ref=qual.parse(request,expanded=expanded)
    assert ref is not None
    return qual.resolve(request,ref,specs,answers,observations=photos,expanded=expanded)


def reading(spec,appearance,**ground):
    return {**spec,"appearance":appearance,**ground}


@pytest.mark.parametrize("words",[
    "Please stop behind the white car.",
    "Drop me between the two red Jeeps.",
    "Stop behind the white car or the red one.",
    "Stop beside the red Toyota and the silver van.",
    "Stop beside the car with a blue sticker and the truck.",
    "Can you take me to Dog Spot?",
    "How much does the red Jeep cost?",
    "Stop in the active traffic lane beside the red Kia.",
    "Drop me next to the bus stop.",
    "Choose a parking bay beside the black Toyota.",
    "Drop me behind the red Jeep unless the driver comes back.",
])
def test_ordinary_ambiguous_hazard_and_other_routes_stay_separate(words):
    assert parse(words) is None


def test_type_body_colour_livery_make_and_named_frontage_are_separate():
    ref=parse("Could you let me out behind the white SUV with the red design on its door?")
    assert (ref["relation"],ref["kind"],ref["color"])==("behind","suv","white")
    assert ref["qualifiers"]==[{"field":"livery","value":"design","colour":"red","part":"door"}]
    ref=parse("Next to the grey Mini outside Dog Spot, on its right.")
    assert (ref["side"],ref["ground_side"],ref["color"])==(None,"right","gray")
    assert ref["qualifiers"]==[{"field":"make","value":"mini"},{"field":"place","value":"dog spot","relation":"outside"}]
    ref=parse("Drop me beside the black sedan nearest the motorcycles, in the gravel gap.")
    assert ref["qualifiers"]==[{"field":"comparison","value":"motorcycle"},{"field":"ground","value":"gravel gap"}]


def test_a_side_inside_a_location_or_graphic_cannot_lift_the_wrong_vehicle():
    ref=parse("Stop behind the white SUV with the red graphic on the left side door.")
    assert ref["side"] is None and any(q["field"]=="unknown" for q in ref["qualifiers"])
    full=reading(LEFT,"white SUV with a red graphic on the door")
    assert not solve("Stop behind the white SUV with the red graphic on the left side door.",
                     [LEFT],[WHITE],[full])[0]
    ref=parse("Beside the silver Audi outside the pharmacy on the right.")
    assert ref["side"] is None and any(q["field"]=="unknown" for q in ref["qualifiers"])


@pytest.mark.parametrize(("words","actor","ground","relation"),[
    ("Next to the car on the left, on its right.","left","right","beside"),
    ("Beside the car on the right of it.",None,"right","beside"),
    ("To the left of the grey Mini.",None,"left","beside"),
    ("Behind the black Kia on the right.","right",None,"behind"),
    ("Behind the black Kia on its right.",None,"right","behind"),
    ("In front of the yellow Ford.",None,None,"front"),
])
def test_image_side_and_ground_side_stay_independent(words,actor,ground,relation):
    ref=parse(words)
    assert (ref["side"],ref["ground_side"],ref["relation"])==(actor,ground,relation)


@pytest.mark.parametrize("caption",[
    "white SUV with black graphics on the door",
    "white SUV with red graphics on the roof and a black door",
    "white SUV beside another car with a red door graphic",
    "white SUV with possibly red graphics on the door",
    "white SUV with a red logo on its bonnet",
])
def test_livery_must_belong_to_same_vehicle_colour_and_panel(caption):
    yes,boxes,msg=solve("Stop behind the white SUV with red graphics on its door.",
                        [LEFT],[WHITE],[reading(LEFT,caption)])
    assert not yes and boxes==[LEFT["box"]] and "red design on the door" in msg


def test_positive_livery_retains_original_reader_orientation_and_ground():
    raw=[["unknown","car","FRONT","BLOCKED"]]
    old=deepcopy(raw)
    yes,boxes,msg=solve("Stop behind the white SUV with red graphics on its door.",
                        [LEFT],raw,[reading(LEFT,"white SUV with a red decal on the door")])
    assert msg is None and boxes==[LEFT["box"]]
    assert yes==[(LEFT,["white","suv","FRONT","BLOCKED"])] and raw==old


def test_make_must_be_on_body_not_an_emblem_or_neighbour():
    ask="Stop behind the red Jeep."
    raw=[["red","suv","SIDE","LANE"]]
    for caption in ("red Toyota showing a Jeep logo","red SUV beside a Jeep","red SUV Jeep logo"):
        assert solve(ask,[LEFT],raw,[reading(LEFT,caption)])[0]==[]
    yes,_,msg=solve(ask,[LEFT,RIGHT],raw*2,[reading(LEFT,"red Jeep SUV"),reading(RIGHT,"red Mazda SUV")])
    assert [s["track_id"] for s,_ in yes]==[1] and msg is None
    yes,boxes,msg=solve(ask,[LEFT,RIGHT],raw*2,[reading(LEFT,"red Jeep SUV")])
    assert not yes and len(boxes)==2 and msg


def test_model_name_on_an_advert_is_not_the_identity():
    words="Behind the white Mini Cooper."
    bad=reading(LEFT,"white Mini SUV with a Cooper logo")
    assert not solve(words,[LEFT],[WHITE],[bad])[0]
    good=reading(LEFT,"white Mini Cooper SUV")
    assert solve(words,[LEFT],[WHITE],[good])[0]


@pytest.mark.parametrize(("track_id","box"),[
    (10,LEFT["box"]),(1,RIGHT["box"]),(1,[100,150,240,261])
])
def test_other_or_moved_crops_cannot_bind_brand(track_id,box):
    yes,boxes,msg=solve("Behind the white Toyota.",[LEFT],[WHITE],
                        [{"track_id":track_id,"box":box,"appearance":"white Toyota SUV"}])
    assert not yes and boxes==[LEFT["box"]] and "Toyota" in msg


def test_missing_or_close_but_across_road_frontage_does_not_bind():
    ask="Beside the grey Mini outside Garden Cafe."
    row=["gray","car","REAR","NONE"]
    record=reading(LEFT,"gray Mini hatchback",frontage="Garden Cafe building",roadside="",point=[225,268])
    assert solve(ask,[LEFT],[row],[record])[2] is None
    for altered in ({"point":[950,271]},{"frontage":"across the street from Garden Cafe"},
                    {"frontage":"other cafe"},{"frontage":"unreadable sign, maybe Garden Cafe"}):
        yes,boxes,msg=solve(ask,[LEFT],[row],[{**record,**altered}])
        assert not yes and boxes==[LEFT["box"]] and "garden cafe" in msg


def test_two_readers_disagree_and_landmark_still_shown_no_ground_granted():
    old=deepcopy(WHITE)
    ask="Beside the grey Mini outside Garden Cafe, on its right."
    readings=[["silver","suv","REAR","CURBSIDE"],["blue","suv","REAR","LANE"]]
    photos=[reading(LEFT,"blue SUV",frontage="Garden Cafe",point=[200,264]),
            reading(RIGHT,"grey hatchback",frontage="Garden Cafe",point=[650,264])]
    yes,boxes,msg=solve(ask,[LEFT,RIGHT],readings,photos)
    assert not yes and sorted(boxes)==sorted([LEFT["box"],RIGHT["box"]])
    assert all(term in msg for term in ("Garden Cafe","disagree","Mini","on its right"))
    assert WHITE==old


def test_near_color_alone_never_passes_as_exact_and_earlier_policy_replays():
    ask="Could you stop behind the grey Mini?"
    row=["silver","car","REAR","CURBSIDE"]
    yes,boxes,msg=solve(ask,[LEFT],[row])
    assert not yes and boxes==[LEFT["box"]] and "silver" in msg and "gray" in msg
    yes,boxes,msg=solve(ask,[LEFT],[row],expanded=False)
    assert not yes and not boxes and "described vehicle" in msg
    historical=parse("Stop behind the white car or the red one.",expanded=False)
    assert historical and "ground_side" not in historical


def test_only_current_motorcycles_and_comparable_vehicles_determine_nearest_in_image():
    ask="Behind the black sedan nearest the motorcycles."
    bike=road_user(8,"Motorbike",[263,169,302,263])
    yes,boxes,msg=solve(ask,[LEFT,RIGHT],[BLACK,BLACK],other=[bike])
    assert yes==[(LEFT,BLACK)] and boxes==[LEFT["box"]] and msg is None
    for other in ([{**bike,"frame_id":0}], [{**bike,"class_name":"Bicycle"}],
                  [bike,road_user(9,"Car",[249,160,293,264])]):
        yes,_,msg=solve(ask,[LEFT,RIGHT],[BLACK,BLACK],other=other)
        assert not yes and "nearest" in msg
    distant=road_user(9,"Car",[269,131,280,142])
    assert solve(ask,[LEFT,RIGHT],[BLACK,BLACK],other=[bike,distant])[0]==[(LEFT,BLACK)]
    equidistant=road_user(8,"Motorbike",[420,169,440,263])
    assert not solve(ask,[LEFT,RIGHT],[BLACK,BLACK],other=[equidistant])[0]


def test_proving_vehicle_does_not_prove_ground_surface_or_a_requested_side():
    bike=road_user(8,"Motorbike",[263,169,302,263])
    yes,boxes,msg=solve("Beside the black sedan nearest the motorcycles, in the gravel gap.",
                        [LEFT],[BLACK],other=[bike])
    assert not yes and boxes==[LEFT["box"]] and "gravel gap" in msg
    yes,boxes,msg=solve("Beside the car on the left, on its right.",[LEFT,RIGHT],[WHITE,WHITE])
    assert not yes and boxes==[LEFT["box"]] and "on its right" in msg
    yes,boxes,msg=solve("In front of the white Toyota.",[LEFT],[WHITE],[reading(LEFT,"white Toyota SUV")])
    assert not yes and boxes==[LEFT["box"]] and "in front" in msg


def test_observation_reader_rechecks_crop_pixels_and_current_track(monkeypatch):
    native=Image.new("RGB",(150,100),(100,50,30))
    out=BytesIO();native.save(out,format="PNG");photo=out.getvalue()
    box=[5,10,50,65];view=sha256(bay_anchors.crop_car(native,box)).hexdigest()
    request={"scene":{}}
    monkeypatch.setattr(parking_evidence,"source_input",lambda r,p:(request,sha256(photo).hexdigest(),"p"))
    entry={"reply":"white Nissan SUV","view":view}
    body={"appearance":{"4":entry}}
    monkeypatch.setattr(bay_anchors,"checked_context",lambda *args:(body,[(4,"car",box)]))
    result=qual.read_observations(request,photo,None,packets={"anchor":{}})
    assert result==[{"track_id":4,"box":box,"appearance":"white Nissan SUV"}]
    entry["view"]="b"*64
    with pytest.raises(ValueError,match="original crop"):
        qual.read_observations(request,photo,None,packets={"anchor":{}})
    body["appearance"]={"5":entry}
    with pytest.raises(ValueError,match="current track"):
        qual.read_observations(request,photo,None,packets={"anchor":{}})


def side_alternative(words,specs,answers,observations=()):
    from trackrag.dropoff.vehicle_ground_reference import direct_reference
    request={"text":words,"scene":{"frame":{"width":1000,"height":600}}}
    ref=direct_reference(request)
    return qual.possible_side_reference(request,ref or {},specs,answers,observations=observations)


@pytest.mark.parametrize(("asked","x","pictured"),[("right",410,"left"),("left",590,"right")])
def test_road_side_can_differ_from_pictured_side_without_proposing_ground(asked,x,pictured):
    target=car(7,[x-60,120,x+60,201])
    boxes,message=side_alternative(f"Stop behind the dark sedan parked on the {asked}.",
                                   [target],[["black","sedan","REAR","CURBSIDE"]])
    assert boxes==[target["box"]]
    assert f"black sedan {pictured} of centre" in message
    assert f"on the {asked}" in message and message.endswith("?")


@pytest.mark.parametrize("words",[
    "Stop behind the dark sedan on the right of the photo.",
    "Stop behind the dark sedan on the screen right.",
    "Stop behind the rightmost dark sedan.",
    "Stop behind the dark sedan on its right.",
    "Stop behind the red sedan on the right.",
    "Stop behind the dark SUV on the right.",
    "Stop behind the dark Toyota sedan on the right.",
])
def test_literal_side_other_identity_and_other_constraints_never_become_a_match(words):
    assert side_alternative(words,[LEFT],[BLACK]) is None


def test_no_side_offer_for_ties_on_either_side_or_existing_exact_match():
    text="Drop me beside the dark sedan on the right."
    farther_left=car(4,[23,80,103,126])
    assert side_alternative(text,[LEFT,farther_left],[BLACK,BLACK]) is None
    assert side_alternative(text,[LEFT,RIGHT],[BLACK,BLACK]) is None
    assert side_alternative(text,[RIGHT],[BLACK]) is None


def test_dark_body_requires_authentic_same_track_same_box_description_or_black():
    words="Stop beside the dark sedan on the right."
    row=["blue","sedan","REAR","CURBSIDE"]
    assert side_alternative(words,[LEFT],[row]) is None
    assert side_alternative(words,[LEFT],[row],[reading(RIGHT,"dark blue sedan")]) is None
    assert side_alternative(words,[LEFT],[row],[reading(LEFT,"blue sedan with dark blue trim")]) is None
    assert side_alternative(words,[LEFT],[row],[reading(LEFT,"dark blue pickup")]) is None
    assert side_alternative(words,[LEFT],[["white","sedan","REAR","BLOCKED"]]) is None
    result=side_alternative(words,[LEFT],[row],[reading(LEFT,"dark blue sedan")])
    assert result and "dark blue sedan" in result[1]
