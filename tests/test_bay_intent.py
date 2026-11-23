from copy import deepcopy

from trackrag.dropoff.bay_intent import Vehicle, match, parse
from trackrag.dropoff.passenger import digest


def bay(x,y, *, spread=1, parallel=True):
    polygon=[[x-60*spread,y+10*spread],[x-38*spread,y-31*spread],
             [x+54*spread,y-12*spread],[x+37*spread,y+33*spread]]
    xs,ys=zip(*polygon)
    return {"point":[x,y],"box":[min(xs),min(ys),max(xs),max(ys)],"polygon":polygon,
            "evidence":{"recipe":"painted_curbside_stall_ends_v1" if parallel else "visually_checked_parking_components_v1"}}


def track(i,box,typ="Car",frame=9):
    return {"track_id":i,"class_name":typ,"frame_id":frame,"bbox_xyxy":list(box)}


def fixture():
    right,left=bay(705,430),bay(250,430)
    scene={"frame":{"frame_id":9,"width":1000,"height":600},"source":"predicted","tracks":[
        track(1,[540,317,630,405]),track(2,[765,345,855,462]),track(3,[100,340,180,412]),track(4,[310,355,395,463])]}
    obs={}
    for opt,pair in [(right,scene["tracks"][:2]),(left,scene["tracks"][2:])]:
        obs[digest(opt["polygon"])]={"frontage":"Al Noor Building | green hedge",
            "pair":[{"track_id":t["track_id"],"box":list(t["bbox_xyxy"]),"appearance":"silver sedan"} for t in pair]}
    return [right,left],scene,obs


def test_compound_grammar_and_existing_routes():
    assert parse("drop me near the restaurant on the right") is None
    assert parse("Can you drop me between the two black cars?") is None
    request=parse("An empty parallel space between the silver cars on the right, outside Al Noor building.")
    assert (request.side,request.orientation,request.vehicles,request.frontage,request.uncertain)==(
        "right","parallel",(Vehicle("car","silver"),Vehicle("car","silver")),("al","noor","building"),False)
    assert parse("in an empty marked space by the gray SUV").vehicles==(Vehicle("suv","gray"),)
    assert parse("one of the empty painted spaces beside the hedge").frontage==("hedge",)
    request=parse("any of the bays along the building beyond that van")
    assert (request.frontage,request.vehicles,request.vehicle_relation,request.uncertain)==(
        ("building",),(Vehicle("van"),),"beyond",True)


def test_all_properties_and_two_current_bracketing_crops_are_needed_for_full_match():
    opts,scene,obs=fixture();right=opts[0];key=digest(right["polygon"])
    words="Use the parallel bay between the two silver cars on the right, outside Al Noor building."
    assert match(words,opts,scene,obs)==([right],[])
    data=deepcopy(obs);data[key]["pair"][0]["appearance"]="blue sedan"
    assert match(words,opts,scene,data)==([],[])
    data=deepcopy(obs);data[key]["frontage"]="NONE";assert match(words,opts,scene,data)==([],[right])
    data=deepcopy(obs);data[key]["frontage"]="a pharmacy";assert match(words,opts,scene,data)==([],[])
    data=deepcopy(obs);data[key]["frontage"]="Al Hayat Building";assert match(words,opts,scene,data)==([],[])
    data=deepcopy(obs);data[key]["pair"][0]["box"][0]-=1
    assert match(words,opts,scene,data)==([],[right])
    stale=deepcopy(scene);stale["tracks"][1]["frame_id"]=8
    assert match(words,opts,stale,obs)==([],[])
    uncalibrated=deepcopy(right);uncalibrated["evidence"]["recipe"]="visually_checked_parking_components_v1"
    assert match(words,[uncalibrated],scene,obs)==([],[uncalibrated])


def test_two_same_side_or_duplicate_vehicles_cannot_substitute_for_a_pair():
    opts,scene,obs=fixture();right=opts[0];pair=obs[digest(right["polygon"])]["pair"]
    scene["tracks"][1]["bbox_xyxy"]=[570,322,638,410];pair[1]["box"]=[570,322,638,410]
    assert match("the bay between the two silver cars on the right",opts,scene,obs)==([],[])
    scene["tracks"][1]=deepcopy(scene["tracks"][0]);pair[1]=deepcopy(pair[0])
    assert match("the bay between the two silver cars on the right",opts,scene,obs)==([],[])


def test_signed_single_vehicle_type_and_color_are_grounded_individually():
    opts,scene,obs=fixture();right=opts[0]
    pair=obs[digest(right["polygon"])]["pair"];pair[1]["appearance"]="gray SUV"
    words="Let me out in an empty marked space by the grey SUV on the right."
    assert match(words,opts,scene,obs)==([right],[])
    pair[1]["appearance"]="gray sedan"
    assert match(words,opts,scene,obs)==([],[])
    pair[1]["appearance"]="unknown"
    assert match(words,opts,scene,obs)==([],[right])
    assert match("the right bay between a silver sedan and a blue truck",opts,scene,obs)==([],[])


def test_frontage_placeholders_and_combinations_preserve_location_and_uncertainty():
    opts,scene,obs=fixture();right=opts[0];key=digest(right["polygon"])
    words="One of the empty painted spaces beside the hedge on the right."
    assert match(words,opts,scene,obs)==([right],[])
    assert match("a bay by the Al Noor building and beside the hedge on the right",opts,scene,obs)==([right],[])
    obs[key]["frontage"]="maybe a hedge | NONE"
    assert match(words,opts,scene,obs)==([],[])
    assert match("a bay beside the pharmacy on the right",opts,scene,obs)==([],[])
    assert match("a bay on the opposite side of the road near a hedge",opts,scene,obs)==([],[])
    assert match("use the marked bay on the right and on the left",opts,scene,obs)==([],[])


def test_beyond_and_landmark_nearest_never_use_camera_y_as_proven_relative_depth():
    opts,scene,obs=fixture();right=opts[0];key=digest(right["polygon"])
    scene["tracks"][1]["class_name"]="Van";obs[key]["pair"][1]["appearance"]="white van"
    words="Use any of the bays on the right along the building beyond that van."
    assert match(words,opts,scene,obs)==([],[right])
    scene["tracks"][1]["bbox_xyxy"]=[760,256,840,336]
    assert match(words,opts,scene,obs)==([],[])
    assert match("the bay closest to the green hedge on the right",opts,scene,obs)==([],[right])


def test_generic_ranking_needs_comparable_outlines_and_a_visible_difference():
    _,scene,_=fixture();near=bay(705,470);far=bay(705,370);left=bay(240,455)
    assert match("Use the nearest marked bay on the right",[far,near,left],scene,{})==([near],[])
    assert match("the nearest parking bay",[near,left],scene,{})==([],[near,left])
    larger=bay(775,421,spread=1.8);small=bay(625,425)
    assert match("use the largest visible bay on the right",[small,larger],scene,{})==([larger],[])
    assert match("use the leftmost marked bay on the right",[larger,small],scene,{})==([small],[])
    assert match("a wheelchair accessible bay",[near,left],scene,{})==([],[])
    assert match("a parking bay",[near],{**scene,"source":"ground_truth"},{})==([],[])
