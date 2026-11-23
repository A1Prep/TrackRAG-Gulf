from copy import deepcopy

import pytest

from trackrag.footage_facts import answer_from_tracks, supports_track_question


FRAME = 71
SIZE = (640, 360)


def observation(tid, name, box, frame=FRAME):
    return {"recording_id":"drive","source":"predicted","frame_id":frame,"track_id":tid,"class_name":name,"bbox_xyxy":box}


def view(name, tid=None, frame=FRAME, box=None):
    return {"evidence_id":name,"interval":{"recording_id":"drive","source":"predicted","track_id":tid,
            "start_frame":frame,"end_frame":frame,"rank":1,"score":1.},
            "images":[{"frame_id":frame,"bbox_xyxy":box,"image_path":"evidence/scene/71.png"}]}


BOXES = [[12,46,104,151], [300,135,450,340], [521,77,586,278]]
TRACKS = [observation(4,"Car",BOXES[0]),observation(7,"Car",BOXES[1]),observation(9,"Pedestrian",BOXES[2]),
          observation(19,"Bus",BOXES[2],frame=80)]
SCOPE = {"status":"available","paused_seconds":10.2,"moments":[{"requested_seconds":7.0,"frame_id":FRAME,"frame_seconds":7.0}]}


def ask(text, evidence=None, tracks=None, scope=None, cutoff=102, size=SIZE):
    q = {"question_id":"q","recording_id":"drive","query_frame":cutoff,"question":text}
    return answer_from_tracks(q,[view("E1")] if evidence is None else evidence,TRACKS if tracks is None else tracks,
                              playback_scope=scope,frame_size=size)


def test_actual_single_frame_count_classes_and_absence_have_citations():
    count=ask("At 00:07, how many cars can you see?",scope=SCOPE)
    assert "2 cars in frame 71" in count["answer"] and count["cited_evidence_ids"]==["E1"]
    assert "0 buses" in ask("How many buses are there in frame 71?")["answer"]
    classes=ask("Which classes of road users are visible at 7 seconds?",scope=SCOPE)
    assert "car (2)" in classes["answer"] and "pedestrian (1)" in classes["answer"]
    assert "bus" not in classes["answer"]
    assert "2 vehicles" in ask("How many vehicles are detected in this frame?",cutoff=FRAME)["answer"]
    assert "0 road users" in ask("How many road users are there in frame 71?",tracks=[])["answer"]


@pytest.mark.parametrize("words",["Which types of road users are detected in this frame?",
                                  "How many vehicles are detected in this frame?", "Which side of the frame is track 4 on?"])
def test_app_suggestion_routes_to_a_cited_local_fact(words):
    assert supports_track_question(words)
    assert ask(words,cutoff=FRAME)["cited_evidence_ids"]==["E1"]


def test_track_class_and_original_pixel_side_use_the_corresponding_observation():
    frames=[view("E3",4,box=BOXES[0]),view("E2",9,box=BOXES[2])]
    answer=ask("What type of road user is track 9 at 0:07?",frames,scope=SCOPE)
    assert "pedestrian" in answer["answer"] and answer["cited_evidence_ids"]==["E2"]
    assert "right side" in ask("Which side of the image is track 9 on at 0:07?",frames,scope=SCOPE)["answer"]
    assert "left side" in ask("Where in the frame is track 4?",frames,cutoff=FRAME)["answer"]
    assert ask("Where is track 4?",frames,cutoff=FRAME) is None
    assert ask("Which side is track 9 on at 0:07?",frames,scope=SCOPE,size=None) is None


def test_missing_collective_evidence_cannot_be_presented_as_a_full_frame_count():
    frames=[view("E1",4,box=BOXES[0]),view("E2",7,box=BOXES[1])]
    assert ask("How many cars are visible in frame 71?",frames)["cited_evidence_ids"]==["E1","E2"]
    assert ask("How many cars are visible in frame 71?",frames[:1]) is None
    assert ask("How many buses are visible in frame 71?",frames) is None
    assert ask("What class is track 19 in frame 71?") is None


@pytest.mark.parametrize("text",[
    "How many red cars are there in frame 71?", "How many cars passed over time?", "What direction is track 4 moving?",
    "Where is the truck parked?", "At frame 71, which street are we on?", "How many trucks are visible in frame 71?",
    "How many cars are visible in frame 71 and frame 80?", "How many cars will there be in the next frame?",
    "How many people are behind the shopping centre?", "How many medium vehicles are turning left in frame 71?",
    "At frame 71, how many cars are there, and are they moving?", "How many cars were in the school zone?",
    "How many cars in 7 seconds?", "7 seconds, how many cars are there?", "How many cars at 7 seconds, 8 seconds?",
])
def test_unknown_appearance_places_motion_and_multiple_instants_defer(text):
    assert not supports_track_question(text)
    assert ask(text) is None


def test_scope_must_agree_with_words_frames_evidence_time_and_pause():
    earlier = ask("How many cars are visible at 0:07?",scope=SCOPE)
    assert earlier and "frame 71" in earlier["answer"]
    assert ask("At 0:07, how many cars are visible?",scope=None) is None
    assert ask("At 0:09, how many cars are visible?",scope=SCOPE) is None
    assert ask("At 0:07 (frame 80), how many cars are visible?",scope=SCOPE) is None
    assert ask("At 0:07, how many cars are visible?",evidence=[view("E1",frame=102)],scope=SCOPE) is None
    assert ask("At 0:07, how many cars are visible?",scope={**SCOPE,"status":"outside_review_window"}) is None
    assert ask("At 0:07, how many cars are visible?",scope={**SCOPE,"paused_seconds":6.9}) is None
    earlier_sample = {**SCOPE, "moments": [{**SCOPE["moments"][0], "frame_seconds": 6.9}]}
    assert ask("At 0:07, how many cars are visible?",scope=earlier_sample) is None
    assert ask("At 0:07, how many cars are visible?",scope={**SCOPE,"moments":SCOPE["moments"]*2}) is None
    assert ask("How many cars are visible in frame 71?",cutoff=70) is None
    assert ask("How many cars are visible?",evidence=[view("E1",frame=71)],cutoff=80) is None


@pytest.mark.parametrize("change",["wrong recording","annotated evidence","altered box","annotation mixed in","duplicate track"])
def test_mismatched_records_are_never_local_facts(change):
    views=[view("E1",4,box=BOXES[0])]
    tracks=deepcopy(TRACKS)
    if change=="wrong recording": views[0]["interval"]["recording_id"]="other"
    if change=="annotated evidence": views[0]["interval"]["source"]="annotated"
    if change=="altered box": views[0]["images"][0]["bbox_xyxy"]=[10,40,100,150]
    if change=="annotation mixed in": tracks[0]["source"]="annotated"
    if change=="duplicate track": tracks.append(tracks[0])
    assert ask("What class is track 4 in frame 71?",views,tracks) is None
