
from pathlib import Path
import json
from unittest.mock import Mock

import cv2
import numpy as np

import pytest


from trackrag.answering import answer_question, prepare_evidence, validate_answer
from trackrag.evidence_time import referenced_track_ids, requested_frame_ids
from trackrag.evidence_motion import horizontal_motion, verify_horizontal_claim


def test_explicit_id_aliases_require_real_cited_observations():
    assert referenced_track_ids("track IDs 7, ID 9 and track #11; tracked under ID 13") == [7, 9, 11, 13]
    assert requested_frame_ids("frame ID 7 through frame #11, 50 kilometres per hour") == [7, 11]
    assert requested_frame_ids("The centre moved from 0.41201 in frame 1 to 0.18868 in frame 100.") == [1, 100]
    assert requested_frame_ids("Frames 1 through 100.") == [1, 100]
    assert referenced_track_ids("The bus was tracked as ID 42 at frame 8.") == [42]
    evidence = [{'evidence_id':'E1','interval':{'track_id':7,'source':'predicted'},
                 'images':[{'frame_id':5,'observations':[{'track_id':7}]}]}]
    question = {'question':'What appears at frame 5?'}
    for phrase in ('track #999', 'track ID 999', 'tracked as ID 999', 'frame #999', 'frame ID 999'):
        with pytest.raises(ValueError):
            validate_answer({'answer_status':'answered','answer':phrase,'cited_evidence_ids':['E1']}, evidence, question)


def test_motion_is_specific_to_the_cited_identity_and_requested_frames():
    def image(frame, x):
        return {'frame_id':frame, 'privacy':{'original_size_wh':[100,50]}, 'observations':[
            {'track_id':7, 'bbox_xyxy':[x,0,x+5,10]}, {'track_id':8,'bbox_xyxy':[90-x,0,95-x,10]}]}
    e = [{'interval':{'track_id':7}, 'images':[image(5,60),image(8,40),image(10,20),image(12,85)]}]
    q = 'Does the vehicle move left or right in the image from frame 5 to frame 10?'
    assert horizontal_motion(e, 7, {5,10})['image_direction'] == 'left'
    verify_horizontal_claim(q, 'Track ID 8 moves toward the right of the image.', e)
    with pytest.raises(ValueError,match='contradicts'):
        verify_horizontal_claim(q, 'Track ID 7 moves toward the right of the image.', e)
    verify_horizontal_claim(q, 'Track ID 7 is on the right.', e)
    verify_horizontal_claim(q, 'Track ID 7 does not move right.', e)
    verify_horizontal_claim(q, 'Track 7 moves right-to-left.', e)
    with pytest.raises(ValueError,match='contradicts'):
        verify_horizontal_claim(q, 'Track 7 moves left to right.', e)
    with pytest.raises(ValueError,match='contradicts'):
        verify_horizontal_claim('Toward which side of the image does track 7 move from frame 5 to frame 10?', 'To the right.', e)
    assert horizontal_motion(e, 7) is None  # The later movement reverses direction.


def test_unavailable_labels_do_not_send_detector_images(sample_question, sample_evidence, tmp_path):
    sample_question['question'] = 'Which predicted track is visible?'
    for item in sample_evidence: item['interval']['source'] = 'annotated'
    model = Mock()
    result = answer_question(sample_question,sample_evidence,tmp_path,model)
    assert result['answer_status'] == 'insufficient_evidence'
    model.generate.assert_not_called()
    receipt = json.loads(next(tmp_path.glob('hosted/*/receipt.json')).read_bytes())
    assert receipt['request_status'] == 'not_requested' and receipt['sent'] == []

def create_test_observations(run_dir):
    """Minimal Task 5 export; not evidence for the real starter answer key."""
    path=run_dir/'tracks.jsonl'
    path.write_text(json.dumps({'recording_id':'video_204647','source':'annotated',
        'frame_id':205,'track_id':33,'class_name':'Pedestrian','bbox_xyxy':[100,100,200,300]})+'\n')
    return path


@pytest.fixture
def sample_question():
    return {
        "question_id": "Q001",
        "recording_id": "video_204647",
        "query_frame": 205,
        "question": "How many pedestrians are visible?",
    }


@pytest.fixture
def sample_evidence(tmp_path):
    # Create a dummy image file for testing.
    image_path = tmp_path / "frame_00205.jpg"
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    create_test_observations(tmp_path)

    return [
        {
            "evidence_id": "E1",
            "interval": {
                "recording_id": "video_204647",
                "source": "annotated",
                "track_id": 33,
                "start_frame": 205,
                "end_frame": 205,
                "rank": 1,
                "score": 0.95,
            },
            "images": [
                {
                    "frame_id": 205,
                    "bbox_xyxy": (100, 100, 200, 300),
                    "image_path": "frame_00205.jpg",
                }
            ],
        }
    ]


def test_no_evidence_returns_insufficient_evidence(
    sample_question, tmp_path
):
    model = Mock()

    result = answer_question(
        sample_question,
        [],
        tmp_path,
        model,
    )

    assert result == {
        "answer_status": "insufficient_evidence",
        "answer": None,
        "cited_evidence_ids": [],
    }

    # The model must not be called.
    model.generate.assert_not_called()


def test_valid_evidence_and_citation_are_accepted(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.return_value = {
        "answer_status": "answered",
        "answer": "One pedestrian is visible.",
        "cited_evidence_ids": ["E1"],
    }

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result == {
        "answer_status": "answered",
        "answer": "One pedestrian is visible.",
        "cited_evidence_ids": ["E1"],
    }

    model.generate.assert_called_once()


def test_invented_evidence_id_is_rejected(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.return_value = {
        "answer_status": "answered",
        "answer": "Two pedestrians are visible.",
        "cited_evidence_ids": ["E9"],
    }

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result == {
        "answer_status": "error",
        "answer": "ValueError: Model cited unknown or duplicate evidence IDs",
        "cited_evidence_ids": [],
    }
    model.generate.assert_called_once()


def test_model_timeout_returns_error(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.side_effect = TimeoutError("Model request timed out")

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result == {
        "answer_status": "error",
        "answer": "TimeoutError: Model request timed out",
        "cited_evidence_ids": [],
    }
    model.generate.assert_called_once()


def test_missing_image_returns_error(
    sample_question, tmp_path
):
    model = Mock()
    create_test_observations(tmp_path)

    evidence = [
        {
            "evidence_id": "E1",
            "interval": {
                "recording_id": "video_204647",
                "source": "annotated",
                "track_id": 33,
                "start_frame": 205,
                "end_frame": 205,
                "rank": 1,
                "score": 0.95,
            },
            "images": [
                {
                    "frame_id": 205,
                    "bbox_xyxy": (100, 100, 200, 300),
                    "image_path": "missing.jpg",
                }
            ],
        }
    ]

    result = answer_question(
        sample_question,
        evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "error"
    model.generate.assert_not_called()
    with pytest.raises(FileNotFoundError,match='Evidence image not found'):
        prepare_evidence(evidence,tmp_path)


def test_model_response_without_citations_is_rejected(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.return_value = {
        "answer_status": "answered",
        "answer": "A pedestrian is visible.",
        "cited_evidence_ids": [],
    }

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "error"
    assert result["cited_evidence_ids"] == []
    model.generate.assert_called_once()


def test_evidence_path_cannot_escape_run_directory(
    sample_question, tmp_path
):
    model = Mock()
    create_test_observations(tmp_path)

    evidence = [
        {
            "evidence_id": "E1",
            "interval": {
                "recording_id": "video_204647",
                "source": "annotated",
                "track_id": 33,
                "start_frame": 205,
                "end_frame": 205,
                "rank": 1,
                "score": 0.95,
            },
            "images": [
                {
                    "frame_id": 205,
                    "bbox_xyxy": (100, 100, 200, 300),
                    "image_path": "../outside.jpg",
                }
            ],
        }
    ]

    result = answer_question(
        sample_question,
        evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "error"
    model.generate.assert_not_called()
    with pytest.raises(ValueError,match='path escapes'):
        prepare_evidence(evidence,tmp_path)

def test_model_receives_processed_image_not_original(
    sample_question, sample_evidence, tmp_path
):
    model = Mock()
    model.generate.return_value = {
        "answer_status": "answered",
        "answer": "One pedestrian is visible.",
        "cited_evidence_ids": ["E1"],
    }

    original_path = (
        tmp_path / sample_evidence[0]["images"][0]["image_path"]
    ).resolve()

    result = answer_question(
        sample_question,
        sample_evidence,
        tmp_path,
        model,
    )

    assert result["answer_status"] == "answered"

    model_evidence = model.generate.call_args.kwargs["evidence"]
    processed_path = model_evidence[0]["images"][0]["image_path"]

    assert Path(processed_path).is_file()
    assert Path(processed_path) != original_path
    assert 'hosted' in Path(processed_path).parts
    # Actual masking changed the synthetic pedestrian pixels; the raw image is
    # untouched. A different filename alone would not establish processing.
    assert np.all(cv2.imread(str(processed_path))[150,150] == 35)
    assert np.all(cv2.imread(str(original_path))[150,150] == 0)


@pytest.mark.parametrize('response,expected',[
    ({'answer_status':'insufficient_evidence','answer':'The views do not establish the requested identity.','cited_evidence_ids':[]},'insufficient_evidence'),
    ({'answer_status':'error','answer':'Provider failed.','cited_evidence_ids':['E1']},'error'),
    ({'answer_status':'answered','answer':'A person.','cited_evidence_ids':['E1','E1']},'error'),
])
def test_status_and_duplicate_citations(sample_question,sample_evidence,tmp_path,response,expected):
    model=Mock()
    model.generate.return_value=response
    result=answer_question(sample_question,sample_evidence,tmp_path,model)
    assert result['answer_status']==expected
    model.generate.assert_called_once()


def test_future_evidence_is_rejected_before_preparation_or_model(sample_question,sample_evidence,tmp_path,monkeypatch):
    sample_evidence[0]['interval']['end_frame']=206
    sample_evidence[0]['images'][0]['frame_id']=206
    prepare=Mock(side_effect=AssertionError('Must not process future evidence'))
    monkeypatch.setattr('trackrag.answering.prepare_evidence',prepare)
    model=Mock()
    result=answer_question(sample_question,sample_evidence,tmp_path,model)
    assert result['answer_status']=='error'
    assert 'after query_frame' in result['answer']
    prepare.assert_not_called()
    model.generate.assert_not_called()


def test_missing_requested_frame_cannot_receive_an_answered_result(sample_question,sample_evidence,tmp_path):
    sample_question['query_frame']=206
    sample_question['question']='Compare frames 205 and 206.'
    model=Mock()
    model.generate.return_value={'answer_status':'answered','answer':'It stays in place.','cited_evidence_ids':['E1']}
    result=answer_question(sample_question,sample_evidence,tmp_path,model)
    assert result['answer_status']=='error'
    assert 'exact cited image frames' in result['answer']
    model.generate.assert_called_once()


def test_absent_tracker_id_returns_no_identity_or_hosted_call(sample_question,sample_evidence,tmp_path):
    model=Mock()
    sample_question["question"]="Is track ID 9876 the pedestrian at frame 205?"
    result=answer_question(sample_question,sample_evidence,tmp_path,model)
    assert result["answer_status"]=="insufficient_evidence"
    assert "9876" in result["answer"] and result["cited_evidence_ids"]==[]
    model.generate.assert_not_called()


def test_requested_exact_target_speed_is_unmeasured_but_visible_speed_displays_can_be_read(sample_question,sample_evidence,tmp_path):
    from trackrag.answering import asks_for_unmeasured_speed
    model=Mock()
    sample_question["question"]="Can you tell the exact speed of the other white SUV?"
    result=answer_question(sample_question,sample_evidence,tmp_path,model)
    assert result["answer_status"]=="insufficient_evidence" and result["cited_evidence_ids"]==[]
    model.generate.assert_not_called()
    for visible in ("What speed limit is written on the sign?", "What speed is displayed on the dashboard?",
                    "What is the exact speed of the car in pixels per second?"):
        assert not asks_for_unmeasured_speed(visible)


def test_other_identity_cannot_donate_endpoint_but_scene_backed_absence_can():
    question={'question_id':'q','recording_id':'r','query_frame':10,'question':'Is track 7 visible in frame 10?'}
    first={'evidence_id':'E1','interval':{'track_id':7},'images':[{'frame_id':1,'observations':[{'track_id':7}]}]}
    second={'evidence_id':'E2','interval':{'track_id':8},'images':[{'frame_id':10,'observations':[{'track_id':8}]}]}
    answer={'answer_status':'answered','answer':'Track 7 is not visible in frame 10.','cited_evidence_ids':['E1','E2']}
    with pytest.raises(ValueError,match='same stated track'):
        validate_answer(answer,[first,second],question)
    # A cited current full scene can support absence while the earlier identity
    # remains grounded in its own observations. Unrelated tracked views cannot.
    second['interval']['track_id']=None
    assert validate_answer(answer,[first,second],question)==answer
    answer['answer']='Track 999 is visible in frame 10.'
    with pytest.raises(ValueError,match='source-labeled observations'):
        validate_answer(answer,[first,second],question)


def test_cited_annotation_observations_can_support_a_complete_id_set():
    question={'question_id':'q','recording_id':'r','query_frame':435,
              'question':'Which pedestrian track IDs are present in both frame 411 and frame 435?'}
    evidence=[{'evidence_id':'E1','interval':{'track_id':37},'images':[
        {'frame_id':frame,'observations':[{'track_id':track} for track in (37,39,40,41)]} for frame in (411,435)]}]
    answer={'answer_status':'answered','answer':'Annotated tracks 37, 39, 40 and 41.','cited_evidence_ids':['E1']}
    assert validate_answer(answer,evidence,question)==answer


def test_predictions_preserve_their_ids_without_opening_human_annotations(sample_question,sample_evidence,tmp_path):
    row=json.loads((tmp_path/'tracks.jsonl').read_text())
    row.update(source='predicted',track_id=99)
    (tmp_path/'tracks.jsonl').write_text(json.dumps(row)+'\n')
    sample_evidence[0]['interval'].update(source='predicted',track_id=99)
    model=Mock()
    model.generate.return_value={'answer_status':'answered','answer':'Predicted track 99 is a pedestrian.','cited_evidence_ids':['E1']}
    result=answer_question(sample_question,sample_evidence,tmp_path,model)
    assert result['answer_status']=='answered'
    received=model.generate.call_args.kwargs['evidence'][0]['images'][0]['observations']
    assert [(item['source'],item['track_id']) for item in received]==[('predicted',99)]
    assert not (tmp_path/'EMT').exists()
