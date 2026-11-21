from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.dropoff_spatial_controls import alternate_sources
from tools.dropoff_spatial_fit import score
from trackrag.dropoff.spatial import (InstructionVocabulary, NoVisibleGround, geometry,
                                     instruction_terms)
from trackrag.dropoff.spatial_vision import FEATURE_CHANNELS, GRID_HEIGHT, GRID_WIDTH


def inputs():
    features = np.zeros((FEATURE_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    features[0] = 1
    logits = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    logits[12, 40] = 10
    extent = np.full((4, GRID_HEIGHT, GRID_WIDTH), .1, dtype=np.float32)
    return logits, extent, features


def test_original_pixels_and_variable_learned_extents():
    logits, extents, features = inputs()
    a = geometry(logits, extents, features, (640, 320))
    extents[3, 12, 40] = .2
    b = geometry(logits, extents, features, (640, 320))
    assert a['destination_point_xy'] == b['destination_point_xy'] == [405., 125.]
    assert a['destination_box_xyxy'][:3] == b['destination_box_xyxy'][:3]
    assert b['destination_box_xyxy'][3] == pytest.approx(a['destination_box_xyxy'][3] + 32)


def test_predicted_boxes_and_nonroad_remove_highest_logit():
    logits, extents, features = inputs()
    logits[9, 10] = 9
    features[0, 9, 10], features[13, 9, 10] = 0, 1
    logits[10, 20] = 5
    got = geometry(logits, extents, features, (640, 320), [[400, 120, 410, 130]])
    assert got['destination_point_xy'] == [205, 105]


def test_no_visible_road_has_no_default_geometry():
    logits, extents, features = inputs()
    with pytest.raises(NoVisibleGround):
        geometry(logits, extents, features, (640, 320), [[0, 0, 640, 320]])


def test_reject_nonfinite_or_inverted_source_values():
    logits, extents, features = inputs()
    with pytest.raises(ValueError, match='exclusion'):
        geometry(logits, extents, features, (640, 320), [[99, 10, 98, 25]])
    features[0, 2, 3] = np.nan
    with pytest.raises(ValueError, match='features'):
        geometry(logits, extents, features, (640, 320))


def test_training_reflection_swaps_words_and_keeps_reference_class():
    assert instruction_terms('stop next to the grey cars on the left', mirror=True) == instruction_terms(
        'stop beside the gray car on the right')
    vocab = InstructionVocabulary.fit(['beside the left car', 'beside the left van'] * 3)
    assert 'right' in vocab.terms and not np.array_equal(vocab.encode('left'), vocab.encode('right'))
    assert not vocab.encode('novelword').any()


def test_utterance_control_retains_every_case_and_excludes_own_scene():
    rows = [{'case_id': str(i), 'scene_group': i // 2, 'request': {'text': f'command {i}'}} for i in range(8)]
    assignment = alternate_sources(rows)
    assert set(assignment) == set(assignment.values()) == set(range(8))
    assert all(rows[a]['scene_group'] != rows[b]['scene_group'] for a, b in assignment.items())


def _scoring_fixture(tmp_path):
    mask = Image.new('L', (20, 10))
    mask.putpixel((7, 4), 255)
    path = Path(tmp_path) / 'native.png'
    mask.save(path)
    case = {'case_id': 'k', 'request': {'scene': {'frame': {'width': 200, 'height': 100}}},
            'source': {'image_sha256': 'abc', 'mask_path': str(path)},
            'target': {'destination_box_xyxy': [70, 40, 80, 50]}}
    pred = {'case_id': 'k', 'image_sha256': 'abc', 'checkpoint_sha256': 'abc1', 'status': 'geometry',
            'destination_point_xy': [75, 45], 'destination_box_xyxy': [70, 40, 80, 50]}
    return case, pred


def test_native_mask_scoring_uses_real_point_and_keeps_abstentions(tmp_path):
    case, pred = _scoring_fixture(tmp_path)
    no_road = {'case_id': 'q', 'image_sha256': 'def', 'checkpoint_sha256': 'abc1', 'status': 'no_visible_road'}
    second = {**case, 'case_id': 'q', 'source': {**case['source'], 'image_sha256': 'def'}}
    measured, rows = score([pred, no_road], [case, second])
    assert (measured['cases'], measured['mask_hits'], measured['proposals']) == (2, 1, 1)
    assert measured['mean_box_iou_all_cases'] == .5
    assert rows[0]['box_iou'] == 1


@pytest.mark.parametrize('change', [{'image_sha256': 'different'}, {'case_id': 'other'},
                                     {'destination_point_xy': [900, 0]}, {'status': 'no_visible_road'}])
def test_scorer_rejects_wrong_scene_and_invalid_outputs(tmp_path, change):
    case, pred = _scoring_fixture(tmp_path)
    with pytest.raises(ValueError):
        score([{**pred, **change}], [case])
