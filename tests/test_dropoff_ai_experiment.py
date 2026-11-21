from copy import deepcopy
import json

import pytest

from tools.dropoff_ai_continuation import (AICollator, checkpoint_resident_suffix, pinned_rows,
                                          training_messages)
from tools.dropoff_train import GroundingCollator
from trackrag.dropoff.formatting import build_messages
from trackrag.dropoff.model import GroundingConfig


def request():
    frame = {'frame_id': 9, 'timestamp_s': 1.25, 'width': 1000, 'height': 500,
             'image_path': 'frame.jpg', 'recording_id': 'rec1'}
    tracks = [{'frame_id': i, 'track_id': 7, 'class_name': 'white car, class', 'source': 'predicted',
               'recording_id': 'rec1', 'bbox_xyxy': [4.322, 10.1, 200., 300.56]} for i in [4, 9]]
    return {'request_id': 'bound', 'scene': {'frame': frame, 'source': 'predicted', 'tracks': tracks},
            'text': 'the one on the right', 'input_mode': 'typed',
            'dialogue': [{'role': 'user', 'text': 'the two cars', 'recording_id': 'rec1', 'frame_id': 9}]}


def test_compact_prompt_round_trips_all_origin_precision_history_and_dialogue():
    q = request()
    normal, compact = build_messages(q), build_messages(q, observation_format='compact_rows_v1')
    assert normal == build_messages(q, observation_format='json')
    assert normal[0] == compact[0] and normal[2:] == compact[2:]
    assert normal[1]['content'][0] == compact[1]['content'][0]
    original = json.loads(normal[1]['content'][1]['text'].split('\n', 1)[1])
    encoded = compact[1]['content'][1]['text'].splitlines()[1:]
    decoded = json.loads(encoded[0])
    assert decoded.pop('observation_columns') == ['frame_id', 'track_id', 'class_name', 'x1', 'y1', 'x2', 'y2']
    decoded['observations'] = []
    for string in encoded[1:]:
        frame, track, label, *box = json.loads(string)
        decoded['observations'].append({'frame_id': frame, 'track_id': track, 'class_name': label, 'bbox_xyxy': box})
    assert decoded == original
    assert training_messages({'request': q, 'completion': 'target'})[:-1] == compact


def test_compact_rejects_future_observations_and_unknown_condition():
    q = request()
    q['scene']['tracks'][0]['frame_id'] = 100
    with pytest.raises(ValueError, match='future'):
        build_messages(q, observation_format='compact_rows_v1')
    with pytest.raises(ValueError, match='format'):
        GroundingConfig(observation_format='unknown')


def test_ai_only_frozen_dataset_cannot_be_replaced_by_other_requests(tmp_path):
    path = tmp_path / 'train-references.jsonl'
    path.write_text('{"request_id":"passenger-a01"}\n')
    with pytest.raises(ValueError, match='Changed frozen'):
        pinned_rows(path)


def test_disputed_geometry_status_does_not_supervise_early_eos(monkeypatch):
    torch = pytest.importorskip('torch')
    def base(self, cases):
        return {'labels': torch.tensor([[-100, 10, 20, 25, 80, 90]])}
    monkeypatch.setattr(GroundingCollator, '__call__', base)

    class Tokenizer:
        def encode(self, text, **kwargs):
            assert text == '{"status":"proposed"'
            return [10, 20, 25]
    processor = type('Processor', (), {'tokenizer': Tokenizer()})()
    got = AICollator(processor, 128)([{'full_target': False, 'completion': '{"status":"proposed"'}])
    assert got['labels'].tolist() == [[-100, 10, 20, 25, -100, -100]]


def test_checkpointed_suffix_reproduces_gradients_with_frozen_inputs():
    torch = pytest.importorskip('torch')
    torch.manual_seed(100)
    original = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(12, 12), torch.nn.GELU()),
                                   torch.nn.Sequential(torch.nn.Linear(12, 8), torch.nn.GELU()))
    copied = deepcopy(original)
    x = torch.randn(9, 12)
    checkpoint_resident_suffix(copied)
    for model in [original, copied]:
        model(x).square().mean().backward()
    for a, b in zip(original.parameters(), copied.parameters()):
        assert a.grad is not None and torch.equal(a.grad, b.grad)
    assert x.grad is None
