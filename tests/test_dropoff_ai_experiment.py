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


@pytest.mark.parametrize('heads,kv,mask_kind', [(4, 1, 'causal'), (4, 2, 'boolean'),
                                                (2, 2, 'additive'), (4, 2, 'none')])
def test_tiled_attention_matches_dense_native_output_and_gradient(heads, kv, mask_kind):
    torch = pytest.importorskip('torch')
    import torch.nn.functional as F
    from trackrag.dropoff.training_attention import kernel

    torch.manual_seed(45)
    length = 17
    args = [torch.randn(1, h, length, 8, dtype=torch.float64, requires_grad=True) for h in (heads, kv, kv)]
    copies = [x.detach().clone().requires_grad_(True) for x in args]
    mask = None
    if mask_kind in ('boolean', 'additive'):
        mask = torch.arange(length)[:, None] >= torch.arange(length)[None, :]
        mask = mask[None, None, :, :]
        if mask_kind == 'additive':
            mask = torch.where(mask, 0., -1e5).to(torch.float64)
    native = F.scaled_dot_product_attention(*args, attn_mask=mask, is_causal=mask_kind == 'causal',
                                           dropout_p=0., scale=.4, enable_gqa=heads != kv)
    tiled = kernel()(*copies, mask, mask_kind == 'causal', .4, 5)
    torch.testing.assert_close(native, tiled, rtol=1e-10, atol=1e-10)
    upstream = torch.randn_like(native)
    native.backward(upstream)
    tiled.backward(upstream)
    for ref, got in zip(args, copies):
        torch.testing.assert_close(ref.grad, got.grad, rtol=1e-9, atol=1e-9)


def test_tiled_dispatch_runs_real_qwen_text_block_without_changing_parent_mask_config():
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
    from trackrag.dropoff.training_attention import install_qwen_training_attention

    torch.manual_seed(11)
    config = Qwen3VLTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    config._attn_implementation = 'sdpa'
    native = Qwen3VLTextDecoderLayer(config, 0).eval()
    bounded = deepcopy(native)
    shared_config = bounded.self_attn.config
    install_qwen_training_attention([bounded], tile=5)
    assert shared_config._attn_implementation == 'sdpa'
    assert bounded.self_attn.config._attn_implementation == 'trackrag_tiled_training_sdpa_v1'
    x = torch.randn(1, 17, 32)
    phase = torch.randn(1, 17, 8)
    rope = (phase.cos(), phase.sin())
    a = native(x, attention_mask=None, position_embeddings=rope)
    b = bounded(x, attention_mask=None, position_embeddings=rope)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    for normal, tiled in zip(native.parameters(), bounded.parameters()):
        torch.testing.assert_close(normal.grad, tiled.grad, rtol=1e-4, atol=1e-5)
