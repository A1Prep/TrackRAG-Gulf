from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from tools.dropoff_prepare_passenger_ground import prepare
from trackrag.dropoff.road_cache import ROAD_RECIPE_SHA256, mask_shape, read_mask, save_mask


SHA = 'a' * 64


def test_roundtrip_eviction_by_image_and_size_and_recipe(tmp_path):
    size = (1600, 900)
    mask = np.zeros(mask_shape(size), dtype=bool)
    mask[150:] = True
    assert read_mask(tmp_path, SHA, size) is None
    receipt = save_mask(tmp_path, SHA, size, mask)
    assert receipt['recipe_sha256'] == ROAD_RECIPE_SHA256
    assert np.array_equal(mask, read_mask(tmp_path, SHA, size))
    assert read_mask(tmp_path, 'b' * 64, size) is None
    assert read_mask(tmp_path, SHA, (800, 450)) is None
    assert mask_shape((300, 1800)) == (768, 128)


def test_concurrent_equal_publication_and_conflicting_writer(tmp_path):
    size = (800, 600)
    mask = np.ones(mask_shape(size), dtype=bool)
    with ThreadPoolExecutor(4) as workers:
        receipts = list(workers.map(lambda _: save_mask(tmp_path, SHA, size, mask), range(8)))
    assert len({r['mask_sha256'] for r in receipts}) == 1
    with pytest.raises(ValueError, match='disagree'):
        save_mask(tmp_path, SHA, size, ~mask)
    assert np.array_equal(read_mask(tmp_path, SHA, size), mask)
    assert not list(tmp_path.rglob('*.tmp'))


def test_invalid_id_shape_type_and_mutation_rejected(tmp_path):
    size = (700, 700)
    mask = np.ones(mask_shape(size), dtype=bool)
    with pytest.raises(ValueError, match='SHA256'):
        save_mask(tmp_path, '../a', size, mask)
    with pytest.raises(ValueError, match='dtype or dimensions'):
        save_mask(tmp_path, SHA, size, mask.astype(np.float32))
    with pytest.raises(ValueError, match='dtype or dimensions'):
        save_mask(tmp_path, SHA, size, mask[:, :7])
    save_mask(tmp_path, SHA, size, mask)
    path, = tmp_path.rglob('*.npz')
    with np.load(path, allow_pickle=False) as entry:
        metadata = entry['metadata']
    with path.open('wb') as stream:
        np.savez_compressed(stream, metadata=metadata, mask=np.zeros_like(mask, dtype=np.uint8))
    with pytest.raises(ValueError, match='hash mismatch'):
        read_mask(tmp_path, SHA, size)


def test_preparation_rechecks_original_and_reuses_pixels_without_model(tmp_path):
    photo = tmp_path / 'image.png'
    Image.new('RGB', (64, 32)).save(photo)
    digest = hashlib.sha256(photo.read_bytes()).hexdigest()
    f = {'frame_id': 4, 'recording_id': 'r', 'timestamp_s': .4, 'width': 64, 'height': 32, 'image_path': str(photo)}
    scene = {'source': 'predicted', 'tracks': [], 'frame': f, 'provenance': {'image_sha256': digest}}
    rows = [{'request_id': str(i), 'scene': scene, 'text': 'request', 'dialogue': [], 'input_mode': 'typed'} for i in [1, 2]]
    source = tmp_path / 'requests.jsonl'
    source.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    class FakeRoad:
        def __init__(self, folder):
            pass
        def mask(self, image, actual):
            assert actual == digest
            return np.ones(mask_shape(image.size), dtype=bool)
    first = prepare([source], tmp_path, tmp_path, tmp_path / 'one.json', encoder_factory=FakeRoad)
    assert first['counts'] == {'computed': 1, 'already_cached': 1}
    def fail_if_constructed(folder):
        pytest.fail('Unexpected model load on two cache hits')
    second = prepare([source], tmp_path, tmp_path, tmp_path / 'two.json', encoder_factory=fail_if_constructed)
    assert second['counts'] == {'already_cached': 2}
    Image.new('RGB', (64, 32), 'red').save(photo)
    third = prepare([source], tmp_path, tmp_path, tmp_path / 'three.json', encoder_factory=fail_if_constructed)
    assert third['counts'] == {'error': 2}
