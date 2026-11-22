"""Torch-free, image- and recipe-bound cache of frozen passenger road masks."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np


ROAD_RECIPE = {
    'schema_version': 1, 'model': 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
    'revision': '21b3847fae21ddee674abd31129307b6a1235bd9',
    'weights_sha256': '027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c',
    'config_sha256': '649ba4afbaa11ee730b14ad5e31bd50f2e2d5be661a291e31a91512510dba8d9',
    'processor_sha256': 'd96ab2cb112985c19d5ea32f1d1f2369104fc78b1379f5ce26e33ca8646202a7',
    'input_size_hw': [384, 640], 'output_width': 512, 'maximum_output_height': 768,
    'output_resize': 'round nearest proportional size with minimum 1; bilinear logits align_corners=False',
    'decision': 'softmax(logits)[0,0] >= 0.5', 'color': 'RGB', 'target_kind': 'predicted_road_support',
}
ROAD_RECIPE_SHA256 = sha256(json.dumps(ROAD_RECIPE, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
SIDEWALK_RECIPE = {**ROAD_RECIPE, 'decision': 'softmax(logits)[0,1] >= 0.5',
                  'target_kind': 'predicted_sidewalk_support'}
SIDEWALK_RECIPE_SHA256 = sha256(json.dumps(SIDEWALK_RECIPE, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
_NAMESPACES = {ROAD_RECIPE_SHA256: 'passenger-road', SIDEWALK_RECIPE_SHA256: 'passenger-sidewalk'}


def mask_shape(size):
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)):
        raise ValueError('Original image size must contain two positive integers')
    width, height = size
    w = ROAD_RECIPE['output_width']
    h = max(1, round(w * height / width))
    if h > ROAD_RECIPE['maximum_output_height']:
        h = ROAD_RECIPE['maximum_output_height']
        w = max(1, round(h * width / height))
    return h, w


def mask_sha256(mask):
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.dtype != np.bool_:
        raise ValueError('A two-dimensional boolean road mask is required')
    return sha256(np.ascontiguousarray(arr).view(np.uint8).tobytes()).hexdigest()


def _path(cache, image_sha, size, *, recipe=ROAD_RECIPE_SHA256):
    mask_shape(size)
    if not isinstance(image_sha, str) or re.fullmatch(r'[a-f0-9]{64}', image_sha) is None:
        raise ValueError('Expected the SHA256 of the verified original image bytes')
    # One digest binds all key fields and avoids two adjacent 64-character
    # components exceeding the standard Windows filesystem path length.
    if recipe not in _NAMESPACES:
        raise ValueError('Unregistered passenger segmentation recipe')
    key = sha256(f'{recipe}:{image_sha}:{size[0]}:{size[1]}'.encode()).hexdigest()
    return Path(cache) / _NAMESPACES[recipe] / f'{key}.npz'


def _metadata(image_sha, size, mask, recipe=ROAD_RECIPE_SHA256):
    return {'recipe_sha256': recipe, 'image_sha256': image_sha,
            'original_size_wh': list(size), 'mask_shape_hw': list(mask.shape), 'mask_sha256': mask_sha256(mask)}


def read_mask(cache, image_sha, size, *, recipe=ROAD_RECIPE_SHA256):
    """A missing entry returns None. A damaged or foreign entry raises an error."""
    path = _path(cache, image_sha, size, recipe=recipe)
    try:
        stream = path.open('rb')
    except FileNotFoundError:
        return None
    shape = mask_shape(size)
    try:
        with stream, zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            if (len(entries) != 2 or {e.filename for e in entries} != {'metadata.npy', 'mask.npy'}
                    or any(e.file_size > (math.prod(shape) + 2048 if e.filename == 'mask.npy' else 2048)
                           for e in entries)):
                raise ValueError('Invalid stored road mask arrays or lengths')
            # Read bounded ZIP members and check NPY headers before allocation. A
            # corrupt header cannot request more data than the declared mask shape.
            from io import BytesIO
            def array(name, dtype, expected_shape=None):
                source = BytesIO(archive.read(name + '.npy'))
                if np.lib.format.read_magic(source) != (1, 0):
                    raise ValueError('Unexpected road cache array format')
                sizes, fortran, actual_dtype = np.lib.format.read_array_header_1_0(source)
                if actual_dtype != dtype or fortran or (expected_shape is not None and sizes != expected_shape):
                    raise ValueError('Road cache array type or dimensions disagree')
                raw = source.read()
                if not sizes or math.prod(sizes) != len(raw) or (expected_shape is None and len(sizes) != 1):
                    raise ValueError('Road cache array data length disagrees')
                return np.frombuffer(raw, dtype=dtype).reshape(sizes)
            stored = array('mask', np.dtype(np.uint8), shape)
            if (stored > 1).any():
                raise ValueError('Road mask contains values other than zero and one')
            mask = stored.astype(np.bool_)
            meta = json.loads(array('metadata', np.dtype(np.uint8)).tobytes())
            if meta != _metadata(image_sha, size, mask, recipe):
                raise ValueError('Road cache image, recipe, shape or mask hash mismatch')
            return mask
    except (zipfile.BadZipFile, EOFError, KeyError, UnicodeError) as exc:
        raise ValueError('Unreadable passenger road mask cache') from exc


def save_mask(cache, image_sha, size, mask, *, recipe=ROAD_RECIPE_SHA256):
    """Publish once atomically. Concurrent writers must agree on every pixel."""
    target = _path(cache, image_sha, size, recipe=recipe)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != mask_shape(size):
        raise ValueError('Road mask dtype or dimensions differ from the frozen recipe')
    meta = _metadata(image_sha, size, mask, recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='mask-', suffix='.tmp', delete=False) as out:
            temporary = Path(out.name)
            payload = json.dumps(meta, sort_keys=True, separators=(',', ':')).encode()
            np.savez_compressed(out, metadata=np.frombuffer(payload, dtype=np.uint8), mask=mask.astype(np.uint8))
            out.flush()
            os.fsync(out.fileno())
        try:
            # A same-volume hard link publishes the fully-written entry without
            # replacing an existing writer's result, including on Windows NTFS.
            os.link(temporary, target)
        except FileExistsError:
            current = read_mask(cache, image_sha, size, recipe=recipe)
            if current is None or not np.array_equal(current, mask):
                raise ValueError('Concurrent road masks disagree for the same image and recipe')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return meta
