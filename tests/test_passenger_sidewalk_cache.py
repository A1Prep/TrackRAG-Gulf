from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff.road_cache import SIDEWALK_RECIPE_SHA256, _path, mask_shape, read_mask, save_mask
from trackrag.dropoff.sidewalk import mask_for


def test_sidewalk_and_road_keep_distinct_receipts_for_the_same_photo(tmp_path):
    size = (640, 360)
    photo = sha256(b"same original").hexdigest()
    road = np.zeros(mask_shape(size), dtype=bool)
    sidewalk = ~road
    save_mask(tmp_path, photo, size, road)
    image = Image.new("RGB", size)
    source = SimpleNamespace(cache=tmp_path, image_hash=photo)
    assert mask_for(source, image) is None
    receipt = save_mask(tmp_path, photo, size, sidewalk, recipe=SIDEWALK_RECIPE_SHA256)
    assert receipt["recipe_sha256"] == SIDEWALK_RECIPE_SHA256
    assert np.array_equal(read_mask(tmp_path, photo, size), road)
    assert np.array_equal(mask_for(source, image), sidewalk)
    with pytest.raises(ValueError, match="different original"):
        mask_for(source, image, sha256(b"another current frame").hexdigest())
    other = SimpleNamespace(cache=tmp_path, image_hash=sha256(b"another photo").hexdigest())
    assert mask_for(other, image) is None
    # An ordinary road decision copied to the sidewalk slot is not valid sidewalk evidence.
    alternate = _path(tmp_path, photo, size, recipe=SIDEWALK_RECIPE_SHA256)
    alternate.write_bytes(_path(tmp_path, photo, size).read_bytes())
    with pytest.raises(ValueError, match="recipe"):
        mask_for(source, image)
