"""Sidewalk pixels for corroborating a parking-row boundary on the same road photograph."""
from hashlib import sha256
from pathlib import Path

from .road_cache import ROAD_RECIPE, SIDEWALK_RECIPE_SHA256, mask_shape, read_mask, save_mask


def mask_for(road, image, image_sha=None):
    """Read the sidewalk for the verified original while holding the shared grounder lock."""
    cache = getattr(road, "cache", None)
    fingerprint = getattr(road, "image_hash", None)
    if cache is None or fingerprint is None:
        return None
    if image_sha is not None and fingerprint != image_sha:
        raise ValueError("Sidewalk and road evidence belong to different original photographs")
    if getattr(road, "_sidewalk_key", None) == (fingerprint, image.size):
        return road._sidewalk_mask
    cached = read_mask(cache, fingerprint, image.size, recipe=SIDEWALK_RECIPE_SHA256)
    if cached is not None:
        return cached
    from .passenger import RoadEvidence
    # Read-only verification can operate on the saved probability decision alone.
    if not isinstance(road, RoadEvidence) or not (Path(cache) / "pytorch_model.bin").is_file():
        return None
    if road.model is None:
        from tools.dropoff_ground_candidates import prepare_model
        for filename, key in (("config.json", "config_sha256"), ("preprocessor_config.json", "processor_sha256")):
            if sha256((Path(cache) / filename).read_bytes()).hexdigest() != ROAD_RECIPE[key]:
                raise ValueError("Segmentation configuration differs from the registered sidewalk recipe")
        road.model, road.processor = prepare_model(cache)
    if road.model.config.id2label.get(1) != "sidewalk":
        raise ValueError("Segmentation model class 1 must be sidewalk")
    import torch
    from torch.nn.functional import interpolate
    height, width = ROAD_RECIPE["input_size_hw"]
    batch = road.processor(images=image, size={"height": height, "width": width}, return_tensors="pt")
    with torch.inference_mode():
        logits = interpolate(road.model(**batch).logits, mask_shape(image.size), mode="bilinear", align_corners=False)
        mask = (logits.softmax(1)[0, 1] >= 0.5).cpu().numpy()
    save_mask(cache, fingerprint, image.size, mask, recipe=SIDEWALK_RECIPE_SHA256)
    road._sidewalk_key, road._sidewalk_mask = (fingerprint, image.size), mask
    return mask
