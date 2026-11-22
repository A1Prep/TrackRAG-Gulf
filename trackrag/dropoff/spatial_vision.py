"""Frozen Cityscapes image features for the passenger spatial learner."""
from pathlib import Path

import numpy as np
from PIL import Image


GRID_HEIGHT, GRID_WIDTH = 32, 64
IMAGE_SIZE = {"height": 320, "width": 640}
FEATURE_CHANNELS = 22  # 19 Cityscapes probabilities plus RGB


class SpatialVision:
    """Extract scene appearance from the supplied current RGB frame only."""

    def __init__(self, cache: str | Path):
        from tools.dropoff_ground_candidates import prepare_model

        self.model, self.processor = prepare_model(cache)

    def encode(self, image: Image.Image) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, size=IMAGE_SIZE, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = F.interpolate(logits, (GRID_HEIGHT, GRID_WIDTH), mode="bilinear",
                                          align_corners=False).softmax(dim=1)[0].numpy()
        reduced = rgb.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.BOX)
        colors = np.asarray(reduced, dtype=np.float32).transpose(2, 0, 1) / 255
        return np.concatenate((probabilities, colors), axis=0).astype(np.float32)
