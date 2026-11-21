"""A learned image-and-instruction destination density, separate from Qwen.

The public masks supervise geometry only. The caller owns dialogue and explicit
passenger confirmation. Peak scores are model diagnostics, not safety estimates.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from trackrag.dropoff.spatial_vision import FEATURE_CHANNELS, GRID_HEIGHT, GRID_WIDTH, SpatialVision


MODEL_KIND = 'segformer-instruction-density-v1'
VISION_CONFIG_HASHES = {
    'config.json': '649ba4afbaa11ee730b14ad5e31bd50f2e2d5be661a291e31a91512510dba8d9',
    'preprocessor_config.json': 'd96ab2cb112985c19d5ea32f1d1f2369104fc78b1379f5ce26e33ca8646202a7',
}

# Lexical normalization only. None of these entries encodes an image location
# or chooses a bounding box. The supervised weights learn all spatial responses.
_SYNONYMS = {'grey': 'gray', 'automobile': 'car', 'cars': 'car', 'lorry': 'truck',
             'lorries': 'truck', 'trucks': 'truck', 'vans': 'van', 'buses': 'bus',
             'bikes': 'bike', 'bicycle': 'bike', 'bicycles': 'bike', 'vehicles': 'vehicle',
             'pedestrians': 'pedestrian', 'people': 'person', 'alongside': 'beside'}
_FILLER = frozenset('a an the and i we our you your please could would should that this there '
                    'it is are have can to of want go pull stop park me my let us just'.split())


def instruction_terms(text: str, *, mirror=False):
    words = re.findall(r'[a-z]+|\d+', text.lower().replace('next to', 'beside'))
    words = [_SYNONYMS.get(word, word) for word in words if word not in _FILLER]
    if mirror:
        swap = {'left': 'right', 'right': 'left', 'leftmost': 'rightmost', 'rightmost': 'leftmost'}
        words = [swap.get(word, word) for word in words]
    return words + [first + '_' + second for first, second in zip(words, words[1:])]


class InstructionVocabulary:
    def __init__(self, terms, weights):
        self.terms, self.weights = list(terms), list(weights)
        if (not self.terms or len(self.terms) != len(self.weights)
                or len(set(self.terms)) != len(self.terms)
                or not all(isinstance(word, str) for word in self.terms)
                or not all(math.isfinite(w) and w > 0 for w in self.weights)):
            raise ValueError('Invalid spatial instruction vocabulary')
        self.indices = {word: i for i, word in enumerate(self.terms)}

    @classmethod
    def fit(cls, texts):
        frequencies = Counter()
        for text in texts:
            frequencies.update(set(instruction_terms(text)) | set(instruction_terms(text, mirror=True)))
        terms = sorted(word for word, count in frequencies.items() if count >= 3)
        return cls(terms, [1 + math.log((1 + len(texts)) / (1 + frequencies[word])) for word in terms])

    def encode(self, text, *, mirror=False):
        feature = np.zeros(len(self.terms), dtype=np.float32)
        for term in set(instruction_terms(text, mirror=mirror)):
            i = self.indices.get(term)
            if i is not None:
                feature[i] = self.weights[i]
        return feature / max(np.linalg.norm(feature), 1.)


def make_network(vocab_size):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DestinationNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            y, x = torch.meshgrid(torch.linspace(-1., 1., GRID_HEIGHT),
                                  torch.linspace(-1., 1., GRID_WIDTH), indexing='ij')
            self.register_buffer('xy', torch.stack([x, y, x * x, y * y, x * y]).unsqueeze(0),
                                 persistent=False)
            self.local = nn.Sequential(nn.Conv2d(27, 20, 3, padding=1), nn.GELU())
            self.middle = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(27, 20, 3, padding=1), nn.GELU(),
                                        nn.Conv2d(20, 20, 3, padding=1), nn.GELU())
            self.broad = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(27, 20, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(20, 20, 3, padding=1), nn.GELU())
            self.visual = nn.Sequential(nn.Conv2d(65, 32, 1), nn.GELU())
            self.text = nn.Linear(vocab_size, 32)
            self.text_prior = nn.Linear(vocab_size, 8 * 16, bias=False)
            self.base_prior = nn.Parameter(torch.zeros(1, 1, 8, 16))
            self.score = nn.Conv2d(32, 1, 1)
            self.extents = nn.Sequential(nn.Conv2d(32, 12, 1), nn.GELU(), nn.Conv2d(12, 4, 1))

        def forward(self, image, words):
            shape = (GRID_HEIGHT, GRID_WIDTH)
            xy = self.xy.expand(image.shape[0], -1, -1, -1)
            visual = torch.cat((image, xy), dim=1)
            up = lambda src: F.interpolate(src, size=shape, mode='bilinear', align_corners=False)
            local = self.visual(torch.cat((self.local(visual), up(self.middle(visual)),
                                           up(self.broad(visual)), xy), dim=1))
            # FiLM-style modulation and low-resolution language-dependent positional prior.
            condition = self.text(words).unsqueeze(-1).unsqueeze(-1)
            fused = local * (1 + condition)
            logits = self.score(fused) + (local * condition).sum(dim=1, keepdim=True)
            logits = logits + up(self.base_prior + self.text_prior(words).reshape(-1, 1, 8, 16))
            # Distances to the region edges are trained in fractions of image axes.
            extent = .003 + .25 * F.softplus(self.extents(fused))
            return logits[:, 0], extent

    return DestinationNetwork()


class NoVisibleGround(ValueError):
    """There is no predicted road-grid point outside the current exclusion boxes."""


def geometry(logits, extent, features, size, exclusion_boxes=()):
    width, height = size
    if min(width, height) < 1:
        raise ValueError('Image has empty dimensions')
    if (features.shape != (FEATURE_CHANNELS, GRID_HEIGHT, GRID_WIDTH) or not np.isfinite(features).all()
            or np.any(features < 0) or np.any(features > 1)):
        raise ValueError('Invalid spatial image features')
    if (logits.shape != (GRID_HEIGHT, GRID_WIDTH) or extent.shape != (4, GRID_HEIGHT, GRID_WIDTH)
            or not np.isfinite(logits).all() or not np.isfinite(extent).all() or np.any(extent <= 0)):
        raise ValueError('Invalid spatial model output')
    xs = (np.arange(GRID_WIDTH) + .5) * width / GRID_WIDTH
    ys = (np.arange(GRID_HEIGHT) + .5) * height / GRID_HEIGHT
    visible = (features[:19].argmax(axis=0) == 0) & (features[0] >= .45)
    for box in exclusion_boxes:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError('Invalid current-frame exclusion box')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Current-frame exclusion box exceeds the original frame')
        visible &= ~((xs[None, :] >= x1) & (xs[None, :] < x2)
                     & (ys[:, None] >= y1) & (ys[:, None] < y2))
    if not visible.any():
        raise NoVisibleGround('No predicted road point remains outside the supplied object boxes')
    supported = np.where(visible, logits, -np.inf)
    row, col = np.unravel_index(np.argmax(supported), supported.shape)
    x, y = float(xs[col]), float(ys[row])
    left, top, right, bottom = extent[:, row, col]
    box = [max(0., x - float(left) * width), max(0., y - float(top) * height),
           min(float(width), x + float(right) * width), min(float(height), y + float(bottom) * height)]
    peak_mass = 1 / np.exp(supported - supported[row, col]).sum()
    return {'destination_point_xy': [x, y], 'destination_box_xyxy': box,
            'peak_mass': float(peak_mass), 'point_road_probability': float(features[0, row, col]),
            'eligible_grid_cells': int(visible.sum())}


class SpatialGrounder:
    def __init__(self, checkpoint_path, *, vision_cache, expected_manifest_sha256=None):
        from safetensors.torch import load_file
        import torch

        folder = Path(checkpoint_path)
        manifest = (folder / 'model.json').read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest).hexdigest()
        if expected_manifest_sha256 is not None and expected_manifest_sha256 != self.manifest_sha256:
            raise ValueError('Spatial grounding manifest differs from configured SHA256')
        self.metadata = json.loads(manifest)
        weights = folder / 'model.safetensors'
        if (self.metadata.get('kind') != MODEL_KIND or
                self.metadata.get('condition') not in {'instruction_conditioned', 'language_blind'}):
            raise ValueError('Unsupported spatial grounding model or condition')
        if hashlib.sha256(weights.read_bytes()).hexdigest() != self.metadata['weights_sha256']:
            raise ValueError('Spatial grounding weights did not match the manifest')
        from tools import dropoff_ground_candidates
        from trackrag.dropoff import spatial_vision
        recipe = self.metadata['feature_recipe']
        for key, module in (('vision_source_sha256', spatial_vision),
                            ('loader_source_sha256', dropoff_ground_candidates)):
            if recipe[key] != hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest():
                # Git may check out either LF or CRLF on the target OS. Verify
                # the sibling serialization of exactly the same source text.
                normalized = Path(module.__file__).read_bytes().replace(b'\r\n', b'\n')
                variants = (hashlib.sha256(normalized).hexdigest(),
                            hashlib.sha256(normalized.replace(b'\n', b'\r\n')).hexdigest())
                if recipe[key] not in variants:
                    raise ValueError('Spatial vision preprocessing differs from the training recipe')
        if recipe['weight_sha256'] != dropoff_ground_candidates.WEIGHTS_SHA256:
            raise ValueError('Spatial grounder uses a different frozen visual encoder')
        if not all((Path(vision_cache) / name).is_file()
                   for name in ('pytorch_model.bin', 'config.json', 'preprocessor_config.json')):
            raise FileNotFoundError('Pinned SegFormer model cache is incomplete')
        for name, expected in VISION_CONFIG_HASHES.items():
            if hashlib.sha256((Path(vision_cache) / name).read_bytes()).hexdigest() != expected:
                raise ValueError('SegFormer configuration differs from the frozen spatial training condition')
        self.vocab = InstructionVocabulary(self.metadata['vocabulary'], self.metadata['idf'])
        self.model = make_network(len(self.vocab.terms))
        self.model.load_state_dict(load_file(str(weights)), strict=True)
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('Nonfinite spatial grounding parameters')
        self.model.eval()
        self.vision = SpatialVision(vision_cache)

    def predict(self, image: Image.Image, text: str, *, exclusion_boxes=()):
        import torch

        if not isinstance(text, str) or not text.strip():
            raise ValueError('Passenger instruction is required')
        f = np.rint(self.vision.encode(image) * 255).astype(np.float32) / 255
        words = self.vocab.encode(text)
        if self.metadata['condition'] == 'language_blind':
            words.fill(0)
        with torch.inference_mode():
            logits, extent = self.model(torch.from_numpy(f[None]), torch.from_numpy(words[None]))
        result = geometry(logits[0].numpy(), extent[0].numpy(), f, image.size, exclusion_boxes)
        return {**result, 'model_kind': MODEL_KIND, 'weights_sha256': self.metadata['weights_sha256']}
