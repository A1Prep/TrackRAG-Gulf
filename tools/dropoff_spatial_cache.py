"""Cache compact image-only features from the two fixed public data partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from trackrag.dropoff.data import load_cases, validate_disjoint_groups
from trackrag.dropoff.spatial_vision import SpatialVision


def public_rows(train: Path, development: Path):
    training = load_cases(train, allowed_partitions={"train"})
    evaluating = load_cases(development, allowed_partitions={"development"})
    validate_disjoint_groups(training, evaluating)
    if any(row['source']['dataset'] != 'Talk2Car-RegSeg' for row in training + evaluating):
        raise ValueError('This benchmark requires the verified Talk2Car-RegSeg source and group guards')
    return training, evaluating


def save_features(path: Path, features: np.ndarray):
    # Quantizing in [0,1] retains the dominant semantic response while bounding the
    # private cache footprint. Inference deliberately uses the same transform.
    with path.with_suffix('.partial').open('wb') as stream:
        np.savez_compressed(stream, image=np.rint(features * 255).astype('uint8'))
    os.replace(path.with_suffix('.partial'), path)


def cache_images(train: Path, development: Path, output: Path, vision_cache: Path, *, verify_existing=False):
    import torch

    torch.set_num_threads(3)
    training, evaluating = public_rows(train, development)
    sites = {r['source']['image_sha256']: r['request']['scene']['frame']['image_path']
             for r in training + evaluating}
    output.mkdir(parents=True, exist_ok=True)
    # Bind every entry to both image bytes and the exact extraction implementation.
    from trackrag.dropoff import spatial_vision
    from tools import dropoff_ground_candidates
    source_hash = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    recipe = {'vision_source_sha256': source_hash(spatial_vision.__file__),
              'loader_source_sha256': source_hash(dropoff_ground_candidates.__file__),
              'weight_sha256': dropoff_ground_candidates.WEIGHTS_SHA256, 'version': 1}
    receipt = output / 'recipe.json'
    if receipt.exists() and json.loads(receipt.read_text()) != recipe:
        raise ValueError('The feature cache was created by a different extractor')
    if not receipt.exists() and any(output.iterdir()):
        raise ValueError('Refusing unversioned existing feature cache')
    receipt.write_text(json.dumps(recipe, indent=2), encoding='utf8')
    vision = SpatialVision(vision_cache)
    entries = {}
    for i, (digest, source) in enumerate(sites.items(), start=1):
        target = output / (digest + '.npz')
        if not target.exists():
            with Image.open(source) as image:
                save_features(target, vision.encode(image))
        elif verify_existing:
            with Image.open(source) as image, np.load(target, allow_pickle=False) as entry:
                fresh = np.rint(vision.encode(image) * 255).astype(np.uint8)
                if not np.array_equal(fresh, entry['image']):
                    raise ValueError('Cached features differ from a fresh original-image extraction: ' + digest)
        entries[digest] = source_hash(target)
        if i % 50 == 0 or i == len(sites):
            print(f'{i}/{len(sites)} verified public images', flush=True)
    manifest = {'recipe': recipe, 'features': entries, 'fresh_existing_features_verified': verify_existing}
    temporary = output / 'manifest.partial'
    temporary.write_text(json.dumps(manifest, sort_keys=True), encoding='utf8')
    os.replace(temporary, output / 'manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train', 'development', 'output', 'vision-cache'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true', help='Re-encode and compare every existing entry')
    args = parser.parse_args()
    cache_images(args.train, args.development, args.output, args.vision_cache, verify_existing=args.verify_existing)


if __name__ == '__main__':
    main()
