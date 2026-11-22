"""Precompute predicted road support once per verified passenger image, on CPU."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from tools.dropoff_predict import read_requests
from trackrag.dropoff.landmarks import resolve_image
from trackrag.dropoff.model import GroundingConfig
from trackrag.dropoff.passenger import RoadEvidence
from trackrag.dropoff.road_cache import ROAD_RECIPE_SHA256, mask_sha256, read_mask, save_mask


def prepare(request_paths, vision_cache, cache, report, *, encoder_factory=RoadEvidence):
    if report.exists():
        raise FileExistsError('Choose a fresh road-preparation receipt')
    reports, inputs, computed = [], {}, set()
    encoder = None
    config = GroundingConfig()
    for path in request_paths:
        content = path.read_bytes()
        inputs[str(path)] = sha256(content).hexdigest()
        for request in read_requests(path, payload=content):
            key = request['request_id']
            row = {'request_id': key, 'recording_id': request['scene']['frame']['recording_id'],
                   'frame_id': request['scene']['frame']['frame_id']}
            try:
                image, binding = resolve_image(request, config)
                with image:
                    image_sha = binding['sha256']
                    row.update(image_sha256=image_sha, original_size_wh=list(image.size))
                    mask = read_mask(cache, image_sha, image.size)
                    if mask is None:
                        if encoder is None:
                            if encoder_factory is RoadEvidence:
                                import torch
                                torch.set_num_threads(2)
                            encoder = encoder_factory(vision_cache)
                        mask = encoder.mask(image, image_sha)
                        save_mask(cache, image_sha, image.size, mask)
                        computed.add(image_sha)
                        row['source'] = 'computed'
                    else:
                        row['source'] = 'already_cached'
                    row.update(mask_sha256=mask_sha256(mask), predicted_road_pixels=int(mask.sum()),
                               mask_shape_hw=list(mask.shape), ok=True)
            except (ValueError, OSError, RuntimeError, ImportError) as exc:
                row.update(ok=False, error={'type': type(exc).__name__, 'message': str(exc)})
            reports.append(row)
    result = {'recipe_sha256': ROAD_RECIPE_SHA256, 'inputs_sha256': inputs,
              'unique_images_computed': len(computed), 'requests': len(reports),
              'counts': dict(Counter(r.get('source', 'error') for r in reports)), 'results': reports}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', type=Path, nargs='+', required=True)
    p.add_argument('--vision-cache', type=Path, required=True, help='Pinned SegFormer files')
    p.add_argument('--cache', type=Path, help='Defaults to the same folder as the pinned vision model')
    p.add_argument('--report', type=Path, required=True)
    opts = p.parse_args()
    result = prepare(opts.requests, opts.vision_cache, opts.cache or opts.vision_cache, opts.report)
    print(json.dumps({k: result[k] for k in ('requests', 'unique_images_computed', 'counts')}))
    return int(any(not r['ok'] for r in result['results']))


if __name__ == '__main__':
    raise SystemExit(main())
