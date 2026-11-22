"""Run the frozen spatial learner on bound scene requests. Geometry diagnostics only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from tools.dropoff_predict import read_requests
from trackrag.dropoff.model import GroundingConfig, _image_snapshot
from trackrag.dropoff.spatial import NoVisibleGround, SpatialGrounder


def run(requests, checkpoint, vision_cache, output, expected_manifest_sha256=None):
    import torch

    torch.set_num_threads(3)
    rows = read_requests(requests)
    if any(row['scene']['source'] != 'predicted' for row in rows):
        raise ValueError('The deployment diagnostic requires predicted-scene inputs')
    output.mkdir(parents=True, exist_ok=False)
    model = SpatialGrounder(checkpoint, vision_cache=vision_cache,
                            expected_manifest_sha256=expected_manifest_sha256)
    outcomes = []
    for row in rows:
        start = time.perf_counter()
        frame = row['scene']['frame']
        record = {'request_id': row['request_id'], 'recording_id': frame['recording_id'],
                  'frame_id': frame['frame_id'], 'input_mode': row['input_mode'],
                  'text': row['text'], 'model_manifest_sha256': model.manifest_sha256,
                  'model_weights_sha256': model.metadata['weights_sha256'], 'dialogue_decided': False}
        try:
            image, binding = _image_snapshot(frame, GroundingConfig())
            with image:
                expected = row['scene'].get('provenance', {}).get('image_sha256')
                if expected is None or binding['sha256'] != expected:
                    raise ValueError('Scene needs the matching original image SHA256')
                current = [t for t in row['scene']['tracks'] if t['frame_id'] == frame['frame_id']]
                record.update(image_sha256=binding['sha256'], current_exclusions=len(current))
                result = model.predict(image, row['text'], exclusion_boxes=[t['bbox_xyxy'] for t in current])
            record.update(outcome='geometry_only', prediction=result)
        except NoVisibleGround as exc:
            record.update(outcome='no_predicted_road', explanation=str(exc))
        except (OSError, ValueError, RuntimeError) as exc:
            record.update(outcome='operational_error', error_type=type(exc).__name__, explanation=str(exc))
        record['elapsed_s'] = time.perf_counter() - start
        outcomes.append(record)
    data = output / 'outcomes.jsonl'
    data.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in outcomes), encoding='utf8')
    receipt = {'cases': len(outcomes), 'source_file_sha256': hashlib.sha256(requests.read_bytes()).hexdigest(),
               'outcomes_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
               'counts': {kind: sum(r['outcome'] == kind for r in outcomes) for kind in
                          ('geometry_only', 'no_predicted_road', 'operational_error')},
               'semantic_status_tested': False, 'audio_transcribed_here': False,
               'scope': 'current-image geometry without a dialogue or authorization decision'}
    (output / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    print(json.dumps(receipt))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('requests', 'checkpoint', 'vision-cache', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256')
    args = parser.parse_args()
    run(args.requests, args.checkpoint, args.vision_cache, args.output, args.expected_manifest_sha256)


if __name__ == '__main__':
    main()
