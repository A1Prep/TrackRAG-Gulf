"""Post-fit language intervention on the frozen 125-command development cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.dropoff_spatial_fit import digest, feature_batch, predict_batch, score, write_json
from trackrag.dropoff.data import load_cases
from trackrag.dropoff.spatial import InstructionVocabulary, make_network


def alternate_sources(rows):
    """Fixed deterministic derangement, excluding same-scene or equivalent text."""
    sort_key = lambda i: hashlib.sha256(('spatial-language-control-1:' + rows[i]['case_id']).encode()).digest()
    order = sorted(range(len(rows)), key=sort_key)
    for offset in range(1, len(rows)):
        shifted = order[offset:] + order[:offset]
        if all(rows[a]['scene_group'] != rows[b]['scene_group'] and
               rows[a]['request']['text'].casefold().strip() != rows[b]['request']['text'].casefold().strip()
               for a, b in zip(order, shifted)):
            return dict(zip(order, shifted))
    raise ValueError('No valid deterministic cross-scene instruction permutation')


def run(development, cache, fit, output):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(3)
    output.mkdir(parents=True, exist_ok=False)
    proto = json.loads((fit / 'protocol.json').read_text(encoding='utf8'))
    if digest(development) != proto['development_sha256']:
        raise ValueError('Development manifest differs from the fit')
    directory = fit / 'instruction_conditioned'
    meta = json.loads((directory / 'model.json').read_text(encoding='utf8'))
    if digest(directory / 'model.safetensors') != meta['weights_sha256']:
        raise ValueError('Spatial model checkpoint digest changed')
    rows = load_cases(development, allowed_partitions={'development'})
    features = feature_batch(rows, cache)
    assignment = alternate_sources(rows)
    vocab = InstructionVocabulary(meta['vocabulary'], meta['idf'])
    model = make_network(len(vocab.terms))
    model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
    model.eval()
    frames = [(r['request']['scene']['frame']['width'], r['request']['scene']['frame']['height']) for r in rows]
    originals = [json.loads(s) for s in (directory / 'predictions.jsonl').read_text().splitlines()]
    blind = [json.loads(s) for s in (fit / 'language_blind' / 'predictions.jsonl').read_text().splitlines()]
    vocabulary_inputs = np.stack([vocab.encode(rows[assignment[i]]['request']['text']) for i in range(len(rows))])
    predictions = predict_batch(model, vocabulary_inputs, features, frames)
    altered = [{'case_id': r['case_id'], 'image_sha256': r['source']['image_sha256'],
                'checkpoint_sha256': meta['weights_sha256'], 'utterance_source': rows[assignment[i]]['case_id'],
                **p} for i, (r, p) in enumerate(zip(rows, predictions, strict=True))]
    (output / 'shuffled.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in altered),
                                          encoding='utf8')
    full, fs = score(originals, rows)
    shifted, ss = score(altered, rows)
    lang_blind, bs = score(blind, rows)
    def pairs(comparator):
        return {'original_only_hits': sum(a['point_in_mask'] and not b['point_in_mask'] for a, b in zip(fs, comparator)),
                'comparator_only_hits': sum(b['point_in_mask'] and not a['point_in_mask'] for a, b in zip(fs, comparator))}
    result = {'public_development_cases': len(rows), 'original': full, 'separately_trained_blind': lang_blind,
              'cross_scene_shuffled_instruction': shifted, 'original_vs_blind': pairs(bs),
              'original_vs_shuffled': pairs(ss), 'same_scene_command_reuse': False,
              'same_instruction_string_reuse': False, 'weights_sha256': meta['weights_sha256'],
              'shuffled_predictions_sha256': digest(output / 'shuffled.jsonl'),
              'scope': 'frozen model, previously used public development scenes; no final holdout'}
    write_json(output / 'result.json', result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('development', 'cache', 'fit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.development, args.cache, args.fit, args.output)


if __name__ == '__main__':
    main()
