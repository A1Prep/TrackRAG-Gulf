"""Fit and compare instruction-conditioned and language-blind destination fields.

This public-mask experiment prespecifies the endpoint epoch. Neither development
targets nor candidate geometries are available to prediction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from tools.dropoff_spatial_cache import public_rows
from trackrag.dropoff.contracts import validate_box, validate_point
from trackrag.dropoff.spatial import (InstructionVocabulary, MODEL_KIND, NoVisibleGround,
                                     geometry, make_network)
from trackrag.dropoff.spatial_vision import FEATURE_CHANNELS, GRID_HEIGHT, GRID_WIDTH

SEED = 1201
EPOCHS = 20
BATCH = 16


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, allow_nan=False, indent=2) + '\n', encoding='utf8')


def feature_batch(cases, cache):
    rows = []
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf8'))
    if manifest['recipe'] != json.loads((cache / 'recipe.json').read_text(encoding='utf8')):
        raise ValueError('Feature manifest belongs to another encoder recipe')
    for case in cases:
        image_sha = case['source']['image_sha256']
        content = (cache / (image_sha + '.npz')).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['features'].get(image_sha):
            raise ValueError('Cached visual features do not match their binding to the source image')
        with np.load(BytesIO(content), allow_pickle=False) as entry:
            features = entry['image']
            if features.dtype != np.uint8 or features.shape != (FEATURE_CHANNELS, GRID_HEIGHT, GRID_WIDTH):
                raise ValueError('Invalid compact vision features')
            rows.append(features)
    return np.stack(rows)


def target_arrays(cases):
    """Training targets only. Positive extent targets are true mask-bounding edges."""
    probs, bboxes = [], []
    yy, xx = np.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH].astype(np.float32)
    xx, yy = (xx + .5) / GRID_WIDTH, (yy + .5) / GRID_HEIGHT
    for row in cases:
        with Image.open(row['source']['mask_path']) as source:
            binary = source.convert('L').point(lambda value: 255 if value else 0)
            mask = np.asarray(binary.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.BOX),
                              dtype=np.float32) / 255
        frame, target = row['request']['scene']['frame'], row['target']
        px, py = np.array(target['destination_point_xy']) / [frame['width'], frame['height']]
        box = np.array(target['destination_box_xyxy']) / ([frame['width'], frame['height']] * 2)
        # Give the interior area more mass than the mask boundary without changing
        # which command and image the destination annotation belongs to.
        sx = max((box[2] - box[0]) * .30, .5 / GRID_WIDTH)
        sy = max((box[3] - box[1]) * .30, .5 / GRID_HEIGHT)
        center = np.exp(-.5 * (((xx - px) / sx) ** 2 + ((yy - py) / sy) ** 2))
        weights = mask * (.15 + center)
        if not weights.sum():
            weights[min(int(py * GRID_HEIGHT), GRID_HEIGHT - 1), min(int(px * GRID_WIDTH), GRID_WIDTH - 1)] = 1
        probs.append(weights / weights.sum())
        bboxes.append(box)
    return np.asarray(probs, dtype=np.float32), np.asarray(bboxes, dtype=np.float32)


def _batch_loss(model, image, words, mass, box):
    import torch
    import torch.nn.functional as F

    logits, extent = model(image, words)
    location_loss = -(mass.flatten(1) * logits.flatten(1).log_softmax(dim=1)).sum(dim=1).mean()
    x = (torch.arange(GRID_WIDTH, dtype=torch.float32)[None, None, :] + .5) / GRID_WIDTH
    y = (torch.arange(GRID_HEIGHT, dtype=torch.float32)[None, :, None] + .5) / GRID_HEIGHT
    offsets = ((x - box[:, 0, None, None]), (y - box[:, 1, None, None]),
               (box[:, 2, None, None] - x), (box[:, 3, None, None] - y))
    offsets = torch.stack([a.expand(-1, GRID_HEIGHT, GRID_WIDTH) for a in offsets], dim=1)
    loss = F.smooth_l1_loss(extent.log(), offsets.clamp_min(.004).log(), reduction='none')
    return location_loss + .65 * (loss.mean(dim=1) * mass).sum(dim=(1, 2)).mean()


def fit_condition(condition, training, vocab, train_features, target, output, hashes, epochs):
    import torch
    from safetensors.torch import load_file, save_file

    start = time.perf_counter()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = make_network(len(vocab.terms))
    initial = {k: v.clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025, weight_decay=.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.0003)
    words = np.array([[vocab.encode(row['request']['text'], mirror=mirror) for row in training]
                      for mirror in (False, True)], dtype=np.float32)
    if condition == 'language_blind':
        words.fill(0)
    heat, box = target
    losses = []
    steps = 0
    for epoch in range(epochs):
        ids = rng.permutation(len(training))
        epoch_total = 0.
        for start_index in range(0, len(ids), BATCH):
            batch = ids[start_index:start_index + BATCH]
            mirrored = rng.random(len(batch)) < .5
            features, gt, bounds = train_features[batch].copy(), heat[batch].copy(), box[batch].copy()
            features[mirrored] = features[mirrored, :, :, ::-1]
            gt[mirrored] = gt[mirrored, :, ::-1]
            bounds[mirrored, 0], bounds[mirrored, 2] = 1 - bounds[mirrored, 2], 1 - bounds[mirrored, 0]
            text = words[mirrored.astype(int), batch]
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, torch.from_numpy(features).float() / 255, torch.from_numpy(text),
                               torch.from_numpy(gt), torch.from_numpy(bounds))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite supervised grounding loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            epoch_total += float(loss.detach()) * len(batch)
            steps += 1
        scheduler.step()
        losses.append(epoch_total / len(training))
        if (epoch + 1) % 4 == 0 or epoch == 0:
            print(f'{condition} epoch {epoch + 1}/{epochs}, loss {losses[-1]:.4f}', flush=True)
    model.eval()
    checkpoint = output / condition
    checkpoint.mkdir()
    artifact = checkpoint / 'model.safetensors'
    save_file(model.state_dict(), str(artifact))
    changed = [key for key, value in model.named_parameters() if not torch.equal(value, initial[key])]
    if not changed:
        raise ValueError('Training did not update any parameters')
    meta = {'kind': MODEL_KIND, 'condition': condition, 'weights_sha256': digest(artifact),
            'vocabulary': vocab.terms, 'idf': vocab.weights, 'seed': SEED, 'batch_size': BATCH,
            'epochs': epochs, 'optimizer_steps': steps, 'training_cases': len(training),
            'updated_tensors': changed, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'loss_by_epoch': losses, 'train_elapsed_s': time.perf_counter() - start, **hashes}
    write_json(checkpoint / 'model.json', meta)
    # All scoring uses a freshly constructed network reloaded from serialized bytes.
    reloaded = make_network(len(vocab.terms))
    reloaded.load_state_dict(load_file(str(artifact)), strict=True)
    reloaded.eval()
    return meta, reloaded


def predict_batch(model, words, features, frames):
    """No reference mask, label, ID, source index or partition field is accepted."""
    import torch

    results = []
    with torch.inference_mode():
        for start in range(0, len(features), BATCH):
            batch = features[start:start + BATCH].astype(np.float32) / 255
            scores, extents = model(torch.from_numpy(batch), torch.from_numpy(words[start:start + BATCH]))
            for j, feat in enumerate(batch):
                try:
                    result = geometry(scores[j].numpy(), extents[j].numpy(), feat, frames[start + j])
                    results.append({'status': 'geometry', **result})
                except NoVisibleGround:
                    results.append({'status': 'no_visible_road'})
    return results


def _iou(a, b):
    overlap = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda q: (q[2] - q[0]) * (q[3] - q[1])
    return overlap / (area(a) + area(b) - overlap) if overlap else 0.


def score(saved, cases):
    if len(saved) != len(cases):
        raise ValueError('Missing prediction rows')
    if not saved or len({r['checkpoint_sha256'] for r in saved}) != 1:
        raise ValueError('Missing or mixed spatial checkpoints in scoring input')
    rows, groups = [], defaultdict(list)
    for pred, case in zip(saved, cases):
        if pred['case_id'] != case['case_id'] or pred['image_sha256'] != case['source']['image_sha256']:
            raise ValueError('Wrong prediction binding')
        frame = case['request']['scene']['frame']
        point = pred.get('destination_point_xy')
        if pred['status'] not in ('geometry', 'no_visible_road') or (
                (pred['status'] == 'geometry') != (point is not None)):
            raise ValueError('Invalid geometry prediction status')
        hit, overlap = False, 0.
        if point is not None:
            point = validate_point(point, frame['width'], frame['height'])
            box = validate_box(pred['destination_box_xyxy'], frame['width'], frame['height'])
            if not box[0] <= point[0] < box[2] or not box[1] <= point[1] < box[3]:
                raise ValueError('Mask-scoring point must be within its actual proposed box')
            with Image.open(case['source']['mask_path']) as mask:
                px = math.floor(point[0] * mask.width / frame['width'])
                py = math.floor(point[1] * mask.height / frame['height'])
                hit = bool(mask.convert('L').getpixel((px, py)))
            overlap = _iou(pred['destination_box_xyxy'], case['target']['destination_box_xyxy'])
        row = {'case_id': case['case_id'], 'point_in_mask': hit, 'box_iou': overlap,
               'produced_geometry': point is not None}
        rows.append(row)
        groups[case['source']['image_sha256']].append(row)
    contrasts = [g for g in groups.values() if len(g) > 1]
    return {'cases': len(rows), 'proposals': sum(r['produced_geometry'] for r in rows),
            'mask_hits': sum(r['point_in_mask'] for r in rows),
            'mean_box_iou_all_cases': float(np.mean([r['box_iou'] for r in rows])),
            'same_image_cases': sum(len(g) for g in contrasts), 'same_image_groups': len(contrasts),
            'same_image_hits': sum(r['point_in_mask'] for g in contrasts for r in g),
            'all_in_group_hits': sum(all(r['point_in_mask'] for r in g) for g in contrasts)}, rows


def run(train, development, cache, output, epochs=EPOCHS):
    import torch

    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    training, dev = public_rows(train, development)
    train_images, dev_images = feature_batch(training, cache), feature_batch(dev, cache)
    targets = target_arrays(training)
    vocab = InstructionVocabulary.fit([r['request']['text'] for r in training])
    output.mkdir(parents=True, exist_ok=False)
    hashes = {'train_sha256': digest(train), 'development_sha256': digest(development),
              'feature_recipe': json.loads((cache / 'recipe.json').read_text(encoding='utf8'))}
    protocol = {'mode': 'fixed_epochs_before_development_scoring', 'epochs': epochs,
                'train_cases': len(training), 'development_cases': len(dev), 'seed': SEED,
                'conditions': ['instruction_conditioned', 'language_blind'],
                'scoring': 'original native mask; point at predicted road cell center; original box IoU',
                'endpoint': 'final epoch, no development selection', 'external_test_used': False, **hashes}
    write_json(output / 'protocol.json', protocol)
    frames = [(r['request']['scene']['frame']['width'], r['request']['scene']['frame']['height']) for r in dev]
    report = {}
    for condition in protocol['conditions']:
        meta, network = fit_condition(condition, training, vocab, train_images, targets, output, hashes, epochs)
        words = np.stack([vocab.encode(r['request']['text']) for r in dev])
        if condition == 'language_blind':
            words.fill(0)
        predictions = predict_batch(network, words, dev_images, frames)
        bound = [{'case_id': row['case_id'], 'image_sha256': row['source']['image_sha256'],
                  'checkpoint_sha256': meta['weights_sha256'], **pred}
                 for row, pred in zip(dev, predictions, strict=True)]
        outcomes = output / condition / 'predictions.jsonl'
        outcomes.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in bound), encoding='utf8')
        measured, scores = score([json.loads(line) for line in outcomes.read_text().splitlines()], dev)
        write_json(output / condition / 'scores.json', scores)
        report[condition] = {**measured, 'checkpoint_sha256': meta['weights_sha256'],
                             'predictions_sha256': digest(outcomes), 'training_seconds': meta['train_elapsed_s']}
        write_json(output / 'results.json', report)
        print(condition, measured, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('train', 'development', 'cache', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error('--epochs must be between 1 and 100')
    run(args.train, args.development, args.cache, args.output, args.epochs)


if __name__ == '__main__':
    main()
