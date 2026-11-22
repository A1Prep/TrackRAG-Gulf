"""Experimental AI-supervised continuation of the public 8B adapter on one EMT recording.

This training cohort is deliberately distinct from the human-reviewed adaptation
runner. Geometry on independently disputed rows never enters either model input
or completion; these rows teach the shared status prefix only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from functools import wraps
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time

from tools.dropoff_adapt_local import FIELD_ORDER, parent_context
from tools.dropoff_fit_local import supervised_forward
from tools.dropoff_predict import read_requests
from tools.dropoff_train import GroundingCollator, adapter_parameter_hashes, package_versions, write_json
from trackrag.dropoff.contracts import FULL_FIELDS, validate_point, validate_target
from trackrag.dropoff.data import inference_input_sha256
from trackrag.dropoff.formatting import build_messages, target_to_model
from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION, _image_snapshot, verify_adapter

MODEL_STAGE = 'experimental_ai_supervised_direct_interpretation'
TRAIN_RECORDING = 'video_054604'
IDS = tuple(f'passenger-a{i:02}' for i in range(1, 18))
PINS = {'train-references.jsonl': 'b00053cdc2c05a58d5b688f54f1c06deb1cedbf99f555d5ea81696dfba570525',
        'train-instructions.jsonl': 'f6374fe85c21a85a1eafb16d46586285ff5d05afd837223520a9224069627a5d',
        'input-bindings.jsonl': '7d751e255bb9114cd7160e7c85b2b44f3cb235a33a497ef2f07ab48c10321e37',
        'second-ai-review.jsonl': '10e86e996c433ee7f2eeba80961cfcf50595efee0d5b477f651550de9988637e'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_sha(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pinned_rows(path):
    content = path.read_bytes().replace(b'\r\n', b'\n')
    _require(sha256(content).hexdigest() == PINS[path.name], 'Changed frozen AI reference or input binding: ' + path.name)
    selected = {}
    for line in content.splitlines():
        row = json.loads(line)
        key = row['request_id']
        # The immutable review and portable input bindings also contain independent
        # development rows. Only the exact train IDs can leave this loader.
        if key in IDS:
            _require(key not in selected, 'Duplicate frozen training identity')
            selected[key] = row
    _require(set(selected) == set(IDS), 'Missing frozen training identities')
    return selected


def prepare(args):
    parent_config, public, parent = parent_context(args.parent_run, args.config)
    refs, utterances, binding, reviews = [pinned_rows(args.annotations / name) for name in PINS]
    requests = read_requests(args.requests)
    _require([r['request_id'] for r in requests] == list(IDS), 'Only the 17 frozen training requests may enter adaptation')
    parent_images = {r['image_sha256'] for r in public['case_artifacts']}
    cases, audit = [], []
    for req in requests:
        key = req['request_id']
        frame, prov = req['scene']['frame'], req['scene'].get('provenance', {})
        ref, words, bound, reviewer = refs[key], utterances[key], binding[key], reviews[key]
        image_sha = prov.get('image_sha256')
        _require(frame['recording_id'] == ref['recording_id'] == words['recording_id'] == bound['recording_id'] == TRAIN_RECORDING
                 and frame['frame_id'] == ref['frame_id'] == words['frame_id'] == bound['frame_id']
                 and ref['partition'] == 'train' and req['input_mode'] == 'typed'
                 and req['text'] == words['text'] and req['dialogue'] == words.get('dialogue', [])
                 and image_sha == bound['image_sha256'] == ref['image_sha256'] and image_sha not in parent_images
                 and inference_input_sha256(req, image_sha) == bound['inference_input_sha256']
                 and prov.get('tracks_sha256') == bound['saved_tracks_sha256']
                 and prov.get('prediction_provenance_sha256') == bound['saved_prediction_provenance_sha256'],
                 'Incorrect source image, frame, transcript, prediction or recording for ' + key)
        image, snap = _image_snapshot(frame, parent_config)
        image.close()
        _require(snap['sha256'] == image_sha, 'Changed local training image')
        target = validate_target(ref['target'], sorted(FULL_FIELDS), frame)
        _require(target['status'] == reviewer['status'], 'Independent AI status assessments disagree')
        full = True
        if target['status'] == 'proposed':
            x, y = validate_point(reviewer['point_xy'], frame['width'], frame['height'])
            x1, y1, x2, y2 = target['destination_box_xyxy']
            full = x1 <= x < x2 and y1 <= y < y2
        if full:
            norm = target_to_model(target, sorted(FULL_FIELDS), frame)
            completion = json.dumps({k: norm[k] for k in FIELD_ORDER}, separators=(',', ':'), allow_nan=False)
        else:
            completion = '{"status":"proposed"'
        cases.append({'case_id': key, 'request': req, 'completion': completion, 'full_target': full,
                      '_training_image_sha256': image_sha})
        audit.append({'request_id': key, 'training_recording': TRAIN_RECORDING, 'frame_id': frame['frame_id'],
                      'input_sha256': bound['inference_input_sha256'], 'image_sha256': image_sha,
                      'status': target['status'], 'supervision': 'full' if full else 'agreed_status_prefix_only',
                      'annotator_disagreement_on_spatial_target': not full,
                      'target_text_sha256': sha256(completion.encode()).hexdigest()})
    _require(sum(c['full_target'] for c in cases) == 11 and len({r['status'] for r in audit}) == 3,
             'Frozen independent comparison or class counts changed')
    # Keep original pixels, all observations, coordinates and prompt serialization.
    # The eligibility of this larger context is measured with the real processor;
    # only the 8-resident suffix remains on the laptop GPU during this stage.
    return replace(parent_config, max_input_tokens=3072, inference_resident_layers=8,
                   observation_format='compact_rows_v1'), parent, cases, audit


def training_messages(case):
    return build_messages(case['request'], task='interpret', observation_format='compact_rows_v1') + [
        {'role': 'assistant', 'content': case['completion']}]


class AICollator(GroundingCollator):
    def __init__(self, processor, max_sequence_length, *, expected=None):
        super().__init__(processor, max_sequence_length, message_builder=training_messages)
        self.expected = expected

    def __call__(self, cases):
        batch = super().__call__(cases)
        for i, case in enumerate(cases):
            if case['full_target']:
                continue
            positions = (batch['labels'][i] != -100).nonzero(as_tuple=False).flatten()
            ids = self.processor.tokenizer.encode(case['completion'], add_special_tokens=False)
            _require(0 < len(ids) < len(positions) and batch['labels'][i, positions[:len(ids)]].tolist() == ids,
                     'Disputed-row status does not align to a strict supervised token prefix')
            batch['labels'][i, positions[len(ids):]] = -100  # never teach an early stop after status
        if self.expected is not None:
            for i, case in enumerate(cases):
                row = self.expected[case['case_id']]
                _require(int(batch['attention_mask'][i].sum()) == row['full_input_tokens']
                         and int((batch['labels'][i] != -100).sum()) == row['target_tokens'],
                         'Runtime sequence differs from the verified CPU preflight')
        return batch


def validate(args):
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(2)
    config, parent, cases, audit = prepare(args)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION,
        cache_dir=str(config.resolved_cache_dir()), local_files_only=True, trust_remote_code=False,
        min_pixels=config.min_pixels, max_pixels=config.max_pixels)
    collator = AICollator(processor, config.max_input_tokens)
    for case, row in zip(cases, audit, strict=True):
        batch = collator([case])
        row['full_input_tokens'] = int(batch['attention_mask'].sum())
        row['target_tokens'] = int((batch['labels'] != -100).sum())
        row['image_tokens'] = int((batch['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum())
        _require(row['image_tokens'] > 0, 'Actual image tokens are missing')
    result = {'kind': MODEL_STAGE, 'mode': 'actual_cpu_processor_preflight', 'pins_lf_sha256': PINS,
              'base_model': config.model_id, 'base_revision': config.revision, 'parent_sha256': _json_sha(parent),
              'gpu_execution': False, 'human_reviewed': False, 'training_recordings': [TRAIN_RECORDING],
              'development_cases_used': 0, 'blind_third_cases_used': 0, 'reserved_final_cases_used': 0,
              'source': 'scripted AI author plus independent blind AI review; no real passenger intentions',
              'coarse_box_and_clarification_wording_authorship': 'primary AI reviewer only; second assessed statuses and proposed points',
              'rows': audit, 'token_budget': config.max_input_tokens, 'truncated': False,
              'observation_format': config.observation_format}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('CPU preflight:', len(audit), 'requests;', sum(c['full_target'] for c in cases), 'full targets;',
          'input length', min(r['full_input_tokens'] for r in audit), '..', max(r['full_input_tokens'] for r in audit), flush=True)
    return result


def save_adapter(runtime, output, initial, progress, corpus_sha):
    directory = output / 'adapter'
    directory.mkdir()
    config = runtime.config
    runtime.model.peft_config['default'].base_model_name_or_path = config.model_id
    runtime.model.peft_config['default'].revision = config.revision
    runtime.model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    runtime.processor.save_pretrained(directory)
    hashes = adapter_parameter_hashes(runtime.model)
    _require(set(hashes) == set(initial) and all(len(v) == 64 for v in hashes.values()),
             'Learned adapter parameter identities changed')
    changed = sum(initial[k] != value for k, value in hashes.items())
    _require(changed > 0, 'No adapter tensor changed')
    files = {name: sha256((directory / name).read_bytes()).hexdigest()
             for name in ('adapter_model.safetensors', 'adapter_config.json')}
    manifest = {'schema_version': 1, 'artifact_kind': 'lora_adapter', 'run_kind': MODEL_STAGE,
                'base_model_id': config.model_id, 'base_revision': config.revision, 'files': files,
                'coordinate_system': 'normalized_0_1000_image_edges', 'suffix_layers': config.suffix_layers,
                'image_processing': {'min_pixels': config.min_pixels, 'max_pixels': config.max_pixels},
                'training_stage': MODEL_STAGE, 'training_manifest_sha256': corpus_sha,
                'observation_format': config.observation_format,
                'counter_scope': 'AI_supervised_child_only', 'human_reviewed': False,
                'target_encoding': 'full_interpretation_or_status_prefix_v1',
                'changed_adapter_tensors': changed, 'adapter_parameter_hashes': hashes, **progress}
    write_json(directory / 'grounding_adapter_manifest.json', manifest)
    child_config = replace(config, adapter_dir=str(directory.resolve()))
    verify_adapter(child_config)
    write_json(output / 'inference_config.json', asdict(child_config))
    return manifest


def checkpoint_resident_suffix(layers):
    """Recompute only the resident LoRA decoder blocks during backward.

    Do not enable PEFT's global input gradient hook or checkpoint the streamed
    frozen prefix. The latter's weights are deliberately offloaded after use.
    Non-reentrant checkpointing trains resident LoRA weights even when the
    first suffix block receives a frozen input with requires_grad=False.
    """
    from torch.utils.checkpoint import checkpoint

    def wrapped(forward):
        @wraps(forward)
        def run(*args, **kwargs):
            return checkpoint(forward, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
        return run

    for layer in layers:
        _require(any(p.requires_grad for p in layer.parameters()), 'Checkpointing a frozen block is disallowed')
        layer.forward = wrapped(layer.forward)


def train(args):
    import torch
    from trackrag.dropoff.streamed import StreamedRuntime

    _require(args.preflight is not None, 'Actual CPU token preflight file is required')
    preflight = json.loads(args.preflight.read_bytes())
    config, parent, cases, audit = prepare(args)
    _require(preflight.get('kind') == MODEL_STAGE and preflight.get('mode') == 'actual_cpu_processor_preflight'
             and preflight.get('observation_format') == config.observation_format
             and preflight.get('pins_lf_sha256') == PINS and preflight.get('parent_sha256') == _json_sha(parent)
             and preflight.get('human_reviewed') is False and preflight.get('token_budget') == config.max_input_tokens
             and len(preflight.get('rows', [])) == len(audit), 'CPU preflight does not match this continuation')
    for validated, actual in zip(preflight['rows'], audit, strict=True):
        _require(all(validated.get(k) == v for k, v in actual.items())
                 and 0 < validated['target_tokens'] < validated['full_input_tokens'] <= config.max_input_tokens
                 and 0 < validated['image_tokens'] < validated['full_input_tokens'],
                 'CPU preflight identity or token binding changed')
    _require(2 <= args.epochs <= 4, 'Use the bounded, predeclared two- to four-epoch schedule')
    _require(torch.cuda.is_available(), 'Exclusive access to the configured CUDA GPU is required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.90, 0)
    torch.manual_seed(42)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    corpus = {'stage': MODEL_STAGE, 'human_reviewed': False, 'rows': audit,
              'references_lf_sha256': PINS, 'actual_preflight_sha256': sha256(args.preflight.read_bytes()).hexdigest(),
              'training_tool_sha256': sha256(source.read_bytes()).hexdigest(), 'seed': 42,
              'optimizer': 'AdamW', 'learning_rate': 5e-5, 'weight_decay': .01, 'accumulation': 2,
              'activation_checkpointing': 'nonreentrant, last eight already-resident decoder blocks only',
              'training_attention': 'query_tiles_64_recompute_backward' if args.tiled_attention else 'native_sdpa',
              'epochs': args.epochs, 'selection': 'fixed final epoch; no development or final case used',
              'config': asdict(config), 'public_parent_binding_sha256': _json_sha(parent)}
    repository = source.resolve().parents[1]
    corpus['implementation_sha256'] = {name: sha256((repository / name).read_bytes()).hexdigest() for name in (
        'tools/dropoff_ai_continuation.py', 'tools/dropoff_train.py', 'tools/dropoff_fit_local.py',
        'tools/dropoff_adapt_local.py', 'trackrag/dropoff/formatting.py', 'trackrag/dropoff/model.py',
        'trackrag/dropoff/streamed.py', 'trackrag/dropoff/streamed_edges.py', 'trackrag/dropoff/streamed_weights.py',
        'trackrag/dropoff/training_attention.py')}
    write_json(output / 'training_manifest.json', corpus)
    write_json(output / 'parent_binding.json', parent)
    write_json(output / 'parent_inference_config.json', asdict(config))
    (output / 'training_source.py').write_bytes(source.read_bytes())
    progress = {'epochs_completed': 0, 'optimizer_steps': 0, 'actual_training_examples_consumed': 0}
    consumed = []
    runtime = None
    started = time.perf_counter()
    try:
        runtime = StreamedRuntime(config, training_lora={'rank': 8, 'alpha': 16, 'dropout': 0.})
        write_json(output / 'runtime-provenance.json', runtime.provenance)
        initial = adapter_parameter_hashes(runtime.model)
        _require(initial == parent['expected_adapter_parameter_hashes'], 'Loaded tensors differ from the verified public parent')
        write_json(output / 'initialization-hashes.json', initial)
        collator = AICollator(runtime.processor, config.max_input_tokens,
                              expected={r['request_id']: r for r in preflight['rows']})
        parameters = [p for p in runtime.model.parameters() if p.requires_grad]
        _require(len(parameters) == 64 and sum(p.numel() for p in parameters) == 1_703_936,
                 'Trainable parameters differ from the frozen 8B QLoRA profile')
        suffix = runtime.model.get_base_model().model.language_model.layers[-config.suffix_layers:]
        _require(sum(p.numel() for layer in suffix for p in layer.parameters() if p.requires_grad)
                 == runtime.trainable_parameter_count, 'Trainable parameters extend outside the resident suffix')
        checkpoint_resident_suffix(suffix)
        if args.tiled_attention:
            from trackrag.dropoff.training_attention import install_qwen_training_attention
            install_qwen_training_attention(runtime.model.get_base_model().model.language_model.layers)
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=.01)
        runtime.model.eval()  # frozen base and LoRA dropout zero
        with (output / 'training-log.jsonl').open('x', encoding='utf8') as log:
            for epoch in range(1, args.epochs + 1):
                order = list(cases)
                random.Random(41 + epoch).shuffle(order)
                for i in range(0, len(order), 2):
                    batch = order[i:i + 2]
                    optimizer.zero_grad(set_to_none=True)
                    for case in batch:
                        with runtime.execution_lock:
                            loss, tokens = supervised_forward(runtime, collator, case)
                            value = float(loss.detach())
                            (loss / len(batch)).backward()
                        _require(math.isfinite(value), 'Nonfinite continuation loss')
                        del loss
                        consumed.append(case['case_id'])
                        progress['actual_training_examples_consumed'] += 1
                        log.write(json.dumps({'epoch': epoch, 'case_id': case['case_id'], 'loss': value,
                                              'target_tokens': tokens, 'full_target': case['full_target']}) + '\n')
                        log.flush()
                    _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
                             'Missing or nonfinite LoRA gradients')
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    progress['optimizer_steps'] += 1
                    write_json(output / 'progress.json', {**progress, 'last_case': consumed[-1]})
                progress['epochs_completed'] = epoch
                print(f'Completed epoch {epoch}, {progress["optimizer_steps"]} updates', flush=True)
        _require(Counter(consumed) == Counter({k: args.epochs for k in IDS}), 'Incomplete training coverage')
        artifact = save_adapter(runtime, output, initial, progress, sha256((output / 'training_manifest.json').read_bytes()).hexdigest())
        result = {'kind': MODEL_STAGE, 'status': 'trained_fresh_reload_pending', **progress,
                  'train_cases': len(cases), 'consumed_case_ids': consumed, 'elapsed_s': time.perf_counter() - started,
                  'changed_tensors': artifact['changed_adapter_tensors'], 'files': artifact['files'],
                  'human_reviewed': False, 'real_passengers': False, 'development_used': False,
                  'fresh_reload_verified': False, 'quality_verified': False, 'packages': package_versions(),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        write_json(output / 'training_result.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'optimizer_steps', 'changed_tensors', 'elapsed_s')}), flush=True)
    except Exception as exc:
        write_json(output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), **progress,
                                            'elapsed_s': time.perf_counter() - started, 'actual_training_completed': False})
        raise
    finally:
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'train'])
    parser.add_argument('--annotations', type=Path, default=Path('annotations/dropoff/passenger-development'))
    parser.add_argument('--requests', type=Path, default=Path('runs/passenger-references/train-prepared/requests.jsonl'))
    parser.add_argument('--parent-run', type=Path, default=Path('runs/dropoff-handoff-v2-local-20261121'))
    parser.add_argument('--config', type=Path, default=Path('runs/dropoff-handoff-v2-local-20261121/inference_config.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--tiled-attention', action='store_true', help='Bound temporary attention memory while retaining all keys')
    args = parser.parse_args()
    if args.mode == 'validate':
        validate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
