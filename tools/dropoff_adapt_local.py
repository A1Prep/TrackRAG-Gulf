"""Continue the verified streamed 8B adapter on reviewed local interpretation labels.

Validation is CPU-only. Training is a separate, explicit continuation stage with
a fresh optimizer. It never marks reload, local quality or publication complete.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

from tools.dropoff_bundle import REQUIRED_ARTIFACT, _artifact_completion, _json, _regular_bytes
from tools.dropoff_fit_local import supervised_forward, teacher_evaluate
from tools.dropoff_package_local import (PROBE_FILES, SOURCE_FILES as PUBLIC_SOURCE_FILES,
    canonical, digest, hash_map, validate_production_probe)
from tools.dropoff_train import (GroundingCollator, TrainConfig, adapter_parameter_hashes,
    _read_snapshot, load_training_cases, package_versions, prepare_reload_probe, sha256_file, write_json)
from trackrag.dropoff.contracts import FULL_FIELDS, STATUSES
from trackrag.dropoff.data import inference_input_sha256, supervision_sha256
from trackrag.dropoff.formatting import build_messages, target_to_model
from trackrag.dropoff.model import GroundingConfig, verify_adapter
from trackrag.dropoff.streamed import STREAMED_MODEL_ID, STREAMED_REVISION, StreamedRuntime, lora_target


ENCODING = 'reviewed_interpretation_status_first_float_v1'
FIELD_ORDER = ('status', 'destination_point_xy', 'destination_box_xyxy', 'clarification_question')
PROTECTED_RECORDING = 'video_122233'
STAGE = 'reviewed_local_interpretation_continuation'
SOURCE_FILES = PUBLIC_SOURCE_FILES | {'tools/dropoff_adapt_local.py', 'tools/dropoff_bundle.py',
    'tools/dropoff_package_local.py', 'tools/dropoff_compare_local.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parent_context(parent_run, config_path):
    """Verify saved public handoff proof; actual parameter hashes are checked after load."""
    parent = Path(parent_run).resolve()
    names = REQUIRED_ARTIFACT | set(PROBE_FILES.values()) | {
        'training_settings.json', 'selected_adapter_manifest.json',
        'evaluation/trained.adapter-hashes.json', 'evaluation/contrast-trained.adapter-hashes.json'}
    contents = {name: _regular_bytes(parent/name, parent, max_bytes=256*1024**2)[1] for name in names}
    for name in ('probe-image.jpg', 'probe-image.png', 'probe-image.webp'):
        if (parent/name).exists() or (parent/name).is_symlink():
            contents[name] = _regular_bytes(parent/name, parent, max_bytes=32*1024**2)[1]
    complete, reason = _artifact_completion(contents)
    require(complete, 'Parent must be a completed verified public handoff: '+reason)
    completion = _json(contents['completion.json'])
    require(completion.get('status') == 'public_destination_completed', 'Parent is not a completed public stage')
    manifest = _json(contents['adapter/grounding_adapter_manifest.json'])
    selected = _json(contents['selected_adapter_manifest.json'])
    settings = _json(contents['training_settings.json'])
    dataset = _json(contents['dataset_manifest.json'])
    result = _json(contents['training_result.json'])
    selected_sha = digest(contents['selected_adapter_manifest.json'])
    require(manifest.get('raw_selected_manifest_sha256') == selected_sha
        == completion.get('raw_selected_manifest_sha256')
        and completion.get('adapter_manifest_sha256') == digest(contents['adapter/grounding_adapter_manifest.json']),
        'Parent selected/staged manifest proof differs')
    require(selected.get('training_manifest_sha256') == digest(contents['dataset_manifest.json'])
        and selected.get('training_settings_sha256') == digest(contents['training_settings.json'])
        and selected.get('files') == manifest.get('files'), 'Parent original selected data/settings/files differ')
    expected = hash_map(selected.get('adapter_parameter_hashes'), 'parent selected')
    require(result.get('selected_adapter_parameter_hashes') == expected
        and all(_json(contents[name]) == expected for name in (
            'evaluation/trained.adapter-hashes.json', 'evaluation/contrast-trained.adapter-hashes.json')),
        'Parent fresh-reload parameter hashes differ from selected checkpoint')
    production = {name: contents[stored] for name, stored in PROBE_FILES.items()}
    probe = _json(contents['reload_probe.json'])
    image_name = probe['request']['scene']['frame']['image_path']
    proof = validate_production_probe(probe, contents[image_name], production, selected, selected_sha, dataset)
    reload = _json(contents['reload_verification.json'])
    require(reload.get('verified') is True and reload.get('fresh_selected_tensor_hashes_match') is True
        and reload.get('public_production_probe') == proof, 'Parent reload/production evidence differs')
    config_blob = _read_snapshot(config_path, 2*1024**2, 'Parent grounding configuration')
    config = GroundingConfig.from_mapping(_json(config_blob))
    adapter = Path(config.adapter_dir or '')
    adapter = adapter.resolve() if adapter.is_absolute() else (parent/adapter).resolve()
    require(config.adapter_dir and adapter == (parent/'adapter').resolve(), 'Configuration must select the verified parent adapter')
    config = replace(config, adapter_dir=str(adapter))
    require((config.model_id, config.revision) == (STREAMED_MODEL_ID, STREAMED_REVISION)
        and config.runtime == 'streamed' and config.quantization == 'nf4' and config.dtype in ('auto', 'bfloat16')
        and config.device != 'cpu' and config.local_files_only and config.suffix_layers == 8 and bool(config.model_dir),
        'Local continuation requires the same pinned streamed 8B NF4/BF16 profile')
    public_config = settings['model_configuration']
    require(all(getattr(config, key) == public_config[key] for key in (
        'model_id', 'revision', 'min_pixels', 'max_pixels', 'max_input_tokens', 'suffix_layers', 'quantization')),
        'Continuation base, image or input profile differs from its parent')
    verify_adapter(config)
    peft = _json(contents['adapter/adapter_config.json'])
    require(type(peft.get('r')) is int and peft['r'] == 8 and type(peft.get('lora_alpha')) is int
        and peft['lora_alpha'] == 16 and not isinstance(peft.get('lora_dropout'), bool)
        and peft.get('lora_dropout') == 0 and peft.get('target_modules') == lora_target(8)
        and peft.get('bias') == 'none' and not any(peft.get(key) for key in (
            'modules_to_save', 'rank_pattern', 'alpha_pattern', 'use_dora', 'use_rslora')),
        'Parent must use suffix-eight LoRA rank8/alpha16/dropout0 without extra trainable modules')
    train_ids, dev_ids = dataset['train_case_ids'], dataset['development_case_ids']
    require(len(train_ids) == len(set(train_ids)) and len(dev_ids) == len(set(dev_ids))
        and not set(train_ids)&set(dev_ids), 'Parent split case identities are duplicate or overlapping')
    artifacts = dataset['case_artifacts']; indexed = {row['case_id']: row for row in artifacts}
    require(len(indexed) == len(artifacts) and set(indexed) == set(train_ids)|set(dev_ids),
        'Parent case artifact identities do not cover its complete corpus')
    binding = {'schema_version': 1, 'kind': 'verified_public_parent',
        'files': {name: {'sha256': digest(blob), 'bytes': len(blob)} for name, blob in sorted(contents.items())},
        'supplied_config_sha256': digest(config_blob), 'effective_configuration': asdict(config),
        'expected_adapter_parameter_hashes': expected, 'parent_packaged_checkpoint': completion['packaged_checkpoint'],
        'saved_reload_evidence_verified': True, 'actual_initialization_hash_check': 'pending_model_load'}
    return config, dataset, binding


def prepare(args):
    config, parent, binding = parent_context(args.parent_run, args.config)
    train, dev, manifest = load_training_cases(args.train, args.development,
        TrainConfig(run_kind='adaptation', max_train_cases=None, max_dev_cases=None))
    train_recordings = {case['request']['scene']['frame']['recording_id'] for case in train}
    dev_recordings = {case['request']['scene']['frame']['recording_id'] for case in dev}
    require(not train_recordings&dev_recordings,
            'Local continuation train/development recordings must be disjoint')
    parent_images = {row['case_id']: row['image_sha256'] for row in parent['case_artifacts']}
    forbidden = {'train': (set(parent['development_scene_groups']),
                           {parent_images[key] for key in parent['development_case_ids']}),
                 'development': (set(parent['train_scene_groups']),
                           {parent_images[key] for key in parent['train_case_ids']})}
    counts = {}; texts = {}; records = {}
    for partition, cases in (('train', train), ('development', dev)):
        counts[partition] = dict.fromkeys(sorted(STATUSES), 0)
        for case in cases:
            frame = case['request']['scene']['frame']; source = case['source']
            require(case['review'].get('status') == 'human_reviewed' and set(case['supervised_fields']) == FULL_FIELDS,
                    'Continuation accepts genuinely reviewed four-field local interpretations only')
            require(source.get('dataset') != 'Talk2Car-RegSeg',
                    'RegSeg belongs to the separate public stage, not this local continuation route')
            protected = (frame['recording_id'], case['scene_group'].split(':')[-1],
                         source.get('recording_id'), source.get('original_recording_id'))
            require(PROTECTED_RECORDING not in protected
                and PROTECTED_RECORDING not in Path(frame['image_path']).parts
                and not Path(frame['image_path']).stem.startswith(PROTECTED_RECORDING+'_'),
                'Protected final recording must not enter continuation train/development')
            groups, images = forbidden[partition]
            require(case['scene_group'] not in groups and source['image_sha256'] not in images,
                    'Child '+partition+' leaks across its parent training/development boundary')
            counts[partition][case['target']['status']] += 1
            target = target_to_model(case['target'], case['supervised_fields'], frame)
            text = json.dumps({key: target[key] for key in FIELD_ORDER}, separators=(',', ':'), allow_nan=False)
            texts[case['case_id']] = text
            records[case['case_id']] = {'partition': partition, 'image_sha256': source['image_sha256'],
                'original_supervision_sha256': supervision_sha256(case['target'], case['supervised_fields'], frame),
                'input_sha256': inference_input_sha256(case['request'], source['image_sha256']),
                'transformed_target_sha256': digest(text.encode())}
    require(all(counts['train'].values()), 'Reviewed training labels must cover all three true statuses')
    receipt = {'schema_version': 1, 'kind': 'reviewed_interpretation_teacher_encoding', 'encoding': ENCODING,
        'field_order': list(FIELD_ORDER), 'case_count': len(records), 'cases': records,
        'source_bindings': {key: manifest[key] for key in ('train_sha256', 'development_sha256')},
        'mask_snapping': False, 'labels_fabricated': False, 'original_cases_modified': False}
    def builder(case):
        identity = case['case_id']; expected = records[identity]; frame = case['request']['scene']['frame']
        require(case['partition'] == expected['partition'] and case['review'].get('status') == 'human_reviewed'
            and supervision_sha256(case['target'], case['supervised_fields'], frame) == expected['original_supervision_sha256']
            and inference_input_sha256(case['request'], case['source']['image_sha256']) == expected['input_sha256'],
            'Reviewed case changed after continuation preflight')
        return build_messages(case['request'], task='interpret')+[{'role': 'assistant', 'content': texts[identity]}]
    return config, train, dev, manifest, binding, receipt, counts, builder


def validate(args):
    config, train, dev, manifest, binding, receipt, counts, _ = prepare(args)
    return {'schema_version': 1, 'kind': 'local_continuation_cpu_validation', 'valid': True,
        'train_cases': len(train), 'development_cases': len(dev), 'status_counts': counts,
        'development_has_all_statuses': all(counts['development'].values()),
        'parent_binding_sha256': digest(canonical(binding)), 'dataset_manifest_sha256': digest(canonical(manifest)),
        'target_encoding': ENCODING, 'target_encoding_sha256': digest(canonical(receipt)),
        'software_fixture_cases': sum(case['review'].get('software_fixture') is True for case in train+dev),
        'actual_initialization_verified': False, 'gpu_fit_verified': False, 'local_quality_verified': False}


def save_child(runtime, directory, out, progress, initial, *, injected=False):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    # PEFT's wrapping may replace this metadata with the local checkpoint path.
    # Restore the pinned distribution identity before writing the child adapter.
    runtime.model.peft_config['default'].base_model_name_or_path = runtime.config.model_id
    runtime.model.peft_config['default'].revision = runtime.config.revision
    runtime.model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    runtime.processor.save_pretrained(directory)
    hashes = adapter_parameter_hashes(runtime.model)
    require(set(hashes) == set(initial), 'Continuation changed the adapter parameter identities')
    changed = sum(hashes[key] != initial[key] for key in hashes)
    require(changed > 0, 'No parent LoRA parameter changed during continuation')
    manifest = {'schema_version': 1, 'artifact_kind': 'lora_adapter', 'run_kind': STAGE,
        'base_model_id': runtime.config.model_id, 'base_revision': runtime.config.revision,
        'files': {name: sha256_file(directory/name) for name in ('adapter_config.json', 'adapter_model.safetensors')},
        'coordinate_system': 'normalized_0_1000_image_edges', 'suffix_layers': 8,
        'image_processing': {'min_pixels': runtime.config.min_pixels, 'max_pixels': runtime.config.max_pixels},
        'training_stage': STAGE, 'target_encoding': ENCODING, 'counter_scope': 'child_continuation_only',
        'optimizer_steps': progress['optimizer_steps'], 'actual_training_examples_consumed': progress['examples_consumed'],
        'epochs_completed': progress['epochs_completed'], 'adapter_parameter_hashes': hashes,
        'changed_adapter_tensors': changed, 'injected_runtime': injected,
        'verification': {'nonzero_updates_verified': not injected, 'fresh_base_reload_verified': False,
                         'local_quality_verified': False, 'publication_ready': False},
        **{key: sha256_file(out/name) for key, name in (
            ('training_settings_sha256', 'training_settings.json'), ('dataset_manifest_sha256', 'dataset_manifest.json'),
            ('parent_binding_sha256', 'parent-binding.json'), ('target_encoding_sha256', 'target-encoding.json'),
            ('initialization_hashes_sha256', 'initialization-hashes.json'))}}
    write_json(directory/'grounding_adapter_manifest.json', manifest)
    return manifest


def train(args, *, runtime_factory=None):
    out = args.out.resolve(); require(not out.exists(), 'Choose a new continuation output directory')
    require(not out.is_relative_to(Path(args.parent_run).resolve()), 'Child output must be outside its immutable parent handoff')
    config, cases, dev, dataset, parent, encoding, counts, builder = prepare(args)
    injected = runtime_factory is not None; factory = runtime_factory or StreamedRuntime
    require(injected or not any(case['review'].get('software_fixture') is True for case in cases+dev),
            'Known software fixtures cannot be used for real continuation training')
    out.mkdir(parents=True, exist_ok=False)
    runtime = None; started = time.monotonic()
    progress = {'optimizer_steps': 0, 'examples_consumed': 0, 'epochs_completed': 0, 'actual_training_case_ids': []}
    try:
        write_json(out/'parent-binding.json', parent); write_json(out/'dataset_manifest.json', dataset)
        write_json(out/'target-encoding.json', encoding)
        source_root = Path(__file__).resolve().parents[1]; sources = {}
        for name in sorted(SOURCE_FILES):
            _, content = _regular_bytes(source_root/name, source_root, max_bytes=8*1024**2)
            destination = out/'source'/name; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content); sources[name] = digest(content)
        for label, source in (('train', args.train), ('development', args.development)):
            payload = _read_snapshot(source, 64*1024**2, 'Child split snapshot')
            require(digest(payload) == dataset[label+'_sha256'], 'Child split changed after validation')
            (out/(label+'-snapshot.jsonl')).write_bytes(payload)
        settings = {'schema_version': 1, 'stage': STAGE, 'model_configuration': asdict(config),
            'epochs': args.epochs, 'learning_rate': args.learning_rate, 'gradient_accumulation': args.accumulation,
            'seed': args.seed, 'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0.0, 'status_counts': counts,
            'optimizer_initialization': 'fresh_AdamW_continuing_parent_LoRA_weights_not_optimizer_resume',
            'target_encoding': ENCODING, 'source_sha256': sources, 'injected_runtime': injected,
            'parent_binding_sha256': sha256_file(out/'parent-binding.json'),
            'target_encoding_sha256': sha256_file(out/'target-encoding.json')}
        write_json(out/'training_settings.json', settings)
        probe_case = next(case for case in sorted(cases, key=lambda row: row['case_id']) if case['target']['status'] == 'proposed')
        probe_dir = out/'reload-input'; probe_dir.mkdir()
        probe = prepare_reload_probe(probe_case, probe_dir)
        (probe_dir/'requests.jsonl').write_text(json.dumps(probe['request'])+'\n', encoding='utf-8')
        import torch
        if not injected:
            torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.90, 0)
        torch.manual_seed(args.seed)
        runtime = factory(config, training_lora={'rank': 8, 'alpha': 16, 'dropout': 0.0})
        initial = adapter_parameter_hashes(runtime.model)
        require(initial == parent['expected_adapter_parameter_hashes'], 'Loaded parent adapter parameters do not match verified initialization')
        write_json(out/'initialization-hashes.json', {'actual_parameter_hashes': initial,
            'matches_verified_parent': True, 'injected_runtime': injected})
        write_json(out/'runtime-provenance.json', runtime.provenance)
        collator = GroundingCollator(runtime.processor, config.max_input_tokens, message_builder=builder)
        baseline = teacher_evaluate(runtime, collator, dev, target_encoding=ENCODING)
        write_json(out/'parent-development-nll.json', baseline)
        parameters = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
        require(bool(parameters), 'Continuation has no trainable parent LoRA parameters')
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
        total = math.ceil(len(cases)/args.accumulation)*args.epochs; warmup = max(1, round(total*.05))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, (step+1)/warmup)*.5*(1+math.cos(
            math.pi*min(1, max(0, step-warmup)/max(1, total-warmup)))))
        best = None
        with (out/'training-log.jsonl').open('w', encoding='utf-8') as log:
            for epoch in range(1, args.epochs+1):
                order = list(range(len(cases))); random.Random(args.seed+epoch-1).shuffle(order)
                runtime.model.eval()
                for start in range(0, len(order), args.accumulation):
                    group = order[start:start+args.accumulation]; optimizer.zero_grad(set_to_none=True)
                    for index in group:
                        case = cases[index]
                        with runtime.execution_lock:
                            loss, tokens = supervised_forward(runtime, collator, case)
                            value = float(loss.detach()); (loss/len(group)).backward()
                        require(math.isfinite(value), 'Continuation loss is nonfinite')
                        progress['examples_consumed'] += 1; progress['actual_training_case_ids'].append(case['case_id'])
                        log.write(json.dumps({'epoch': epoch, 'case_id': case['case_id'], 'loss': value, 'target_tokens': tokens})+'\n'); log.flush()
                        del loss
                    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
                    require(len(gradients) == len(parameters) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
                            'Continuation has missing or nonfinite LoRA gradients')
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True); optimizer.step(); scheduler.step()
                    progress['optimizer_steps'] += 1
                    write_json(out/'progress.json', {**progress, 'status': 'training', 'injected_runtime': injected})
                progress['epochs_completed'] = epoch; optimizer.zero_grad(set_to_none=True)
                metrics = teacher_evaluate(runtime, collator, dev, target_encoding=ENCODING)
                require(math.isfinite(metrics['token_weighted_nll']), 'Development NLL is nonfinite')
                write_json(out/f'epoch-{epoch}-development-nll.json', metrics)
                save_child(runtime, out/f'epoch-{epoch}-adapter', out, progress, initial, injected=injected)
                if best is None or metrics['token_weighted_nll'] < best['token_weighted_nll']:
                    best = {'epoch': epoch, 'token_weighted_nll': metrics['token_weighted_nll'],
                            'optimizer_steps': progress['optimizer_steps'], 'presentations': progress['examples_consumed']}
                    selected = save_child(runtime, out/'adapter', out, progress, initial, injected=injected)
        require(progress['optimizer_steps'] == total and Counter(progress['actual_training_case_ids']) ==
            Counter({case['case_id']: args.epochs for case in cases}), 'Continuation schedule did not consume every planned presentation')
        result = {'schema_version': 1, 'run_kind': STAGE, 'completed_at': datetime.now(timezone.utc).isoformat(),
            'status': 'software_fixture_completed_not_real_training' if injected else 'local_interpretation_training_completed_reload_pending',
            **progress, 'counter_scope': 'child_continuation_only', 'training_examples_unique': len(cases),
            'parent_development_nll': baseline['token_weighted_nll'], 'selected_checkpoint': best,
            'selected_adapter_parameter_hashes': selected['adapter_parameter_hashes'], 'selected_adapter_files': selected['files'],
            'status_counts': counts, 'target_encoding': ENCODING, 'injected_runtime': injected,
            'elapsed_s': time.monotonic()-started, 'packages': package_versions(),
            'actual_training_completed': not injected, 'fresh_reload_verified': False,
            'local_quality_verified': False, 'publication_ready': False, 'task5_complete': False}
        write_json(out/'training_result.json', result)
        write_json(out/'inference_config.json', asdict(replace(config, adapter_dir=str(out/'adapter'))))
        write_json(out/'parent_inference_config.json', asdict(config))
        return result
    except Exception as error:
        write_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(), 'progress': progress, 'injected_runtime': injected,
            'training_completed': False, 'fresh_reload_verified': False, 'local_quality_verified': False})
        raise
    finally:
        if runtime is not None:
            original_error = sys.exc_info()[1]
            try:
                runtime.close()
            except Exception as error:
                write_json(out/'cleanup-error.json', {'type': type(error).__name__, 'message': str(error)})
                if original_error is None:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'train'):
        command = commands.add_parser(name)
        command.add_argument('--parent-run', type=Path, required=True, help='Completed verified staged public handoff')
        command.add_argument('--config', type=Path, required=True, help='Streamed config selecting that parent adapter')
        command.add_argument('--train', type=Path, required=True); command.add_argument('--development', type=Path, required=True)
        if name == 'train':
            command.add_argument('--out', type=Path, required=True); command.add_argument('--epochs', type=int, default=2)
            command.add_argument('--learning-rate', type=float, default=1e-4); command.add_argument('--accumulation', type=int, default=2)
            command.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == 'train':
        if not 1 <= args.epochs <= 5 or not 1 <= args.accumulation <= 8 or not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .001 or args.seed < 0:
            parser.error('Invalid continuation training settings')
        result = train(args)
    else:
        result = validate(args)
    print(json.dumps(result, indent=2, allow_nan=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
