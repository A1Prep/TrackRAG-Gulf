"""Synthetic review/proof and optimizer fixtures; no real reviews, model or GPU.

These temporary examples and mocked losses test software boundaries and schedule
accounting. They are never a project training corpus or an accuracy measurement.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

from PIL import Image
import pytest

from tools import dropoff_adapt_local as adapt
from tools.dropoff_package_local import canonical, digest, validate_production_probe
from trackrag.dropoff.contracts import FULL_FIELDS
from trackrag.dropoff.data import inference_input_sha256, supervision_sha256
from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION, GroundingConfig
from trackrag.dropoff.model import verify_adapter
from trackrag.dropoff.streamed import METADATA_SHA256, WEIGHT_SHA256, lora_target


def put(root, name, value):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n', encoding='utf-8')
    return path


def parameter_hashes(value):
    return {name: digest(f'software-fixture:{name}:{value}'.encode()) for name in ('lora-a', 'lora-b')}


def review(case):
    """Self-consistent fictional review metadata for contract tests only."""
    case['review'] = {'status': 'human_reviewed', 'reviewers': ['fixture-reviewer-a', 'fixture-reviewer-b'],
        'reviewed_at': '2026-11-21T12:00:00+04:00', 'software_fixture': True,
        'image_sha256': case['source']['image_sha256'],
        'supervision_sha256': supervision_sha256(case['target'], case['supervised_fields'], case['request']['scene']['frame']),
        'input_sha256': inference_input_sha256(case['request'], case['source']['image_sha256'])}
    return case


def write_cases(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


@pytest.fixture
def local_files(tmp_path):
    parent = tmp_path/'verified-parent'; parent.mkdir()
    Image.new('RGB', (64, 32), (1, 2, 3)).save(parent/'probe-image.png')
    parent_dev = tmp_path/'parent-dev.png'
    Image.new('RGB', (64, 32), (3, 2, 1)).save(parent_dev)
    image_sha = adapt.sha256_file(parent/'probe-image.png')
    config = GroundingConfig(runtime='streamed', model_id=adapt.STREAMED_MODEL_ID,
        revision=adapt.STREAMED_REVISION, adapter_dir=str(parent/'adapter'), model_dir='unused-base',
        cache_dir='unused-processor-cache', quantization='nf4', dtype='bfloat16', device='cuda:0',
        min_pixels=65536, max_pixels=262144, max_time_s=300)
    config_path = put(parent, 'inference_config.json', asdict(config))
    settings = {'model_configuration': asdict(config), 'epochs': 1, 'gradient_accumulation': 1}
    put(parent, 'training_settings.json', settings)
    dataset = {'train_case_ids': ['parent-train'], 'development_case_ids': ['parent-dev'],
        'train_sha256': 'a'*64, 'development_sha256': 'd'*64,
        'train_scene_groups': ['parent-training-scene'], 'development_scene_groups': ['parent-development-scene'],
        'case_artifacts': [{'case_id': 'parent-train', 'image_sha256': image_sha},
            {'case_id': 'parent-dev', 'image_sha256': adapt.sha256_file(parent_dev)}]}
    put(parent, 'dataset_manifest.json', dataset)
    put(parent, 'training_config.json', {'schema_version': 1, 'run_kind': 'public_destination',
        'base_model_id': config.model_id, 'base_revision': config.revision})
    put(parent, 'adapter/adapter_config.json', {'peft_type': 'LORA', 'base_model_name_or_path': config.model_id,
        'revision': config.revision, 'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0,
        'target_modules': lora_target(8), 'bias': 'none'})
    (parent/'adapter/adapter_model.safetensors').write_bytes(b'SOFTWARE_FIXTURE_NOT_MODEL_WEIGHTS')
    files = {name: adapt.sha256_file(parent/'adapter'/name) for name in ('adapter_config.json', 'adapter_model.safetensors')}
    selected = {'schema_version': 1, 'artifact_kind': 'lora_adapter', 'base_model_id': config.model_id,
        'base_revision': config.revision, 'suffix_layers': 8,
        'image_processing': {'min_pixels': 65536, 'max_pixels': 262144},
        'coordinate_system': 'normalized_0_1000_image_edges', 'files': files,
        'training_manifest_sha256': adapt.sha256_file(parent/'dataset_manifest.json'),
        'training_settings_sha256': adapt.sha256_file(parent/'training_settings.json'),
        'adapter_parameter_hashes': parameter_hashes(0)}
    selected_path = put(parent, 'selected_adapter_manifest.json', selected)
    for name in ('trained', 'contrast-trained'):
        put(parent, 'evaluation/'+name+'.adapter-hashes.json', parameter_hashes(0))
    result = {'schema_version': 1, 'run_kind': 'public_destination', 'nonzero_updates_verified': True,
        'optimizer_steps': 1, 'changed_adapter_tensors': 2, 'selected_adapter_parameter_hashes': parameter_hashes(0)}
    put(parent, 'training_result.json', result)
    flags = dict.fromkeys(('nonzero_updates_verified', 'fresh_base_reload_verified', 'generation_smoke_completed',
                          'structured_generation_valid', 'spatial_proposal_generated'), True)
    staged = {**selected, 'run_kind': 'public_destination', 'verification': flags,
        'raw_selected_manifest_sha256': adapt.sha256_file(selected_path),
        **{key: adapt.sha256_file(parent/name) for key, name in (
            ('training_config_sha256', 'training_config.json'), ('dataset_manifest_sha256', 'dataset_manifest.json'),
            ('training_result_sha256', 'training_result.json'))}}
    staged_path = put(parent, 'adapter/grounding_adapter_manifest.json', staged)
    request = {'request_id': 'parent-train', 'text': 'Stop beside the entrance', 'input_mode': 'typed', 'dialogue': [],
        'scene': {'source': 'predicted', 'tracks': [], 'provenance': {'mode': 'image_only_public_training'},
            'frame': {'recording_id': 'parent-train', 'frame_id': 0, 'timestamp_s': 0,
                      'width': 64, 'height': 32, 'image_path': 'probe-image.png'}}}
    probe = {'schema_version': 1, 'case_id': 'parent-train', 'partition': 'train', 'request': request, 'image_sha256': image_sha}
    put(parent, 'reload_probe.json', probe)
    actual = deepcopy(request); actual['scene']['frame']['image_path'] = str(parent/'probe-image.png')
    raw_text = json.dumps({'status': 'proposed', 'destination_point_xy': [500, 500],
        'destination_box_xyxy': [400, 400, 600, 600], 'clarification_question': None})
    from trackrag.dropoff.formatting import parse_model_output
    parsed = parse_model_output(raw_text, 64, 32)
    provenance = {'backend_kind': 'transformers_local_streamed', 'injected': False, 'requested_condition': 'fine_tuned',
        'model_id': config.model_id, 'revision': config.revision, 'request_sha256': digest(canonical(actual)),
        'image': {'sha256': image_sha}, 'adapter': {'manifest_sha256': adapt.sha256_file(selected_path),
            'base_model_id': config.model_id, 'base_revision': config.revision,
            'files': {name: {'sha256': value} for name, value in files.items()}},
        'backend_details': {'native_fault_isolation': 'spawn_process', 'worker_pid': 123, 'runtime': 'streamed',
            'dtype': 'bfloat16', 'device': 'cuda:0', 'model_id': config.model_id, 'revision': config.revision,
            'processor_model_id': BASE_MODEL_ID, 'processor_revision': BASE_REVISION,
            'suffix_layers': 8, 'effective_resident_layers': 20, 'image_processing': selected['image_processing'],
            'weight_identity': {'model_id': config.model_id, 'revision': config.revision,
                'verified_weights': True, 'verified_metadata': True,
                'files': {name: {'sha256': value} for name, value in {**WEIGHT_SHA256, **METADATA_SHA256}.items()}}}}
    receipt = {'request_id': actual['request_id'], 'request': actual, 'status': 'completed',
        'raw_text': raw_text, 'result': parsed, 'provenance': provenance}
    production = {}
    for name, value in (('receipts.jsonl', receipt), ('model-outcomes.jsonl', {'ok': True,
        'request_id': actual['request_id'], 'result': parsed, 'provenance': provenance}),
        ('resolved-requests.jsonl', actual), ('input-snapshot.jsonl', actual)):
        production[name] = (json.dumps(value)+'\n').encode()
    summary = {'configuration': asdict(config), 'requested_rows': 1, 'semantic_results': 1, 'operational_errors': 0,
        **{key: digest(production[name]) for key, name in (('requests_sha256', 'input-snapshot.jsonl'),
            ('resolved_requests_sha256', 'resolved-requests.jsonl'), ('outcomes_sha256', 'model-outcomes.jsonl'),
            ('receipts_sha256', 'receipts.jsonl'))}}
    production['run.json'] = (json.dumps(summary)+'\n').encode()
    for name, stored in adapt.PROBE_FILES.items(): (parent/stored).write_bytes(production[name])
    proof = validate_production_probe(probe, (parent/'probe-image.png').read_bytes(), production,
                                      selected, adapt.sha256_file(selected_path), dataset)
    put(parent, 'reload_verification.json', {'verified': True, 'fresh_selected_tensor_hashes_match': True,
                                           'public_production_probe': proof})
    put(parent, 'completion.json', {'schema_version': 1, 'status': 'public_destination_completed',
        'packaged_checkpoint': {'epoch': 1, 'optimizer_steps': 1, 'presentations': 1},
        'raw_selected_manifest_sha256': adapt.sha256_file(selected_path),
        'adapter_manifest_sha256': adapt.sha256_file(staged_path)})
    rows = []
    for index, (partition, status) in enumerate((('train', 'proposed'), ('train', 'needs_clarification'),
        ('train', 'insufficient_visual_evidence'), ('development', 'proposed'))):
        identity = 'local-'+str(index); image = tmp_path/(identity+'.png')
        Image.new('RGB', (64, 32), (index+10, 20, 30)).save(image)
        proposed = status == 'proposed'
        rows.append(review({'schema_version': 1, 'case_id': identity, 'scene_group': 'child-scene-'+str(index),
            'partition': partition, 'request': {'request_id': identity, 'text': 'Stop next to the visible entrance',
                'input_mode': 'typed', 'dialogue': [], 'scene': {'source': 'predicted', 'tracks': [],
                    'frame': {'recording_id': identity, 'frame_id': 0, 'timestamp_s': 0,
                              'width': 64, 'height': 32, 'image_path': str(image)}}},
            'source': {'dataset': 'SyntheticLocalSoftwareFixture', 'source_split': partition,
                       'image_sha256': adapt.sha256_file(image)}, 'supervised_fields': list(FULL_FIELDS),
            'target': {'status': status, 'destination_point_xy': [32, 16] if proposed else None,
                'destination_box_xyxy': [16, 8, 48, 24] if proposed else None,
                'clarification_question': 'Which visible entrance do you mean?' if status == 'needs_clarification' else None}}))
    train_path, dev_path = tmp_path/'train.jsonl', tmp_path/'development.jsonl'
    write_cases(train_path, rows[:3]); write_cases(dev_path, rows[3:])
    args = SimpleNamespace(parent_run=parent, config=config_path, train=train_path, development=dev_path,
        out=tmp_path/'new-child-run', epochs=2, learning_rate=.0001, accumulation=2, seed=42)
    return args, rows, parent_dev


def test_validate_is_cpu_only_and_reports_true_status_coverage(local_files):
    args, rows, _ = local_files
    before = {str(path): path.read_bytes() for path in args.parent_run.parent.rglob('*') if path.is_file()}
    report = adapt.validate(args)
    assert report['train_cases'] == 3 and report['development_cases'] == 1
    assert set(report['status_counts']['train'].values()) == {1}
    assert report['status_counts']['development']['proposed'] == 1
    assert report['development_has_all_statuses'] is False
    assert report['software_fixture_cases'] == 4
    assert report['actual_initialization_verified'] is False and report['local_quality_verified'] is False
    assert all(path.read_bytes() == before[str(path)] for path in args.parent_run.parent.rglob('*') if path.is_file())
    assert not args.out.exists()


def test_full_teacher_codec_is_status_first_without_changing_original_cases(local_files):
    args, _, _ = local_files
    _, train, dev, _, _, encoding, _, builder = adapt.prepare(args)
    original = deepcopy(train+dev)
    for case in train+dev:
        target = json.loads(builder(case)[-1]['content'])
        assert list(target) == list(adapt.FIELD_ORDER)
        assert target['status'] == case['target']['status']
        assert set(target) == FULL_FIELDS
        if target['status'] != 'proposed': assert target['destination_point_xy'] is None
    assert train+dev == original and encoding['mask_snapping'] is False


def test_local_route_rejects_even_human_reviewed_regseg_cases(local_files, monkeypatch):
    args, _, _ = local_files
    real_loader = adapt.load_training_cases
    def loaded(*call_args, **kwargs):
        train, dev, manifest = real_loader(*call_args, **kwargs)
        train[0]['source']['dataset'] = 'Talk2Car-RegSeg'
        return train, dev, manifest
    # No real RegSeg labels are created; this isolates the post-loader scope guard.
    monkeypatch.setattr(adapt, 'load_training_cases', loaded)
    with pytest.raises(ValueError, match='RegSeg belongs to the separate public stage'):
        adapt.validate(args)


@pytest.mark.parametrize('bad', ['public', 'partial', 'anchors', 'missing_status'])
def test_rejects_partial_public_or_incomplete_status_training(local_files, bad):
    args, rows, _ = local_files
    row = rows[0]
    if bad == 'public': row['review']['status'] = 'public_annotation'
    elif bad == 'partial':
        row['target'] = {key: row['target'][key] for key in ('destination_point_xy', 'destination_box_xyxy')}
        row['supervised_fields'] = list(row['target']); review(row)
    elif bad == 'anchors':
        row['target']['anchor_boxes_xyxy'] = []; row['supervised_fields'].append('anchor_boxes_xyxy'); review(row)
    else:
        rows[1]['target'] = deepcopy(rows[0]['target']); review(rows[1])
    write_cases(args.train, rows[:3])
    with pytest.raises(ValueError): adapt.validate(args)


@pytest.mark.parametrize('partition', ['train', 'development'])
def test_rejects_protected_final_recording(local_files, partition):
    args, rows, _ = local_files
    row = rows[0] if partition == 'train' else rows[3]
    row['request']['scene']['frame']['recording_id'] = adapt.PROTECTED_RECORDING; review(row)
    write_cases(args.train if partition == 'train' else args.development, rows[:3] if partition == 'train' else rows[3:])
    with pytest.raises(ValueError, match='Protected final recording'): adapt.validate(args)


@pytest.mark.parametrize('leak', ['train_scene', 'dev_scene', 'train_image', 'dev_image', 'child_overlap'])
def test_rejects_child_or_cross_parent_leakage(local_files, leak):
    args, rows, parent_dev = local_files
    if leak == 'train_scene': rows[0]['scene_group'] = 'parent-development-scene'
    elif leak == 'dev_scene': rows[3]['scene_group'] = 'parent-training-scene'
    elif leak == 'child_overlap': rows[3]['scene_group'] = rows[0]['scene_group']
    else:
        row, source = (rows[0], parent_dev) if leak == 'train_image' else (rows[3], args.parent_run/'probe-image.png')
        Path(row['request']['scene']['frame']['image_path']).write_bytes(source.read_bytes())
        row['source']['image_sha256'] = adapt.sha256_file(source); review(row)
    write_cases(args.train, rows[:3]); write_cases(args.development, rows[3:])
    with pytest.raises(ValueError, match='leak|overlap|identical image'): adapt.validate(args)


def test_distinct_images_and_groups_from_one_recording_cannot_cross_splits(local_files):
    args, rows, _ = local_files
    rows[3]['request']['scene']['frame'].update(recording_id=rows[0]['request']['scene']['frame']['recording_id'],
                                              frame_id=1, timestamp_s=.1)
    review(rows[3]); write_cases(args.development, rows[3:])
    assert rows[0]['scene_group'] != rows[3]['scene_group']
    assert rows[0]['source']['image_sha256'] != rows[3]['source']['image_sha256']
    with pytest.raises(ValueError, match='recordings must be disjoint'):
        adapt.validate(args)


def test_known_fixture_markers_stop_real_training_before_model_or_output(local_files, monkeypatch):
    args, _, _ = local_files
    def forbidden(*args, **kwargs): pytest.fail('A real model must not be loaded for known software fixtures')
    monkeypatch.setattr(adapt, 'StreamedRuntime', forbidden)
    with pytest.raises(ValueError, match='Known software fixtures'):
        adapt.train(args)
    assert not args.out.exists()


@pytest.mark.parametrize('mismatch', ['adapter_bytes', 'fresh_hashes', 'profile', 'missing_completion'])
def test_parent_must_be_verified_and_match_config(local_files, mismatch):
    args, _, _ = local_files
    if mismatch == 'adapter_bytes': (args.parent_run/'adapter/adapter_model.safetensors').write_bytes(b'changed')
    elif mismatch == 'fresh_hashes': put(args.parent_run, 'evaluation/trained.adapter-hashes.json', parameter_hashes(99))
    elif mismatch == 'missing_completion': (args.parent_run/'completion.json').unlink()
    else:
        config = json.loads(args.config.read_text()); config['max_pixels'] = 131072; put(args.parent_run, args.config.name, config)
    with pytest.raises((ValueError, OSError)): adapt.validate(args)


class FixtureLoss:
    def __init__(self, runtime, scale=1): self.runtime, self.scale = runtime, scale
    def detach(self): return .5
    def __truediv__(self, divisor): return FixtureLoss(self.runtime, self.scale/divisor)
    def backward(self):
        if self.runtime.fail_gradients == 'missing':
            return
        for parameter in self.runtime.model.parameters():
            parameter.grad = float('nan') if self.runtime.fail_gradients == 'nonfinite' else (parameter.grad or 0)+self.scale


def software_training(monkeypatch, *, nlls=(1.0, .4, .5), bad_initialization=False, fail_gradients=False):
    runtime_list, optimizer_steps = [], []
    class Model:
        def __init__(self):
            self.value = 99 if bad_initialization else 0
            self.params = [SimpleNamespace(requires_grad=True, grad=None) for _ in range(2)]
            self.peft_config = {'default': SimpleNamespace(base_model_name_or_path='owner-local-checkpoint-path',
                                                          revision='overwritten-local-revision')}
        def parameters(self): return self.params
        def eval(self): pass
        def save_pretrained(self, path, **kwargs):
            peft = self.peft_config['default']
            put(path, 'adapter_config.json', {'peft_type': 'LORA', 'fixture': True,
                'base_model_name_or_path': peft.base_model_name_or_path, 'revision': peft.revision})
            (path/'adapter_model.safetensors').write_bytes(('fixture:'+str(self.value)).encode())
    class Runtime:
        def __init__(self, config, *, training_lora):
            assert training_lora == {'rank': 8, 'alpha': 16, 'dropout': 0.0}
            self.config, self.model = config, Model(); self.execution_lock = threading.RLock()
            self.processor = SimpleNamespace(save_pretrained=lambda path: None)
            self.provenance = {'runtime': 'explicit_software_fixture_not_8b_execution'}; self.closed = False
            self.fail_gradients = fail_gradients
            runtime_list.append(self)
        def close(self): self.closed = True
    class Optimizer:
        def __init__(self, parameters, **kwargs): self.parameters = parameters
        def zero_grad(self, **kwargs):
            for parameter in self.parameters: parameter.grad = None
        def step(self):
            optimizer_steps.append([parameter.grad for parameter in self.parameters])
            runtime_list[0].model.value += 1
    fake_torch = SimpleNamespace(manual_seed=lambda seed: None,
        optim=SimpleNamespace(AdamW=Optimizer, lr_scheduler=SimpleNamespace(
            LambdaLR=lambda *a: SimpleNamespace(step=lambda: None))),
        nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a, **kw: 1.0)),
        isfinite=lambda value: SimpleNamespace(all=lambda: math.isfinite(value)))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setattr(adapt, 'adapter_parameter_hashes', lambda model: parameter_hashes(model.value))
    scores = iter(nlls)
    def teacher(runtime, collator, cases, *, target_encoding):
        assert target_encoding == adapt.ENCODING
        assert json.loads(collator.message_builder(cases[0])[-1]['content'])['status'] == cases[0]['target']['status']
        return {'kind': 'software_fixture_nll', 'token_weighted_nll': next(scores), 'target_encoding': target_encoding}
    def forward(runtime, collator, case):
        assert list(json.loads(collator.message_builder(case)[-1]['content'])) == list(adapt.FIELD_ORDER)
        return FixtureLoss(runtime), 12
    monkeypatch.setattr(adapt, 'teacher_evaluate', teacher)
    monkeypatch.setattr(adapt, 'supervised_forward', forward)
    return Runtime, runtime_list, optimizer_steps


def test_software_schedule_selects_best_epoch_and_never_claims_reload_or_quality(local_files, monkeypatch):
    args, _, _ = local_files; factory, runtimes, steps = software_training(monkeypatch)
    result = adapt.train(args, runtime_factory=factory)
    assert result['status'] == 'software_fixture_completed_not_real_training'
    assert result['optimizer_steps'] == 4 and result['examples_consumed'] == 6
    assert result['epochs_completed'] == 2 and result['selected_checkpoint']['epoch'] == 1
    assert result['selected_checkpoint']['optimizer_steps'] == 2 and result['selected_checkpoint']['presentations'] == 3
    assert all(gradients == [1, 1] for gradients in steps)  # Last one-case groups divide by one.
    assert result['selected_adapter_parameter_hashes'] == parameter_hashes(2)
    assert runtimes[0].model.value == 4 and runtimes[0].closed
    assert all(result[key] is False for key in ('actual_training_completed', 'fresh_reload_verified',
        'local_quality_verified', 'publication_ready', 'task5_complete'))
    assert not (args.out/'completion.json').exists()
    initialized = json.loads((args.out/'initialization-hashes.json').read_text())
    assert initialized['actual_parameter_hashes'] == parameter_hashes(0) and initialized['injected_runtime'] is True
    selected = json.loads((args.out/'adapter/grounding_adapter_manifest.json').read_text())
    assert selected['training_stage'] == adapt.STAGE and selected['verification']['nonzero_updates_verified'] is False
    assert selected['parent_binding_sha256'] == adapt.sha256_file(args.out/'parent-binding.json')
    assert selected['target_encoding_sha256'] == adapt.sha256_file(args.out/'target-encoding.json')
    peft = json.loads((args.out/'adapter/adapter_config.json').read_text())
    assert peft['base_model_name_or_path'] == adapt.STREAMED_MODEL_ID and peft['revision'] == adapt.STREAMED_REVISION
    assert 'owner-local-checkpoint-path' not in (args.out/'adapter/adapter_config.json').read_text()
    child_config = GroundingConfig.from_mapping(json.loads((args.out/'inference_config.json').read_text()))
    assert verify_adapter(child_config)['base_model_id'] == adapt.STREAMED_MODEL_ID


def test_actual_parent_hash_mismatch_stops_before_updates_and_keeps_failure(local_files, monkeypatch):
    args, _, _ = local_files; factory, runtimes, steps = software_training(monkeypatch, bad_initialization=True)
    with pytest.raises(ValueError, match='Loaded parent adapter parameters'): adapt.train(args, runtime_factory=factory)
    failure = json.loads((args.out/'failure.json').read_text())
    assert failure['progress']['optimizer_steps'] == 0 and steps == [] and runtimes[0].closed
    assert failure['training_completed'] is False and not (args.out/'training_result.json').exists()


@pytest.mark.parametrize('gradients', ['missing', 'nonfinite'])
def test_invalid_gradients_never_reach_optimizer_or_completion(local_files, monkeypatch, gradients):
    args, _, _ = local_files; factory, runtimes, steps = software_training(monkeypatch, fail_gradients=gradients)
    with pytest.raises(ValueError, match='missing or nonfinite LoRA gradients'):
        adapt.train(args, runtime_factory=factory)
    failure = json.loads((args.out/'failure.json').read_text())
    assert steps == [] and failure['progress']['optimizer_steps'] == 0
    assert runtimes[0].closed and not (args.out/'completion.json').exists()


def test_import_and_help_do_not_import_torch():
    code = "import sys; import tools.dropoff_adapt_local; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-m', 'tools.dropoff_adapt_local', '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and 'validate' in result.stdout and 'train' in result.stdout
