"""Software-only recovery and source checks. No private cases, weights or accelerator required."""
from dataclasses import replace
from hashlib import sha256
import json

from PIL import Image
import pytest

from tools import passenger_reviewed_runs as study
from trackrag.dropoff.model import GroundingConfig, GroundingModel


def request(image, index):
    return {'request_id': f'software-fixture-{index}', 'text': 'Stop by the software test marker.',
            'input_mode': 'typed', 'dialogue': [], 'scene': {'source': 'predicted', 'tracks': [], 'provenance': {},
                'frame': {'image_path': str(image), 'width': 80, 'height': 60, 'recording_id': 'software-fixture',
                          'frame_id': index, 'timestamp_s': 0}}}


def write_rows(path, values):
    path.write_text(''.join(json.dumps(row) + '\n' for row in values), encoding='utf-8')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    work = tmp_path / 'test workspace'
    monkeypatch.setattr(study, 'WORK', work)
    def forbidden(*args, **kwargs):
        pytest.fail('A software-only source-integrity check attempted to load a GPU worker')
    monkeypatch.setattr('trackrag.dropoff.streamed.StreamedProcessBackend', forbidden)
    before = GroundingConfig.from_mapping(study.read(study.ROOT / 'configs/dropoff/passenger-final-pretrained.json'))
    configs = {'pretrained': before, 'trained': replace(before, adapter_dir='software-fixture/unavailable')}
    registry = {'pretrained': {'config_sha256': 'f'*64}, 'trained': {'config_sha256': 'e'*64,
                'adapter_manifest_sha256': 'd'*64, 'adapter_weights_sha256': 'c'*64}}
    return work, configs, registry


def begin(work, role, number):
    folder = work / 'attempts' / f'{role}-{number:02}'
    launch = {'role': role, 'attempt': folder.name, 'started_utc': study.now(), 'eligibility_sha256': study.ELIGIBILITY_SHA}
    study.save(folder / 'launch.json', launch)
    return folder, launch


@pytest.fixture
def recovered(workspace, monkeypatch):
    def make(role='pretrained'):
        work, configs, registry = workspace
        requests = [request(work / f'deliberately-missing-{i}.jpg', i) for i in range(1, 4)]
        monkeypatch.setattr(study, 'image_hashes', lambda: {q['request_id']: '0'*64 for q in requests})
        original = GroundingModel.interpret_with_receipt
        calls = []
        def observe(model, q):
            calls.append(q['request_id'])
            return original(model, q)
        monkeypatch.setattr(GroundingModel, 'interpret_with_receipt', observe)
        first, _ = begin(work, role, 1)
        native = study.ObservedNative(role, first, requests, configs, registry, 'software-fixture')
        outcome, receipt = native.interpret_with_receipt(requests[0])
        native.close()
        assert outcome['ok'] is False and receipt['error']['code'] == 'invalid_image'
        initial_hash = study.file_sha256(study.row_path(role, 1))
        second, _ = begin(work, role, 2)
        resumed = study.ObservedNative(role, second, requests, configs, registry, 'software-fixture')
        assert resumed.interpret_with_receipt(requests[0]) == (outcome, receipt)
        for q in requests[1:]:
            resumed.interpret_with_receipt(q)
        resumed.close()
        assert calls == [q['request_id'] for q in requests]
        assert study.file_sha256(study.row_path(role, 1)) == initial_hash
        assert not (second / 'started-calls/001.json').exists()
        return requests
    return make


@pytest.mark.parametrize('role', ('pretrained', 'trained'))
def test_resume_preserves_genuine_errors_with_no_duplicate_model_call(recovered, role):
    recovered(role)
    assert study.journal_prefix(role) == 3


def test_damaged_call_rejected_before_reuse(recovered):
    recovered()
    path = study.row_path('pretrained', 1)
    envelope = study.read(path)
    envelope['record']['receipt']['request_id'] = 'software-fixture-tamper'
    study.save(path, envelope, replace=True)
    with pytest.raises(ValueError, match='content has changed'):
        study.load_journal(path)


def test_missing_middle_call_aborts_without_replacing_following_calls(recovered):
    recovered()
    third = study.row_path('pretrained', 3)
    original = study.file_sha256(third)
    study.row_path('pretrained', 2).unlink()
    with pytest.raises(ValueError, match='missing or unexpected'):
        study.journal_prefix('pretrained')
    assert study.file_sha256(third) == original


@pytest.mark.parametrize('wrong', ('input', 'condition', 'injected', 'launch-time'))
def test_incompatible_journal_or_provenance_is_rejected(recovered, workspace, wrong):
    requests = recovered()
    _, configs, registry = workspace
    row = study.load_journal(study.row_path('pretrained', 1))
    role = 'trained' if wrong == 'condition' else 'pretrained'
    q = requests[1] if wrong == 'input' else requests[0]
    if wrong == 'injected':
        row['outcome']['provenance']['injected'] = True
    if wrong == 'launch-time':
        row['started_utc'] = '2000-01-01T00:00:00+00:00'
    with pytest.raises(ValueError, match='source|timestamp|launch'):
        study.validate_row(row, q, role, 1, configs, registry, 'software-fixture')


def test_immutable_record_and_exclusive_worker_lock(workspace):
    work, _, _ = workspace
    path = work / 'immutable.json'
    study.save(path, {'software_fixture': True})
    with pytest.raises(FileExistsError):
        study.save(path, {'replacement': True})
    assert study.read(path) == {'software_fixture': True}
    with study.exclusive(work / 'worker.lock'):
        with pytest.raises(OSError), study.exclusive(work / 'worker.lock'):
            pass
    with study.exclusive(work / 'worker.lock'):
        pass


def test_preflight_rechecks_every_bound_original_without_invoking_a_model(workspace, tmp_path, monkeypatch):
    work, configs, registry = workspace
    inputs, bindings, eligibility = (tmp_path / name for name in ('inputs.jsonl', 'input-bindings.jsonl', 'eligibility.json'))
    requests, sources = [], []
    for i in range(44):
        image = tmp_path / f'{i}.png'
        Image.new('RGB', (80, 60), (i, 0, 0)).save(image)
        q = request(image, i)
        requests.append(q)
        sources.append({'case_id': q['request_id'], 'image_sha256': study.file_sha256(image)})
    write_rows(inputs, requests)
    write_rows(bindings, sources)
    scorer = (study.ROOT / 'tools/passenger_reviewed_baseline.py').read_bytes().replace(b'\r\n', b'\n')
    study.save(eligibility, {'predeclared_utc': '2000-01-01T00:00:00+00:00', 'model_registry_sha256': 'a'*64,
                             'scorer_sha256_lf': sha256(scorer).hexdigest()})
    for key, val in {'INPUTS': inputs, 'INPUT_SHA': study.file_sha256(inputs), 'BINDING_HASH': study.file_sha256(bindings),
                     'ELIGIBILITY': eligibility, 'ELIGIBILITY_SHA': study.file_sha256(eligibility)}.items():
        monkeypatch.setattr(study, key, val)
    monkeypatch.setattr(study, 'registered_models', lambda: (configs, registry))
    monkeypatch.setattr(study, 'GroundingModel', lambda *a, **kw: pytest.fail('preflight attempted model construction'))
    assert len(study.preflight(create=True)[0]) == 44
    assert not (work / 'journal').exists()
    Image.new('RGB', (80, 60), (0, 200, 0)).save(tmp_path / '43.png')
    with pytest.raises(ValueError, match='Original photo changed'):
        study.preflight()


def test_normal_run_rejects_reordered_resolved_requests_even_with_rehashed_metadata(workspace, tmp_path, monkeypatch):
    work, configs, registry = workspace
    inputs = tmp_path / 'requests.jsonl'
    qs = [request(tmp_path / f'missing-{i}.jpg', i) for i in range(44)]
    write_rows(inputs, qs)
    monkeypatch.setattr(study, 'INPUT_SHA', study.file_sha256(inputs))
    monkeypatch.setattr(study, 'image_hashes', lambda: {q['request_id']: '0'*64 for q in qs})
    study.save(work / 'registration.json', {'software_fixture': True})
    seal = study.file_sha256(work / 'registration.json')
    folder, launch = begin(work, 'pretrained', 1)
    target = folder / 'normal-run'
    study.predict(inputs, configs['pretrained'], target, adapter_factory=lambda cfg: study.ObservedNative(
        'pretrained', folder, qs, configs, registry, seal))
    launch['exit_observed_utc'] = study.now()
    condition = {'started_utc': launch['started_utc'], 'attempts': [launch]}
    study.save(work / 'execution.json', {'conditions': {'pretrained': condition}})
    assert study.check_run(target, 'pretrained', qs, configs, registry)['operational_errors'] == {'invalid_image': 44}
    resolved = target / 'resolved-requests.jsonl'
    write_rows(resolved, list(reversed(qs)))
    manifest = study.read(target / 'run.json')
    manifest['resolved_requests_sha256'] = study.file_sha256(resolved)
    study.save(target / 'run.json', manifest, replace=True)
    with pytest.raises(ValueError, match='Incomplete native output'):
        study.check_run(target, 'pretrained', qs, configs, registry)
