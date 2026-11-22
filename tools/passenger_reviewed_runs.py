"""Run the fixed, original-as-reviewed Gulf comparison, without opening evaluation targets.

Preflight is CPU-only. Run from the project root with the registered local Python.
Only the serial scheduler's GPU owner should invoke ``run``. ``verify`` checks
normal native output and its recovery journal without computing quality metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

from evaluation.grounding_inputs import TASK5_RUN_FILES
from tools.dropoff_native_episodes import digest, moment
from tools.dropoff_predict import predict, read_requests
from tools.passenger_reviewed_baseline import BINDING_HASH, registered_models, verify_outcome
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.inputs import _inspect_image
from trackrag.dropoff.model import GroundingModel


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/task5-independent-typed-20261122/gulf'
INPUTS = ROOT / 'runs/task5-evaluator-delivery-20261122/installed/inputs/requests-as-human-reviewed.jsonl'
ELIGIBILITY = ROOT / 'experiments/dropoff/passenger-reviewed-original-eligibility-20261122.json'
INPUT_SHA = 'fe8c8e310bc1c44eb8e768eb05b0f591c224c9baab85c838050d8954d777ad2c'
ELIGIBILITY_SHA = 'b101317179063159e8c01f9fd10ba848422fe3473b7ac0c0d3d2f42357951dcd'
ROLES = ('pretrained', 'trained')
SOURCES = (
    'tools/passenger_reviewed_runs.py', 'tools/dropoff_predict.py', 'tools/passenger_reviewed_baseline.py',
    'tools/dropoff_native_episodes.py', 'evaluation/grounding_inputs.py', 'trackrag/dropoff/model.py',
    'trackrag/dropoff/inputs.py', 'trackrag/dropoff/contracts.py', 'trackrag/dropoff/data.py',
    'trackrag/dropoff/formatting.py', 'trackrag/dropoff/streamed.py', 'trackrag/dropoff/streamed_edges.py',
    'trackrag/dropoff/streamed_weights.py', 'trackrag/dropoff/structured.py', 'trackrag/dropoff/projected_pair.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def image_hashes():
    path = INPUTS.parent / 'input-bindings.jsonl'
    require(file_sha256(path) == BINDING_HASH, 'Original Task 1 input bindings differ')
    return {entry['case_id']: entry['image_sha256'] for entry in rows(path)}


def save(path, value, *, replace=False):
    """Publish a complete fsynced file; journal members can only be created once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
        try:
            tmp.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
            tmp.flush()
            os.fsync(tmp.fileno())
        except BaseException:
            tmp.close()
            temp_path.unlink(missing_ok=True)
            raise
    try:
        if replace:
            os.replace(temp_path, path)
        else:
            os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def exclusive(path):
    """File locks are released by the OS even if the supervisor is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        if not lock.seek(0, 2):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def preflight(*, create=False):
    require(Path.cwd().resolve() == ROOT, 'Run from the project root so native model paths stay unchanged')
    require(file_sha256(ELIGIBILITY) == ELIGIBILITY_SHA and file_sha256(INPUTS) == INPUT_SHA,
            'Reviewed inputs or fixed preanalysis differ')
    eligibility = read(ELIGIBILITY)
    checked_scorer = (ROOT / 'tools/passenger_reviewed_baseline.py').read_bytes().replace(b'\r\n', b'\n')
    require(sha256(checked_scorer).hexdigest() == eligibility['scorer_sha256_lf'], 'Scorer differs from preanalysis')
    configs, registry = registered_models()
    requests = read_requests(INPUTS)
    require(len(requests) == 44 and all(row['input_mode'] == 'typed' and row['scene']['tracks'] == []
                                         for row in requests), 'Unexpected reviewed input mode or denominator')
    originals = image_hashes()
    require(set(originals) == {r['request_id'] for r in requests}, 'Input or photo binding missing')
    for request in requests:
        scene = request['scene']
        require(_inspect_image(scene['frame'])[1] == originals[request['request_id']], 'Original photo changed')
    spec = {'kind': 'direct_native_reviewed_gulf_execution', 'eligibility_sha256': ELIGIBILITY_SHA,
            'requests_sha256': INPUT_SHA, 'original_photo_bindings_sha256': BINDING_HASH,
            'requested_each': len(requests), 'ordering': list(ROLES),
            'config_sha256': {role: registry[role]['config_sha256'] for role in ROLES},
            'source_sha256': {name: file_sha256(ROOT / name) for name in SOURCES},
            'source_hash_convention': 'Exact bytes of this executing checkout, including line endings.',
            'registered_models_sha256': eligibility['model_registry_sha256'],
            'adapter_manifest_sha256': registry['trained']['adapter_manifest_sha256'],
            'adapter_weights_sha256': registry['trained']['adapter_weights_sha256']}
    path = WORK / 'registration.json'
    if create and not path.exists():
        save(path, {'registered_utc': now(), **spec})
    frozen = read(path)
    require({key: val for key, val in frozen.items() if key != 'registered_utc'} == spec, 'Registered sources changed')
    require(moment(eligibility['predeclared_utc']) <= moment(frozen['registered_utc']), 'Invalid registration order')
    return requests, configs, registry


def row_path(role, index):
    return WORK / 'journal' / role / f'{index:03}.json'


def journal_prefix(role):
    files = sorted((WORK / 'journal' / role).glob('*.json'))
    require(len(files) <= 44 and [p.name for p in files] == [f'{i:03}.json' for i in range(1, len(files) + 1)],
            'A previously finished native journal has a missing or unexpected member')
    return len(files)


def load_journal(path):
    envelope = read(path)
    require(envelope['record_sha256'] == digest(envelope['record']), 'Raw call journal content has changed')
    return envelope['record']


def validate_row(row, request, role, index, configs, registry, seal):
    require(row['index'] == index and row['role'] == role and row['registration_sha256'] == seal
            and row['request_sha256'] == digest(request), 'Journal source or position differs')
    attempt = WORK / 'attempts' / row['attempt']
    launch = read(attempt / 'launch.json')
    require(launch['role'] == role and launch['attempt'] == row['attempt'] and launch['eligibility_sha256'] == ELIGIBILITY_SHA
            and moment(launch['started_utc']) <= moment(row['started_utc']) <= moment(row['end_utc']) <= moment(now()),
            'Journal timestamp or original process launch differs')
    started = attempt / 'started-calls' / f'{index:03}.json'
    require(read(started) == {k: row[k] for k in ('index', 'role', 'registration_sha256', 'attempt', 'request_sha256', 'started_utc')},
            'Original call start differs from retained native receipt')
    verify_outcome(request, row['receipt'], row['outcome'], role, configs[role], registry,
                   image_hashes()[request['request_id']])


class ObservedNative:
    """Delegate once for new rows and preserve validated original receipts across interruptions."""

    def __init__(self, role, attempt, requests, configs, registry, seal):
        self.role, self.attempt, self.requests = role, attempt, requests
        self.configs, self.registry, self.seal = configs, registry, seal
        self.model = GroundingModel(configs[role])
        self.index = 0

    def interpret_with_receipt(self, request):
        self.index += 1
        index = self.index
        require(self.requests[index - 1] == request, 'Unexpected request sequence')
        path = row_path(self.role, index)
        if path.exists():
            row = load_journal(path)
        else:
            binding = {'index': index, 'role': self.role, 'registration_sha256': self.seal,
                       'attempt': self.attempt.name, 'request_sha256': digest(request), 'started_utc': now()}
            save(self.attempt / 'started-calls' / f'{index:03}.json', binding)
            outcome, receipt = self.model.interpret_with_receipt(request)
            row = {**binding, 'end_utc': now(), 'outcome': outcome, 'receipt': receipt}
            save(path, {'record_sha256': digest(row), 'record': row})
        validate_row(row, request, self.role, index, self.configs, self.registry, self.seal)
        return row['outcome'], row['receipt']

    def close(self):
        self.model.close()


def check_run(folder, role, requests, configs, registry):
    summary = read(folder / 'run.json')
    require(summary['configuration'] == asdict(configs[role]) and summary['requests_sha256'] == INPUT_SHA
            and file_sha256(folder / 'input-snapshot.jsonl') == INPUT_SHA and summary['requested_rows'] == 44,
            'Normal run does not match registered sources')
    for filename, field in TASK5_RUN_FILES:
        require(file_sha256(folder / filename) == summary[field], 'Normal run member changed: ' + filename)
    resolved, outcomes, receipts = [rows(folder / path) for path, _ in TASK5_RUN_FILES]
    require(resolved == requests and len(outcomes) == len(receipts) == 44, 'Incomplete native output')
    seal = file_sha256(WORK / 'registration.json')
    require(journal_prefix(role) == 44, 'Missing native calls in the private journal')
    errors = Counter()
    journal_hashes = []
    condition = read(WORK / 'execution.json')['conditions'][role]
    attempts = {entry['attempt']: entry for entry in condition['attempts']}
    previous_end = moment(condition['started_utc'])
    for index, (request, outcome, receipt) in enumerate(zip(requests, outcomes, receipts, strict=True), 1):
        path = row_path(role, index)
        row = load_journal(path)
        validate_row(row, request, role, index, configs, registry, seal)
        original = attempts[row['attempt']]
        require(moment(original['started_utc']) <= moment(row['started_utc']) and previous_end <= moment(row['started_utc'])
                and moment(row['end_utc']) <= moment(original['exit_observed_utc']),
                'Native calls overlap or a call falls outside its original process lifetime')
        previous_end = moment(row['end_utc'])
        require(row['receipt'] == receipt and row['outcome'] == outcome, 'Normal output differs from durable native call')
        journal_hashes.append(file_sha256(path))
        if not outcome['ok']:
            errors[outcome['error']['code']] += 1
    require(summary['operational_errors'] == sum(errors.values()) and summary['semantic_results'] == 44 - sum(errors.values()),
            'Native counts differ from retained calls')
    return {'run_sha256': file_sha256(folder / 'run.json'), 'rows': 44,
            'semantic_results': summary['semantic_results'], 'operational_errors': dict(errors),
            'ordered_individual_journal_hashes_sha256': digest(journal_hashes)}


def child(role, attempt_name):
    requests, configs, registry = preflight()
    attempt = WORK / 'attempts' / attempt_name
    require(attempt.parent == WORK / 'attempts' and attempt == attempt.resolve(), 'Invalid attempt path')
    launch = read(attempt / 'launch.json')
    require(launch['role'] == role and launch['attempt'] == attempt_name and launch['eligibility_sha256'] == ELIGIBILITY_SHA,
            'Child process missing valid launch intent')
    with exclusive(WORK / 'native.lock'):
        seal = file_sha256(WORK / 'registration.json')
        predict(INPUTS, configs[role], attempt / 'normal-run',
                adapter_factory=lambda config: ObservedNative(role, attempt, requests, configs, registry, seal))
        save(attempt / 'closed.json', {'closed_utc': now(), 'process_id': os.getpid(), 'role': role})


def run():
    import psutil
    requests, configs, registry = preflight()
    require(Path(sys.executable).samefile(ROOT / '.venv-grounding-local/Scripts/python.exe'), 'Use registered local interpreter')
    execution_path = WORK / 'execution.json'
    with exclusive(WORK / 'supervisor.lock'):
        with exclusive(WORK / 'native.lock'):
            pass  # An orphan native supervisor must close before a recovery starts.
        state = read(execution_path) if execution_path.exists() else {
            'eligibility_sha256': ELIGIBILITY_SHA, 'requests_sha256': INPUT_SHA,
            'registration_sha256': file_sha256(WORK / 'registration.json'), 'conditions': {}, 'status': 'running'}
        require(state['registration_sha256'] == file_sha256(WORK / 'registration.json'), 'Execution seal differs')
        for role in ROLES:
            item = state['conditions'].setdefault(role, {'started_utc': now(), 'attempts': []})
            target = WORK / role
            if 'run_sha256' in item:
                require(check_run(target, role, requests, configs, registry)['run_sha256'] == item['run_sha256'],
                        'Published native output differs')
                continue
            if item['attempts']:
                last = item['attempts'][-1]
                if 'pid' in last and psutil.pid_exists(last['pid']):
                    try:
                        alive = psutil.Process(last['pid']).create_time() == last['process_created']
                    except psutil.NoSuchProcess:
                        alive = False
                    require(not alive, 'Previous owned native process is still active')
                previous = WORK / 'attempts' / last['attempt']
                if not last.get('exit_observed_utc'):
                    last['exit_observed_utc'] = now()
                    last['recovered_after_supervisor_interruption'] = True
                    save(execution_path, state, replace=True)
                staging = previous / 'normal-run'
                if (previous / 'closed.json').is_file() and (target.exists() or (staging / 'run.json').is_file()):
                    details = check_run(target if target.exists() else staging, role, requests, configs, registry)
                    if not target.exists():
                        staging.rename(target)
                    item.update(end_utc=last['exit_observed_utc'], **details)
                    save(execution_path, state, replace=True)
                    continue
            number = len(item['attempts']) + 1
            require(number <= 2, 'Preserved interrupted attempts; investigate before any further launch')
            attempt_name = f'{role}-{number:02}'
            attempt = WORK / 'attempts' / attempt_name
            complete = journal_prefix(role)
            started_before = {int(p.stem) for p in (WORK / 'attempts').glob(f'{role}-*/started-calls/*.json')}
            launch = {'role': role, 'attempt': attempt_name, 'started_utc': now(), 'eligibility_sha256': ELIGIBILITY_SHA,
                      'previously_journaled_calls': complete,
                      'previous_started_calls_without_durable_native_response': sorted(n for n in started_before if n > complete)}
            item['attempts'].append(launch.copy())
            state['status'] = 'running'
            save(execution_path, state, replace=True)  # Durable times before Popen.
            save(attempt / 'launch.json', launch)
            with (attempt / 'native.log').open('xb') as log:
                command = [sys.executable, '-B', '-u', '-m', 'tools.passenger_reviewed_runs', '_child', role, attempt_name]
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                current = item['attempts'][-1]
                current.update(pid=proc.pid, process_created=psutil.Process(proc.pid).create_time())
                save(execution_path, state, replace=True)
                code = proc.wait()
                current.update(exit_code=code, exit_observed_utc=now())
                save(execution_path, state, replace=True)
            if code or not (attempt / 'closed.json').is_file():
                state['status'] = 'interrupted'
                save(execution_path, state, replace=True)
                raise RuntimeError(f'{role} process exited {code}. Inspect preserved attempt before recovery.')
            staging = attempt / 'normal-run'
            details = check_run(staging, role, requests, configs, registry)
            staging.rename(target)
            item.update(end_utc=current['exit_observed_utc'], **details)
            save(execution_path, state, replace=True)
        state.update(status='completed', verification_utc=now())
        save(execution_path, state, replace=True)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'run', 'verify', '_child'))
    parser.add_argument('child_args', nargs='*')
    args = parser.parse_args()
    if args.action == '_child':
        role, name = args.child_args
        require(role in ROLES, 'Unknown condition')
        child(role, name)
    elif args.action == 'run':
        print(json.dumps(run(), indent=2))
    else:
        requests, configs, registry = preflight(create=args.action == 'preflight')
        if args.action == 'verify':
            print(json.dumps({role: check_run(WORK / role, role, requests, configs, registry) for role in ROLES}, indent=2))
        else:
            print(json.dumps({'registered_rows': len(requests), 'roles': list(configs), 'native_calls': 0,
                              'registration_sha256': file_sha256(WORK / 'registration.json')}, indent=2))


if __name__ == '__main__':
    main()
