"""Prepare matched, annotation-free direct 8B evaluation on the frozen development requests."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

from tools.dropoff_adapt_local import parent_context
from tools.dropoff_ai_continuation import MODEL_STAGE, PINS, _json_sha, prepare as training_data, training_messages
from tools.dropoff_predict import read_requests
from tools.dropoff_train import write_json
from trackrag.dropoff.data import inference_input_sha256
from trackrag.dropoff.formatting import build_messages
from trackrag.dropoff.landmarks import resolve_image
from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION, GroundingConfig, verify_adapter


def development_inputs(requests, binding_path):
    raw = binding_path.read_bytes().replace(b'\r\n', b'\n')
    if sha256(raw).hexdigest() != PINS['input-bindings.jsonl']:
        raise ValueError('Portable development input bindings changed')
    bank = {r['request_id']: r for r in map(json.loads, raw.splitlines()) if r['recording_id'] == 'video_054907'}
    rows = read_requests(requests)
    if len(rows) != 19 or {r['request_id'] for r in rows} != set(bank):
        raise ValueError('Matched direct evaluation requires all 19 original development requests')
    for r in rows:
        ref = bank[r['request_id']]
        image_sha = r['scene']['provenance']['image_sha256']
        if (ref['image_sha256'] != image_sha or r['scene']['frame']['recording_id'] != 'video_054907'
                or ref['inference_input_sha256'] != inference_input_sha256(r, image_sha)):
            raise ValueError('Matched development scene or instruction changed')
    return rows


def checked_token_counter(config, checks, args):
    """Use the release tokenizer without importing Torch alongside a live GPU job.

    The actual processor has already measured merged visual tokens for these image
    dimensions. Reproduce every actual full training-token total before estimating
    any matching-size evaluation input. Unknown image geometry is rejected.
    """
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    from types import SimpleNamespace

    snapshot = config.resolved_cache_dir() / ('models--' + BASE_MODEL_ID.replace('/', '--')) / 'snapshots' / BASE_REVISION
    tokenizer = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
    template = json.loads((snapshot / 'chat_template.json').read_bytes())['chat_template']
    compiler = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template)
    sizes = {}

    def length(messages, size, *, generation):
        if size not in sizes:
            raise ValueError('Evaluation image geometry lacks an actual pinned-processor visual token measurement')
        rendered = compiler.render(messages=messages, add_generation_prompt=generation, enable_thinking=False)
        placeholder = '<|image_pad|>'
        if rendered.count(placeholder) != 1:
            raise ValueError('Expected exactly one native visual token placeholder')
        rendered = rendered.replace(placeholder, placeholder * sizes[size])
        return len(tokenizer.encode(rendered, add_special_tokens=False).ids)

    _, _, cases, _ = training_data(SimpleNamespace(parent_run=args.parent, config=args.parent/'inference_config.json',
        annotations=args.bindings.parent, requests=args.training_requests))
    for case, known in zip(cases, checks['rows'], strict=True):
        f = case['request']['scene']['frame']
        key = (f['width'], f['height'])
        if sizes.setdefault(key, known['image_tokens']) != known['image_tokens']:
            raise ValueError('Inconsistent actual image-token counts on identical original geometry')
        if length(training_messages(case), key, generation=False) != known['full_input_tokens']:
            raise ValueError('Tokenizer-only length did not reproduce the actual pinned processor')
    return length


def prepare(args):
    base, _, parent = parent_context(args.parent, args.parent / 'inference_config.json')
    train_check = json.loads(args.preflight.read_bytes())
    if (train_check['parent_sha256'] != _json_sha(parent) or train_check['kind'] != MODEL_STAGE
            or train_check['observation_format'] != 'compact_rows_v1'):
        raise ValueError('Local preflight and public parent are not the same experimental condition')
    config = replace(base, max_input_tokens=train_check['token_budget'], inference_resident_layers=8,
                     observation_format='compact_rows_v1')
    rows = development_inputs(args.requests, args.bindings)
    configs = {'pretrained': replace(config, adapter_dir=None), 'public_parent': config}
    if args.child is not None:
        trained = json.loads((args.child / 'training_result.json').read_bytes())
        child = GroundingConfig.from_mapping(json.loads((args.child / 'inference_config.json').read_bytes()))
        proof = verify_adapter(child)
        if (trained.get('kind') != MODEL_STAGE or proof is None or trained['changed_tensors'] < 1
                or trained['files']['adapter_model.safetensors'] != proof['files']['adapter_model.safetensors']['sha256']
                or replace(child, adapter_dir=config.adapter_dir) != config
                or _json_sha(json.loads((args.child / 'parent_binding.json').read_bytes())) != _json_sha(parent)):
            raise ValueError('Trained adapter does not match the frozen local comparison profile')
        configs['local_child'] = child
    args.out.mkdir(parents=True, exist_ok=False)
    length = checked_token_counter(config, train_check, args)
    checked = []
    for r in rows:
        messages = build_messages(r, observation_format=config.observation_format)
        image, binding = resolve_image(r, config)
        with image:
            total = length(messages, image.size, generation=True)
        checked.append({'request_id': r['request_id'], 'input_tokens': total, 'image_sha256': binding['sha256'],
                        'within_budget': total <= config.max_input_tokens,
                        'messages_sha256': _json_sha(messages)})
    for name, cond in configs.items():
        write_json(args.out / (name + '.json'), asdict(cond))
    result = {'kind': 'CPU preflight for normal direct four-field interpretation', 'predictions_generated': False,
              'summary': {'requests': len(rows), 'over_budget': sum(not r['within_budget'] for r in checked),
                          'minimum_input_tokens': min(r['input_tokens'] for r in checked),
                          'maximum_input_tokens': max(r['input_tokens'] for r in checked)},
              'budget': config.max_input_tokens, 'rows': checked, 'conditions': list(configs),
              'token_count_method': 'Pinned tokenizer and native chat template; matched all 17 actual Qwen processor totals and reused measured visual-token count for verified equal dimensions',
              'settings_differences': ['adapter_dir'], 'inputs': str(args.requests),
              'requests_sha256': sha256(args.requests.read_bytes()).hexdigest(), 'evaluation_labels_read': False}
    write_json(args.out / 'preflight.json', result)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=Path('runs/dropoff-handoff-v2-local-20261121'))
    p.add_argument('--preflight', type=Path, default=Path('runs/passenger-ai-continuation/preflight-compact.json'))
    p.add_argument('--requests', type=Path, default=Path('runs/passenger-references/development-prepared/requests.jsonl'))
    p.add_argument('--training-requests', type=Path, default=Path('runs/passenger-references/train-prepared/requests.jsonl'))
    p.add_argument('--bindings', type=Path, default=Path('annotations/dropoff/passenger-development/input-bindings.jsonl'))
    p.add_argument('--child', type=Path, help='Completed AI-supervised local training run, if available')
    p.add_argument('--out', type=Path, required=True)
    prepare(p.parse_args())
