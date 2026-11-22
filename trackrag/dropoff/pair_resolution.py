"""Resolve two explicitly named 'between' references against the same original photograph."""

from hashlib import sha256
import json
import re

from .contracts import validate_request
from .landmarks import landmark_messages, parse_landmark, resolve_image
from .model import verify_adapter
from .structured import EMPTY_LANDMARK_CONTRACT, LANDMARK_CONTRACT


VERSION = "explicit_between_references_v1"
_CORRECTION = re.compile(r"\b(?:actually|instead(?!\s+of)|I mean|forget (?:that|it)|change (?:that|it))\b", re.I)
_NEGATED = re.compile(r"\b(?:not|never|don't|don’t|avoid|rather than|instead of)\b[^,;.!?]{0,90}$", re.I)
_TAIL = re.compile(r"[,;.!?\n]|\s+\b(?:just\s+)?(?:behind|ahead of|in front of|next to)\s+(?:them|both)\b|\s+(?:please|if possible|instead)\s*[,;.!?]*$", re.I)
_DETERMINER = re.compile(r"(?:the|a|an|that|this|my|your|our|their|another)\s+", re.I)
_INCOMPLETE = re.compile(r"(?:(?:the|a|an|that|this)\s+)?(?:one|two|first|second|other|top|bottom|left|right|small|large|big|black|white|red|blue|green|yellow|grey|gray|silver)\Z", re.I)


class PairResolutionError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def explicit_pair_quotes(text):
    """Conservatively return two literal spans from the current turn. Leave elliptical/plural forms intact."""
    matches = list(re.finditer(r"\bbetween\b", text, re.I))
    if not matches:
        return None
    between = matches[-1]
    # A fresh instruction introduced after the pair supersedes it. A final 'instead' qualifies the pair itself.
    if _NEGATED.search(text[:between.start()]) or any(
            text[m.end():].strip(" ,.;:!?") for m in _CORRECTION.finditer(text, between.end())):
        return None
    stop = _TAIL.search(text, between.end())
    end = stop.start() if stop else len(text)
    phrase = text[between.end():end]
    if re.search(r"\b(?:or|not|except)\b", phrase, re.I):
        return None
    joins = list(re.finditer(r"\s+and\s+", phrase, re.I))
    if len(joins) > 1:
        joins = [j for j in joins if _DETERMINER.match(phrase[j.end():])]
    if len(joins) != 1:
        return None
    join = joins[0]
    spans = [(between.end(), between.end()+join.start()), (between.end()+join.end(), end)]
    quotes = []
    for start, stop in spans:
        raw = text[start:stop]
        quote = raw.strip(" \t\"“”")
        if (not quote or len(quote) > 180 or not re.search(r"[a-z]", quote, re.I)
                or _INCOMPLETE.fullmatch(quote) or re.search(r"\b(?:between|instead|and the)\b", quote, re.I)):
            return None
        start += raw.index(quote)
        quotes.append({"text": quote, "start": start, "end": start+len(quote)})
    return quotes


def _subrequest(request, quote, index):
    # Derived model queries are separate from the actual rider's utterance and speech transcript.
    return {"request_id": f"pair-{_digest(request)}-{index}", "scene": request["scene"],
            "text": f"Beside {quote['text']}, please.", "input_mode": "typed", "dialogue": []}


def collect_pair_evidence(request, image, resolver):
    """Return an auditable record, or None when the current rider turn does not name two references."""
    from .passenger import operational_code
    request = validate_request(request)
    quotes = explicit_pair_quotes(request['text'])
    if quotes is None:
        return None
    original, binding = resolve_image(request, resolver.config)
    with original:
        if image.size != original.size or image.mode != original.mode or image.tobytes() != original.tobytes():
            raise PairResolutionError('artifact_invalid', 'The pair queries require the same verified original photograph.')
    evidence = {'version': VERSION, 'request_sha256': _digest(request), 'image': binding,
                'source_text_sha256': sha256(request['text'].encode()).hexdigest(), 'subqueries': []}
    for index, quote in enumerate(quotes):
        subrequest = _subrequest(request, quote, index)
        row = {'quote': quote, 'subrequest_sha256': _digest(subrequest)}
        try:
            row['resolver'] = resolver.resolve(subrequest, image)
        except Exception as exc:
            row['error'] = {'code': operational_code(exc), 'message': str(exc)[:4000] or type(exc).__name__}
        evidence['subqueries'].append(row)
    evidence['receipt_sha256'] = _digest(evidence)
    return evidence


def replay_pair_decision(request, binding, evidence, config):
    """Revalidate both raw subdecisions. Never assign a destination point or supply a missing identity."""
    request = validate_request(request)
    quotes = explicit_pair_quotes(request['text'])
    if not isinstance(evidence, dict):
        raise PairResolutionError('artifact_invalid', 'Pair evidence must be a serialized object.')
    payload = {k:v for k,v in evidence.items() if k != 'receipt_sha256'}
    rows = evidence.get('subqueries')
    if (quotes is None or evidence.get('receipt_sha256') != _digest(payload) or evidence.get('version') != VERSION
            or evidence.get('request_sha256') != _digest(request) or evidence.get('image') != binding
            or binding.get('sha256') != request['scene']['provenance'].get('image_sha256')
            or evidence.get('source_text_sha256') != sha256(request['text'].encode()).hexdigest()
            or not isinstance(rows, list) or len(rows) != 2):
        raise PairResolutionError('artifact_invalid', 'Pair evidence differs from its active instruction or original image.')
    adapter = verify_adapter(config)
    decisions = []
    for i, (row, quote) in enumerate(zip(rows, quotes)):
        sub = _subrequest(request, quote, i)
        request_hash = _digest(sub)
        if not isinstance(row, dict) or row.get('quote') != quote or row.get('subrequest_sha256') != request_hash:
            raise PairResolutionError('artifact_invalid', 'A quoted reference differs from the active instruction.')
        if 'error' in row:
            if 'resolver' in row or not isinstance(row['error'], dict) or not isinstance(row['error'].get('code'), str):
                raise PairResolutionError('artifact_invalid', 'A pair subquery has a malformed error receipt.')
            raise PairResolutionError(row['error']['code'], str(row['error'].get('message', 'Pair model call failed.')))
        model = row.get('resolver')
        messages, ids = landmark_messages(sub)
        expected = {'request_sha256': request_hash, 'image_sha256': binding['sha256'], 'model_id': config.model_id,
                    'revision': config.revision, 'adapter': adapter, 'messages': messages,
                    'messages_sha256': sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                    'contract': LANDMARK_CONTRACT if ids else EMPTY_LANDMARK_CONTRACT}
        if not isinstance(model, dict) or any(k not in model or model[k] != v for k,v in expected.items()):
            raise PairResolutionError('artifact_invalid', 'A pair model response belongs to a different prompt, model or frame.')
        try:
            parsed = parse_landmark(model.get('raw_text'), ids, request['scene']['frame'])
        except ValueError as error:
            raise PairResolutionError('invalid_model_output', str(error)) from error
        if model.get('decision') != parsed or 'parse_error' in model:
            raise PairResolutionError('artifact_invalid', 'The raw pair model reply differs from its saved parsed decision.')
        decisions.append(parsed)
    if any(d['kind'] == 'missing' for d in decisions):
        return {'kind': 'missing'}
    for quote, decision in zip(quotes, decisions):
        if decision['kind'] == 'ask':
            return {'kind': 'ask', 'question': decision['question']}
        if len(decision['boxes']) != 1:
            return {'kind': 'ask', 'question': f"Which single landmark do you mean by “{quote['text']}”?"}
    a, b = (d['boxes'][0] for d in decisions)
    identities = [d['track_ids'][0] if d['track_ids'] else None for d in decisions]
    labels = [' '.join(re.findall(r"\w+", re.sub(r'^(?:the|an?)\s+', '', q['text'].casefold()))) for q in quotes]
    overlap = min(a[2],b[2]) > max(a[0],b[0]) and min(a[3],b[3]) > max(a[1],b[1])
    if labels[0] == labels[1] or overlap or (identities[0] is not None and identities[0] == identities[1]):
        return {'kind':'ask', 'question':'The two descriptions point to the same or overlapping visible area. Which two separate landmarks do you mean?'}
    tracked = all(identity is not None for identity in identities)
    # An untracked object must never shift a later detector ID onto the wrong box in the visual verifier.
    return {'kind':'landmark' if tracked else 'object', 'boxes':[a,b],
            'track_ids':identities if tracked else [], 'anchor_track_ids':identities}
