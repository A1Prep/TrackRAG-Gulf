"""Explicit INTERPRET JSON constraints, independent of destination-only research.

The grammar constrains syntax and mutually exclusive field types. The native
model still chooses its status, question and coordinates. Original geometry
validation remains authoritative after decoding; no missing values are filled.
Imports stay lightweight until an application requests constrained generation.
"""

import hashlib
from importlib import metadata
import json


INTERPRET_CONTRACT = "interpret_v1"
LANDMARK_CONTRACT = "landmark_v1"
EMPTY_LANDMARK_CONTRACT = "landmark_without_candidates_v1"
# Whole-number coordinates keep the normalized grid auditable. Box ordering is validated after decoding.
_COORDINATE = r"(0|[1-9][0-9]{0,2}|1000)"
# This English interaction uses an explicit question alphabet. A negated character class would
# expand to the entire multilingual tokenizer alphabet at every prefix-trie node on a laptop.
EMPTY_LANDMARK_PATTERN = rf"(OBJECT( {_COORDINATE}){{4}}|ASK [A-Za-z][A-Za-z0-9 ,.'()/\-]+\?|MISSING)"
LMFE_COMMIT = "817f944fcc8851917c40300a47f62d1defc3ffc3"
LMFE_SOURCE_URL = f"https://github.com/noamgat/lm-format-enforcer/archive/{LMFE_COMMIT}.zip"
_FIELDS = ("status", "destination_point_xy", "destination_box_xyxy", "clarification_question")


def interpretation_schema():
    """Return three complete objects, with status selected before its fields."""
    branches = []
    for status in ("proposed", "needs_clarification", "insufficient_visual_evidence"):
        proposed = status == "proposed"
        branches.append({
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": [status]},
                "destination_point_xy": ({"type": "array", "items": {"type": "number"},
                                          "minItems": 2, "maxItems": 2} if proposed else {"type": "null"}),
                "destination_box_xyxy": ({"type": "array", "items": {"type": "number"},
                                          "minItems": 4, "maxItems": 4} if proposed else {"type": "null"}),
                "clarification_question": ({"type": "string", "minLength": 1, "maxLength": 2048}
                                           if status == "needs_clarification" else {"type": "null"}),
            },
            "required": list(_FIELDS),
            "additionalProperties": False,
        })
    return {"anyOf": branches}


def schema_sha256():
    encoded = json.dumps(interpretation_schema(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def new_interpretation_parser():
    """Create independent parser state; never cache a request's parser."""
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    return JsonSchemaParser(interpretation_schema(), config=CharacterLevelParserConfig(
        force_json_field_order=True, max_consecutive_whitespaces=12, max_json_array_length=4))


def new_empty_landmark_parser():
    from lmformatenforcer import RegexParser
    return RegexParser(EMPTY_LANDMARK_PATTERN)


def _dependency_identity():
    """Version 0.11.3 alone cannot identify the Transformers 5-compatible source."""
    try:
        distribution = metadata.distribution("lm-format-enforcer")
        source = json.loads(distribution.read_text("direct_url.json") or "null")
    except (metadata.PackageNotFoundError, TypeError, ValueError) as error:
        raise ImportError("Install requirements-grounding-format.txt for constrained interpretation") from error
    if not isinstance(source, dict) or source.get("url") != LMFE_SOURCE_URL:
        raise ImportError("Constrained interpretation requires the pinned lm-format-enforcer source archive")
    return {"package": "lm-format-enforcer", "package_version": distribution.version,
            "source_url": LMFE_SOURCE_URL, "upstream_commit": LMFE_COMMIT}


def _integration_helpers():
    # Import only the supported callback path. The diagnostic generate_enforced
    # wrapper uses generation internals that changed in Transformers 5.
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
    return build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn


class InterpretationConstraint:
    """One tokenizer trie per runtime, a fresh prefix parser for each request.

    Call only while the owning runtime's execution lock is held. Model output
    heads may contain padding beyond the tokenizer: none of those IDs belong to
    this grammar, even when the complete head remains part of the native model.
    """

    def __init__(self, tokenizer, model_vocab_size):
        tokenizer_vocab_size = len(tokenizer)
        if (type(model_vocab_size) is not int or model_vocab_size <= 0
                or type(tokenizer_vocab_size) is not int or tokenizer_vocab_size <= 0):
            raise ValueError("Constrained decoding requires positive tokenizer and model vocabulary sizes")
        self.vocab_size = min(tokenizer_vocab_size, model_vocab_size)
        eos = tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        if not eos_ids or any(type(value) is not int or not 0 <= value < self.vocab_size for value in eos_ids):
            raise ValueError("Tokenizer EOS must lie inside the usable output vocabulary")
        dependency = _dependency_identity()
        build_data, self._build_callback = _integration_helpers()
        self._tokenizer_data = build_data(tokenizer, use_bitmask=False, vocab_size=self.vocab_size)
        self.details = {
            "mode": "lm_format_enforcer_prefix_allowed_tokens", **dependency,
            "tokenizer_vocab_size": tokenizer_vocab_size, "model_output_vocab_size": model_vocab_size,
            "allowed_vocab_size": self.vocab_size, "use_bitmask": False,
        }

    def for_request(self):
        return self._for_parser(new_interpretation_parser(), {
            "contract": INTERPRET_CONTRACT, "schema_sha256": schema_sha256(), "force_json_field_order": True})

    def for_empty_landmarks(self):
        return self._for_parser(new_empty_landmark_parser(), {
            "contract": EMPTY_LANDMARK_CONTRACT, "regex": EMPTY_LANDMARK_PATTERN,
            "regex_sha256": hashlib.sha256(EMPTY_LANDMARK_PATTERN.encode()).hexdigest()})

    def _for_parser(self, parser, grammar_details):
        callback = self._build_callback(self._tokenizer_data, parser)
        limit = self.vocab_size

        def allowed_tokens(batch_id, sent):
            allowed = callback(batch_id, sent)
            if not allowed or any(type(value) is not int or not 0 <= value < limit for value in allowed):
                raise ValueError("Structured decoder returned empty or out-of-vocabulary token IDs")
            return allowed

        return allowed_tokens, {**self.details, **grammar_details}
