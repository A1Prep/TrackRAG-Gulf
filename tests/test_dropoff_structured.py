"""CPU format-contract tests; no model weights, CUDA calls or quality claims."""

from contextlib import nullcontext
from importlib import metadata
import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from trackrag.dropoff import structured, streamed


def proposal(**changes):
    return {"status": "proposed", "destination_point_xy": [13.5, 22],
            "destination_box_xyxy": [0, 0, 100, 100], "clarification_question": None, **changes}


def grammar_accepts(value):
    pytest.importorskip("lmformatenforcer")
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    parser = structured.new_interpretation_parser()
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.mark.parametrize("value", [
    proposal(),
    proposal(destination_point_xy=[0, -3.14], destination_box_xyxy=[1e2, 3, 8, 1000]),
    {"status": "needs_clarification", "destination_point_xy": None, "destination_box_xyxy": None,
     "clarification_question": "Which of the two cars do you mean?"},
    {"status": "insufficient_visual_evidence", "destination_point_xy": None,
     "destination_box_xyxy": None, "clarification_question": None},
])
def test_real_grammar_accepts_complete_branch_structures(value):
    assert grammar_accepts(value)


@pytest.mark.parametrize("value", [
    {"status": "proposed"},
    {key: value for key, value in proposal().items() if key != "destination_point_xy"},
    proposal(status="needs_clarification", clarification_question="Which car?"),
    proposal(status="insufficient_visual_evidence"),
    proposal(clarification_question="Which car?"),
    proposal(destination_point_xy=None),
    proposal(destination_point_xy=[1]),
    proposal(destination_point_xy=[1, 2, 3]),
    proposal(destination_box_xyxy=[0, 0, 1]),
    proposal(destination_box_xyxy=[0, 0, 1, 1, 2]),
    proposal(destination_point_xy=[True, 2]),
    proposal(destination_point_xy=["1", 2]),
    proposal(status="confirmed"),
    proposal(extra="forbidden"),
    {"status": "needs_clarification", "destination_point_xy": None, "destination_box_xyxy": None,
     "clarification_question": ""},
    {"status": "needs_clarification", "destination_point_xy": None, "destination_box_xyxy": None,
     "clarification_question": None},
    {"status": "insufficient_visual_evidence", "destination_point_xy": None,
     "destination_box_xyxy": None, "clarification_question": "Which car?"},
    '{"destination_point_xy":[1,2],"status":"proposed","destination_box_xyxy":[0,0,3,4],"clarification_question":null}',
    '{"status":"proposed","destination_point_xy":[1,2],"destination_point_xy":[2,3],"destination_box_xyxy":[0,0,3,4],"clarification_question":null}',
])
def test_real_grammar_rejects_partial_contradictory_or_unordered_outputs(value):
    assert not grammar_accepts(value)


def test_geometry_still_requires_existing_semantic_validation():
    from trackrag.dropoff.formatting import parse_model_output
    value = proposal(destination_point_xy=[-3, 22])
    assert grammar_accepts(value)  # Syntax alone does not assert a valid destination.
    with pytest.raises(ValueError):
        parse_model_output(json.dumps(value), 640, 480)


def test_schema_is_fresh_and_hash_binds_all_branches():
    before = structured.schema_sha256()
    first = structured.interpretation_schema()
    first["anyOf"][0]["required"].pop()
    assert structured.schema_sha256() == before
    for branch in structured.interpretation_schema()["anyOf"]:
        assert branch["required"] == list(branch["properties"])
        assert branch["additionalProperties"] is False


def test_import_does_not_load_nn_or_optional_decoder_dependencies():
    result = subprocess.run([sys.executable, "-c", "import sys; import trackrag.dropoff.structured; "
        "assert not any(x in sys.modules for x in ['torch','transformers','peft','lmformatenforcer'])"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_dependency_rejects_pypi_version_without_pinned_source(monkeypatch):
    monkeypatch.setattr(structured.metadata, "distribution", lambda name:
                        SimpleNamespace(version="0.11.3", read_text=lambda name: None))
    with pytest.raises(ImportError, match="pinned"):
        structured._dependency_identity()


def test_dependency_rejects_missing_package(monkeypatch):
    def missing(name): raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(structured.metadata, "distribution", missing)
    with pytest.raises(ImportError, match="requirements-grounding-format"):
        structured._dependency_identity()


def test_dependency_records_actual_pinned_distribution(monkeypatch):
    monkeypatch.setattr(structured.metadata, "distribution", lambda name:
        SimpleNamespace(version="0.11.3", read_text=lambda name: json.dumps({"url": structured.LMFE_SOURCE_URL})))
    identity = structured._dependency_identity()
    assert identity["upstream_commit"] == structured.LMFE_COMMIT
    assert identity["package_version"] == "0.11.3"


class FakeTokenizer:
    eos_token_id = 151645
    def __len__(self): return 151669


def mock_integrations(monkeypatch, *, allowed=None):
    data, parsers = [], []
    def build_data(tokenizer, **kwargs):
        result = {"tokenizer": tokenizer, **kwargs}
        data.append(result)
        return result
    def build_callback(trie, parser):
        assert trie is data[0]
        parsers.append(parser)
        return lambda batch_id, sent: [0, 151645] if allowed is None else allowed
    monkeypatch.setattr(structured, "_dependency_identity", lambda: {"upstream_commit": structured.LMFE_COMMIT})
    monkeypatch.setattr(structured, "_integration_helpers", lambda: (build_data, build_callback))
    monkeypatch.setattr(structured, "new_interpretation_parser", object)
    return data, parsers


def test_cached_trie_fresh_parser_and_padded_output_ids_excluded(monkeypatch):
    data, parsers = mock_integrations(monkeypatch)
    constraint = structured.InterpretationConstraint(FakeTokenizer(), 151936)
    first, details = constraint.for_request()
    second, _ = constraint.for_request()
    assert len(data) == 1
    assert data[0]["vocab_size"] == 151669 and data[0]["use_bitmask"] is False
    assert len(parsers) == 2 and parsers[0] is not parsers[1]
    assert first is not second
    assert first(0, None) == [0, 151645]
    assert details["model_output_vocab_size"] - details["allowed_vocab_size"] == 267


@pytest.mark.parametrize("allowed", [[], [151669], [-1], [True]])
def test_callback_invalid_or_padded_ids_fail_without_fallback(monkeypatch, allowed):
    mock_integrations(monkeypatch, allowed=allowed)
    callback, _ = structured.InterpretationConstraint(FakeTokenizer(), 151936).for_request()
    with pytest.raises(ValueError, match="token IDs"):
        callback(0, None)


@pytest.mark.parametrize("head", [True, 0, -1, 151645])
def test_invalid_head_or_eos_range_rejected_before_integration(head):
    with pytest.raises(ValueError):
        structured.InterpretationConstraint(FakeTokenizer(), head)


class FakeTensor:
    def __init__(self, width): self.shape = (1, width)
    def __getitem__(self, key): return FakeTensor(self.shape[-1] - key[1].start)


class FakeInputs(dict):
    def to(self, device): return self


def fake_runtime():
    calls = []
    runtime = object.__new__(streamed.StreamedRuntime)
    runtime.execution_lock = threading.RLock()
    runtime.config = SimpleNamespace(max_time_s=300, max_new_tokens=192)
    runtime.device, runtime.dtype, runtime.provenance = "cpu", "bfloat16", {"model_id": "mock only"}
    runtime.torch = SimpleNamespace(no_grad=nullcontext)
    runtime.prepare_inputs = lambda messages, config: FakeInputs(input_ids=FakeTensor(3))
    runtime.model = SimpleNamespace(generate=lambda **kwargs: calls.append(kwargs) or FakeTensor(7),
                                   get_output_embeddings=lambda: SimpleNamespace(out_features=151936))
    runtime.processor = SimpleNamespace(tokenizer=FakeTokenizer(), batch_decode=lambda *a, **kw: ["raw model JSON"])
    return runtime, calls


def test_direct_free_generation_kwargs_and_provenance_remain_unconstrained(monkeypatch):
    runtime, calls = fake_runtime()
    monkeypatch.setattr(structured, "InterpretationConstraint", lambda *a: pytest.fail("Unexpected grammar setup"))
    ticks = iter([10, 11, 12])
    monkeypatch.setattr(streamed.time, "monotonic", lambda: next(ticks))
    response = runtime.generate([])
    assert set(calls[0]) == {"input_ids", "do_sample", "use_cache", "max_new_tokens", "max_time", "logits_to_keep"}
    assert {key: value for key, value in calls[0].items() if key != "input_ids"} == {
        "do_sample": False, "use_cache": True, "max_new_tokens": 192, "max_time": 299, "logits_to_keep": 1}
    assert "structured_decoding" not in response.details


def test_explicit_interpretation_builds_once_but_callback_and_parser_are_per_request(monkeypatch):
    runtime, calls = fake_runtime()
    built, callbacks = [], []
    class Constraint:
        def __init__(self, tokenizer, head): built.append((tokenizer, head))
        def for_request(self):
            callback = object()
            callbacks.append(callback)
            return callback, {"contract": "interpret_v1", "schema_sha256": "mock format identity"}
    monkeypatch.setattr(structured, "InterpretationConstraint", Constraint)
    first = runtime.generate([], output_contract="interpret_v1")
    second = runtime.generate([], output_contract="interpret_v1")
    assert len(built) == 1 and len(callbacks) == 2
    assert callbacks[0] is not callbacks[1]
    assert calls[0]["prefix_allowed_tokens_fn"] is callbacks[0]
    assert calls[1]["prefix_allowed_tokens_fn"] is callbacks[1]
    assert first.details["structured_decoding"]["contract"] == "interpret_v1"
    assert second.details["structured_decoding"] == first.details["structured_decoding"]


def test_empty_landmarks_use_fresh_regex_and_share_existing_tokenizer_trie(monkeypatch):
    runtime, generations = fake_runtime()
    tries, parsers = mock_integrations(monkeypatch)
    monkeypatch.setattr(structured, "new_empty_landmark_parser", object)
    interpretation = runtime.generate([], output_contract=structured.INTERPRET_CONTRACT)
    first = runtime.generate([], output_contract=structured.EMPTY_LANDMARK_CONTRACT)
    second = runtime.generate([], output_contract=structured.EMPTY_LANDMARK_CONTRACT)
    assert len(tries) == 1 and len({id(parser) for parser in parsers}) == 3
    assert all("prefix_allowed_tokens_fn" in arguments for arguments in generations)
    assert first.details["structured_decoding"] == second.details["structured_decoding"]
    assert first.details["structured_decoding"]["contract"] == structured.EMPTY_LANDMARK_CONTRACT
    assert first.details["structured_decoding"]["regex"] == structured.EMPTY_LANDMARK_PATTERN
    assert "schema_sha256" not in first.details["structured_decoding"]
    assert "regex" not in interpretation.details["structured_decoding"]


def test_grammar_setup_is_included_in_cooperative_deadline(monkeypatch):
    runtime, calls = fake_runtime()
    clock = [0]
    class Constraint:
        def __init__(self, *args): clock[0] += 301
        def for_request(self): return object(), {}
    monkeypatch.setattr(structured, "InterpretationConstraint", Constraint)
    monkeypatch.setattr(streamed.time, "monotonic", lambda: clock[0])
    with pytest.raises(TimeoutError, match="preparation"):
        runtime.generate([], output_contract="interpret_v1")
    assert calls == []


def test_missing_decoder_dependency_does_not_generate_unconstrained(monkeypatch):
    runtime, calls = fake_runtime()
    def missing(*args): raise ImportError("missing decoder")
    monkeypatch.setattr(structured, "InterpretationConstraint", missing)
    with pytest.raises(ImportError, match="decoder"):
        runtime.generate([], output_contract="interpret_v1")
    assert calls == []


def test_unknown_contract_does_not_generate(monkeypatch):
    runtime, calls = fake_runtime()
    with pytest.raises(ValueError, match="Unknown"):
        runtime.generate([], output_contract="destination_only")
    assert calls == []


@pytest.mark.parametrize("contract,expected", [(None, structured.INTERPRET_CONTRACT),
    (structured.LANDMARK_CONTRACT, None), (structured.EMPTY_LANDMARK_CONTRACT, structured.EMPTY_LANDMARK_CONTRACT)])
def test_application_worker_selects_the_requested_output_contract(monkeypatch, contract, expected):
    from dataclasses import asdict
    calls = []
    class Runtime:
        def __init__(self, config): pass
        def generate(self, messages, *, output_contract):
            calls.append(output_contract)
            return streamed.BackendResponse("raw", {"structured_decoding": {"contract": output_contract}})
        def close(self): calls.append("closed")
    class Connection:
        def __init__(self):
            command = {"kind": "generate", "messages": []}
            if contract is not None:
                command["contract"] = contract
            self.commands = iter([command, {"kind": "close"}])
            self.sent, self.closed = [], False
        def recv(self): return next(self.commands)
        def send(self, response): self.sent.append(response)
        def close(self): self.closed = True
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: None, cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(streamed, "checkpoint_identity", lambda *a, **kw: {})
    monkeypatch.setattr(streamed, "StreamedRuntime", Runtime)
    configuration = streamed.GroundingConfig(model_id=streamed.STREAMED_MODEL_ID,
        revision=streamed.STREAMED_REVISION, runtime="streamed", model_dir="fake-only",
        quantization="nf4", dtype="bfloat16")
    connection = Connection()
    streamed._streamed_worker(connection, asdict(configuration))
    assert calls == [expected, "closed"] and connection.closed
    assert connection.sent[0] == {"kind": "ready"}
    assert connection.sent[1]["details"]["structured_decoding"]["contract"] == expected
    assert connection.sent[1]["details"]["native_fault_isolation"] == "spawn_process"
