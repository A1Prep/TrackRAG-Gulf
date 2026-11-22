"""Generated artifact boundary tests; these are not real detector, ASR or Qwen runs."""

from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
import pytest

from tools import task4_check_integration as preflight
from trackrag.dropoff.model import ADAPTER_MANIFEST, BASE_MODEL_ID, BASE_REVISION
from trackrag.vision.config import config_from_dict


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    data = tmp_path / "data"
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    vision_data = {"name": "software-test-only", "detector": {
        "label_space": "emt", "checkpoint": "fixture.pth", "checkpoint_sha256": "a" * 64}}
    vision = config_from_dict(vision_data)
    vision_path = tmp_path / "vision.json"
    write_json(vision_path, vision_data)
    runs = {}
    for rid in ("recording_a", "recording_b"):
        images = data / "images" / "train" / "images" / rid
        images.mkdir(parents=True)
        for frame in (1, 4, 10):
            Image.new("RGB", (320, 180), (frame, 80, 100)).save(images / f"{frame:05d}.png")
        # Keep frame 4 empty: input preparation must not reuse another frame's object.
        rows = [{"recording_id": rid, "frame_id": frame, "track_id": 7, "class_name": "Car",
                 "source": "predicted", "bbox_xyxy": [100, 60, 160, 120]} for frame in (1, 10)]
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (predictions / f"{rid}.tracks.jsonl").write_bytes(payload)
        provenance = {"recording_id": rid, "identity_hash": vision.identity_hash(), "frame_ids": [1, 4, 10],
                      "frames": 3, "first_frame": 1, "last_frame": 10, "first_frame_size": [320, 180],
                      "observations": len(rows), "tracks_sha256": sha256(payload).hexdigest()}
        provenance_path = predictions / f"{rid}.provenance.json"
        write_json(provenance_path, provenance)
        runs[rid] = {"tracks_sha256": sha256(payload).hexdigest(),
                     "provenance_sha256": sha256(provenance_path.read_bytes()).hexdigest()}
    reviewed = tmp_path / "reviewed-manifest.json"
    write_json(reviewed, {"runs": {vision.identity_hash(): runs}})
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_json(adapter / "adapter_config.json", {"peft_type": "LORA", "base_model_name_or_path": BASE_MODEL_ID})
    (adapter / "adapter_model.safetensors").write_bytes(b"hash-only-software-fixture")
    write_json(adapter / ADAPTER_MANIFEST, {
        "schema_version": 1, "artifact_kind": "lora_adapter", "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION, "files": {
            name: sha256((adapter / name).read_bytes()).hexdigest()
            for name in ("adapter_config.json", "adapter_model.safetensors")},
    })
    cache = tmp_path / "cache"
    snapshot = cache / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / BASE_REVISION
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
        write_json(snapshot / name, {})
    (snapshot / "model.safetensors").write_bytes(b"accessibility-only-software-fixture")
    grounding = tmp_path / "grounding.json"
    write_json(grounding, {"adapter_dir": str(adapter), "cache_dir": str(cache), "device": "cpu"})
    speech = tmp_path / "vosk"
    speech.mkdir()
    (speech / "fixture.dat").write_bytes(b"software-test-only")
    digest = sha256(f"fixture.dat\0{sha256(b'software-test-only').hexdigest()}\n".encode()).hexdigest()
    from trackrag import local_speech
    monkeypatch.setattr(local_speech, "MODEL_CONTENT_SHA256", digest)
    monkeypatch.setattr(preflight, "_dependency", lambda d, m, expected=None: {
        "distribution": d, "module": m, "version": expected or "software-metadata-fixture",
        "module_imported": False})
    return {"data_root": data, "vision_config": vision_path, "predictions": predictions,
            "grounding_config": grounding, "scenes": [("recording_a", 4), ("recording_b", 10)],
            "asr_model": speech, "prediction_manifest": reviewed}


def indexed(report):
    return {row["check"]: row for row in report["checks"]}


def test_preflight_uses_strict_saved_route_and_never_initializes_models(artifacts, monkeypatch):
    from trackrag.dropoff import app, model
    from trackrag import local_speech, whisper_speech

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not construct a runtime or recognizer")
    monkeypatch.setattr(app, "LocalApplication", forbidden)
    monkeypatch.setattr(model, "GroundingModel", forbidden)
    monkeypatch.setattr(local_speech, "LocalSpeechRecognizer", forbidden)
    monkeypatch.setattr(whisper_speech, "WhisperSpeechRecognizer", forbidden)
    before = {path: path.read_bytes() for path in artifacts["data_root"].parent.rglob("*") if path.is_file()}
    report = preflight.check_integration(**artifacts)
    rows = indexed(report)
    assert report["ready_for_runtime_attempt"] is True
    assert rows["scene:1"]["details"]["current_observations"] == 0
    assert rows["scene:2"]["details"]["current_observations"] == 1
    for name in ("scene:1", "scene:2"):
        details = rows[name]["details"]
        assert details["prediction_receipt"]["kind"] == "validated_saved_vision_predictions"
        assert details["reviewed_prediction_manifest_matched"] is True
        assert details["prediction_receipt"]["prefix_last_frame"] == details["frame"]["frame_id"]
    assert rows["trained_adapter"]["status"] == "passed"
    assert all(value is False or value is None for value in report["execution"].values())
    assert {path: path.read_bytes() for path in artifacts["data_root"].parent.rglob("*") if path.is_file()} == before
    json.dumps(report, allow_nan=False)


def test_each_scene_and_other_components_survive_prediction_failure(artifacts):
    (artifacts["predictions"] / "recording_a.tracks.jsonl").write_bytes(b"tampered")
    report = preflight.check_integration(**artifacts)
    rows = indexed(report)
    assert report["ready_for_runtime_attempt"] is False
    assert rows["scene:1"]["status"] == "error"
    assert "sha256" in rows["scene:1"]["error"]["message"]
    assert rows["scene:2"]["status"] == rows["trained_adapter"]["status"] == rows["vosk_model"]["status"] == "passed"


def test_model_and_speech_failures_are_retained_without_hiding_scene_success(artifacts):
    config = json.loads(artifacts["grounding_config"].read_text())
    Path(config["adapter_dir"], "adapter_model.safetensors").write_bytes(b"changed")
    artifacts["asr_model"] = artifacts["asr_model"] / "missing"
    report = preflight.check_integration(**artifacts)
    rows = indexed(report)
    assert rows["trained_adapter"]["status"] == rows["vosk_model"]["status"] == "error"
    assert rows["base_checkpoint_accessibility"]["status"] == rows["scene:1"]["status"] == rows["scene:2"]["status"] == "passed"


def test_pretrained_configuration_cannot_pass_the_trained_milestone(artifacts):
    config = json.loads(artifacts["grounding_config"].read_text())
    config.pop("adapter_dir")
    write_json(artifacts["grounding_config"], config)
    rows = indexed(preflight.check_integration(**artifacts))
    assert rows["trained_adapter"]["status"] == "error"
    assert "pretrained only" in rows["trained_adapter"]["error"]["message"]
    assert rows["base_checkpoint_accessibility"]["status"] == "passed"


def test_invalid_config_keeps_blocked_dependent_checks_and_checks_other_modules(artifacts):
    artifacts["vision_config"].write_text('{"name":"a","name":"b"}')
    artifacts["grounding_config"].write_text('{"revision":"main"}')
    rows = indexed(preflight.check_integration(**artifacts))
    assert rows["vision_configuration"]["status"] == rows["grounding_configuration"]["status"] == "error"
    for name in ("scene:1", "scene:2", "trained_adapter", "base_checkpoint_accessibility", "processor_accessibility"):
        assert rows[name]["status"] == "blocked"
    assert rows["vosk_model"]["status"] == rows["dependency:torch"]["status"] == "passed"


def test_two_distinct_recordings_required_but_supplied_scene_still_checked(artifacts):
    artifacts["scenes"] = [("recording_a", 4), ("recording_a", 10)]
    rows = indexed(preflight.check_integration(**artifacts))
    assert rows["recording_cohort"]["status"] == "error"
    assert rows["scene:1"]["status"] == rows["scene:2"]["status"] == "passed"


def test_manifest_mismatch_is_not_treated_as_proof_of_genuine_predictions(artifacts):
    manifest = json.loads(artifacts["prediction_manifest"].read_text())
    profile = next(iter(manifest["runs"].values()))
    profile["recording_a"]["provenance_sha256"] = "0" * 64
    write_json(artifacts["prediction_manifest"], manifest)
    rows = indexed(preflight.check_integration(**artifacts))
    assert rows["scene:1"]["status"] == "error"
    assert "reviewed run manifest" in rows["scene:1"]["error"]["message"]
    assert rows["scene:2"]["status"] == "passed"


def test_base_index_cannot_follow_a_shard_outside_the_selected_snapshot(artifacts):
    config = json.loads(artifacts["grounding_config"].read_text())
    snapshot = Path(config["cache_dir"]) / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / BASE_REVISION
    write_json(snapshot / "model.safetensors.index.json", {"weight_map": {"parameter": "../../outside.safetensors"}})
    rows = indexed(preflight.check_integration(**artifacts))
    assert rows["base_checkpoint_accessibility"]["status"] == "error"
    assert "nonlocal shard" in rows["base_checkpoint_accessibility"]["error"]["message"]
    assert rows["processor_accessibility"]["status"] == rows["trained_adapter"]["status"] == "passed"


def streamed_artifacts(artifacts, monkeypatch):
    from trackrag.dropoff import streamed, structured
    config = json.loads(artifacts["grounding_config"].read_text())
    config.update(runtime="streamed", model_id=streamed.STREAMED_MODEL_ID, revision=streamed.STREAMED_REVISION,
                  model_dir=str(artifacts["data_root"].parent / "streamed"), device="cuda", quantization="nf4", dtype="bfloat16")
    folder = Path(config["model_dir"])
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"software-test-only")
    write_json(artifacts["grounding_config"], config)
    adapter = Path(config["adapter_dir"])
    write_json(adapter / "adapter_config.json", {"peft_type": "LORA", "base_model_name_or_path": config["model_id"]})
    manifest = json.loads((adapter / ADAPTER_MANIFEST).read_text())
    manifest.update(base_model_id=config["model_id"], base_revision=config["revision"])
    manifest["files"]["adapter_config.json"] = sha256((adapter / "adapter_config.json").read_bytes()).hexdigest()
    write_json(adapter / ADAPTER_MANIFEST, manifest)
    calls = []

    def verify(directory, *, verify_weights):
        calls.append((directory, verify_weights))
        return {"path": directory, "verified_weights": verify_weights,
                "files": {"model.safetensors": {"bytes": 18}}}

    monkeypatch.setattr(streamed, "checkpoint_identity", verify)
    original_source_identity = structured._dependency_identity
    monkeypatch.setattr(structured, "_dependency_identity", lambda: {"source_url": structured.LMFE_SOURCE_URL,
                                                                  "upstream_commit": structured.LMFE_COMMIT})
    return folder, calls, original_source_identity


def test_streamed_runtime_uses_existing_verifier_and_exact_dependency_pins(artifacts, monkeypatch):
    folder, calls, _ = streamed_artifacts(artifacts, monkeypatch)
    probes = []
    monkeypatch.setattr(preflight, "_dependency", lambda d, m, expected=None:
                        probes.append((d, expected)) or {"version": expected or "metadata-fixture"})
    rows = indexed(preflight.check_integration(**artifacts))
    assert calls == [(str(folder), False)]
    assert rows["base_checkpoint_accessibility"]["details"]["verified_weights"] is False
    assert rows["processor_accessibility"]["status"] == "passed"
    assert rows["structured_decoder_source"]["status"] == "passed"
    assert ("transformers", "5.1.0") in probes and ("torch", "2.6.0") in probes
    assert ("interegular", "0.3.3") in probes and ("huggingface-hub", "1.33.0") in probes
    assert rows["streamed_configuration"]["status"] == "passed"


@pytest.mark.parametrize("source", [None, {"url": "https://example.invalid/unpinned.zip"}])
def test_missing_or_wrong_structured_decoder_source_blocks_streamed_preflight(artifacts, monkeypatch, source):
    from types import SimpleNamespace
    from trackrag.dropoff import structured
    _, _, original_identity = streamed_artifacts(artifacts, monkeypatch)
    monkeypatch.setattr(structured, "_dependency_identity", original_identity)
    monkeypatch.setattr(structured.metadata, "distribution", lambda name:
                        SimpleNamespace(version="0.11.3", read_text=lambda filename: json.dumps(source)))
    report = preflight.check_integration(**artifacts)
    rows = indexed(report)
    assert report["ready_for_runtime_attempt"] is False
    assert rows["structured_decoder_source"]["status"] == "error"
    assert "pinned lm-format-enforcer source" in rows["structured_decoder_source"]["error"]["message"]
    assert rows["dependency:interegular"]["status"] == rows["trained_adapter"]["status"] == "passed"


def test_wrong_streamed_package_version_does_not_hide_other_checks(artifacts, monkeypatch):
    streamed_artifacts(artifacts, monkeypatch)
    original_probe = preflight._dependency

    def probe(distribution, module, expected=None):
        if distribution == "transformers":
            raise ValueError(f"{distribution} must be {expected}; installed 4.50.0")
        return original_probe(distribution, module, expected)
    monkeypatch.setattr(preflight, "_dependency", probe)
    report = preflight.check_integration(**artifacts)
    rows = indexed(report)
    assert report["ready_for_runtime_attempt"] is False
    assert rows["streamed_dependency:transformers"]["status"] == "error"
    assert rows["structured_decoder_source"]["status"] == rows["dependency:interegular"]["status"] == rows["vosk_model"]["status"] == "passed"


def test_ambiguous_speech_selection_retains_both_supplied_model_checks(artifacts):
    artifacts["whisper_model"] = artifacts["asr_model"] / "missing-whisper"
    rows = indexed(preflight.check_integration(**artifacts))
    assert rows["speech_selection"]["status"] == rows["whisper_model"]["status"] == "error"
    assert rows["vosk_model"]["status"] == "passed"
    assert "dependency:faster_whisper" in rows and "dependency:ctranslate2" in rows


def test_dependency_probe_checks_metadata_and_spec_without_importing_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(preflight.metadata, "version", lambda name: calls.append(("version", name)) or "1.2.1")
    monkeypatch.setattr(preflight.util, "find_spec", lambda name: calls.append(("spec", name)) or object())
    assert preflight._dependency("faster-whisper", "faster_whisper", "1.2.1")["module_imported"] is False
    assert calls == [("version", "faster-whisper"), ("spec", "faster_whisper")]
    with pytest.raises(ValueError, match="must be"):
        preflight._dependency("faster-whisper", "faster_whisper", "wrong")


def test_cli_outputs_json_with_all_failed_checks_and_nonzero_exit(tmp_path, capsys):
    missing = str(tmp_path / "missing")
    result = preflight.main(["--data-root", missing, "--vision-config", missing,
                             "--predictions", missing, "--grounding-config", missing,
                             "--scene", "recording_a:4", "--scene", "recording_b:10",
                             "--asr-model", missing])
    report = json.loads(capsys.readouterr().out)
    assert result == 1
    rows = indexed(report)
    assert rows["vision_configuration"]["status"] == rows["grounding_configuration"]["status"] == "error"
    assert "scene:1" in rows and "scene:2" in rows and "vosk_model" in rows


@pytest.mark.parametrize("value", ["../outside:1", "recording_a:-1", "recording_a:many", "missing-frame"])
def test_scene_arguments_cannot_select_paths_or_non_original_frame_ids(value):
    with pytest.raises(Exception, match="scene must be"):
        preflight._scene(value)
