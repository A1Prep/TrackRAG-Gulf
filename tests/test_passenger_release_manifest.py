"""Release fixtures exercise labels, byte bindings and loading on a fresh checkout."""
from dataclasses import asdict
import json

import pytest

from tools import passenger_release as release
from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION, GroundingConfig


def put(path,value):
    path.write_bytes(json.dumps(value,sort_keys=True).encode())


@pytest.fixture
def bundle(tmp_path):
    folder=tmp_path/"experimental"
    adapter=folder/"adapter"
    adapter.mkdir(parents=True)
    put(adapter/"adapter_config.json",{"peft_type":"LORA","base_model_name_or_path":BASE_MODEL_ID,"revision":BASE_REVISION})
    (adapter/"adapter_model.safetensors").write_bytes(b"software-only release integrity fixture")
    put(adapter/"grounding_adapter_manifest.json",{"schema_version":1,"artifact_kind":"lora_adapter",
        "base_model_id":BASE_MODEL_ID,"base_revision":BASE_REVISION,"files":{
        name:release.file_hash(adapter/name) for name in ("adapter_config.json","adapter_model.safetensors")}})
    cfg=GroundingConfig(adapter_dir="adapter",runtime="streamed")
    put(folder/"grounding.json",asdict(cfg))
    put(folder/"geometry.json",{"format":"anchor-relative-regression-v1","vocabulary":[],
                                "coefficients":[[0.0]*6 for _ in range(5)],"training_count":0})
    manifest={"format":"passenger-app-handoff-v1","referent_policy":False,"verify_landmarks":False,
              "base_model":cfg.model_id,"revision":cfg.revision,"files":{
              p.relative_to(folder).as_posix():release.file_hash(p) for p in folder.rglob("*") if p.is_file()}}
    put(folder/"manifest.json",manifest)
    return folder,manifest


def test_independent_model_and_adapter_checks_on_cold_load(bundle):
    path,m=bundle
    assert release.verify_bundle(path)==m
    cfg=json.loads((path/"grounding.json").read_bytes())
    cfg["revision"]="a"*40
    put(path/"grounding.json",cfg)
    m["files"]["grounding.json"]=release.file_hash(path/"grounding.json")
    m["revision"]=cfg["revision"]
    put(path/"manifest.json",m)
    with pytest.raises(ValueError,match="different base"):
        release.verify_bundle(path)


@pytest.mark.parametrize("setting,value", [("referent_policy","false"),("verify_landmarks",1),
                                          ("base_model","unrelated-model"),("revision","wrong")])
def test_semantic_manifest_types_and_model_identity(bundle,setting,value):
    path,m=bundle
    m[setting]=value
    put(path/"manifest.json",m)
    with pytest.raises(ValueError):
        release.verify_bundle(path)


def test_disabled_ranker_cannot_disagree_with_inventory(bundle):
    path,m=bundle
    (path/"referent.json").write_text("{}")
    m["files"]["referent.json"]=release.file_hash(path/"referent.json")
    put(path/"manifest.json",m)
    with pytest.raises(ValueError,match="declared model components"):
        release.verify_bundle(path)


def test_published_bundle_is_bound_to_experiment_after_copy(bundle,tmp_path,monkeypatch):
    old,m=bundle
    monkeypatch.setattr(release,"ROOT",tmp_path)
    official=tmp_path/"models"/"passenger-feature"
    official.parent.mkdir()
    old.rename(official)
    protocol=tmp_path/"experiments"/"dropoff"/"passenger-release-protocol-20261122.json"
    protocol.parent.mkdir(parents=True)
    put(protocol,{"components":m})
    assert release.verify_bundle(official)==m
    m["verify_landmarks"]=True
    put(official/"manifest.json",m)
    with pytest.raises(ValueError,match="frozen release protocol"):
        release.verify_bundle(official)


def test_duplicate_json_fields_are_rejected(bundle):
    path,m=bundle
    text=(path/"manifest.json").read_text()
    (path/"manifest.json").write_text(text[:-1]+',"verify_landmarks":false}')
    with pytest.raises(ValueError,match="Duplicate"):
        release.verify_bundle(path)


def test_packaged_speech_chooses_installed_model_and_respects_explicit_modes(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACKRAG_PASSENGER_WHISPER", raising=False)
    vosk = tmp_path / release.ASR_ID
    whisper = tmp_path / "faster-whisper-small.en"
    vosk.mkdir()
    whisper.mkdir()
    monkeypatch.setattr(release, "find_spec", lambda name: None)
    assert release.speech_arguments([tmp_path]) == ["--asr-model", str(vosk)]
    monkeypatch.setattr(release, "find_spec", lambda name: object())
    assert release.speech_arguments([tmp_path]) == ["--whisper-model", str(whisper)]
    assert release.speech_arguments([tmp_path], text_only=True) == []
    monkeypatch.setenv("TRACKRAG_PASSENGER_WHISPER", str(whisper))
    assert release.speech_arguments([], vosk=vosk) == ["--asr-model", str(vosk)]
    assert release.speech_arguments([]) == ["--whisper-model", str(whisper)]
    with pytest.raises(ValueError):
        release.speech_arguments([], text_only=True, whisper=whisper)


def test_standard_launch_adds_installed_still_and_honors_explicit_banks(bundle, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from trackrag import passenger_app
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(passenger_app, "main", lambda argv: [path for prev, path in zip(argv, argv[1:]) if prev == "--requests"])
    normal = tmp_path/"runs/passenger-inputs/requests.jsonl"
    still = tmp_path/"runs/passenger-pair-scene/requests.jsonl"
    normal.parent.mkdir(parents=True)
    normal.touch()
    args = SimpleNamespace(bundle=bundle[0], base_dir=tmp_path, processor_cache=tmp_path, vision_cache=tmp_path,
                           text_only=True, asr_model=None, requests=None, qa_model="test", qa_data_root=None,
                           qa_predictions=None, qa_frame_rate=None, port=8765, out=tmp_path/"sessions")
    monkeypatch.setenv("GEMINI_MODEL", "test")
    assert release.start(args) == [str(normal)]
    still.parent.mkdir(parents=True)
    still.touch()
    assert release.start(args) == [str(normal), str(still)]
    photos = tmp_path/"data/uae-parking-study/passenger-requests.jsonl"
    photos.parent.mkdir(parents=True)
    photos.touch()
    assert release.start(args) == [str(normal), str(still), str(photos)]
    args.requests = [still]
    assert release.start(args) == [str(still)]
    args.require_footage = 2
    monkeypatch.setattr(passenger_app, "main", lambda argv: argv[argv.index("--require-footage") + 1])
    assert release.start(args) == "2"
    monkeypatch.setattr(passenger_app, "main", lambda argv: argv[argv.index("--between-road-mode") + 1])
    assert release.start(args) == "edge"
    args.between_road_mode = "edge-or-bay"
    assert release.start(args) == "edge-or-bay"
