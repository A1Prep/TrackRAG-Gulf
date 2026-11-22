"""Synthetic packaging evidence fixtures; no model execution or quality claims."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from PIL import Image
import pytest

from tools import dropoff_package_local as package
from tools.dropoff_bundle import bundle_artifacts, verify_bundle
from tools.dropoff_compare_local import compare
from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION, GroundingConfig
from trackrag.dropoff.streamed import METADATA_SHA256, WEIGHT_SHA256


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def blob(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    run, production, probe = (tmp_path / name for name in ("raw-run", "public-production", "probe-input"))
    for root in (run, production, probe): root.mkdir()
    Image.new("RGB", (100, 100), (10, 20, 30)).save(probe / "probe-image.png")
    image_sha = sha(probe / "probe-image.png")
    config = asdict(GroundingConfig(runtime="streamed", model_id=package.STREAMED_MODEL_ID,
        revision=package.STREAMED_REVISION, model_dir="local-base", cache_dir="local-processor-cache",
        quantization="nf4", dtype="bfloat16", device="cuda:0", min_pixels=65536, max_pixels=262144))
    source_hashes = {}
    for name in package.SOURCE_FILES:
        source_hashes[name] = sha(blob(run, "source/" + name, ("# synthetic captured source " + name + "\n").encode()))
    settings = {"stage": "public_destination_geometry", "epochs": 2, "gradient_accumulation": 1,
                "model_configuration": config, "source_sha256": source_hashes}
    put(run, "training_settings.json", settings)
    dataset = {"train_case_ids": ["train-a", "train-b"], "development_case_ids": ["dev-a", "dev-b"],
        "train_sha256": "a" * 64, "development_sha256": "d" * 64,
        "train_scene_groups": ["training-scene"], "development_scene_groups": ["development-scene"],
        "case_artifacts": [{"case_id": "train-a", "image_sha256": image_sha}]}
    put(run, "dataset_manifest.json", dataset)
    initial = {name: hashlib.sha256(("initial-" + name).encode()).hexdigest() for name in ("lora-a", "lora-b")}
    selected_hashes = {name: hashlib.sha256(("selected-" + name).encode()).hexdigest() for name in initial}
    put(run, "adapter/adapter_config.json", {"peft_type": "LORA", "base_model_name_or_path": config["model_id"],
        "revision": config["revision"], "r": 8})
    blob(run, "adapter/adapter_model.safetensors", b"SYNTHETIC_ADAPTER_NOT_A_MODEL")
    selected = {"schema_version": 1, "artifact_kind": "lora_adapter", "base_model_id": config["model_id"],
        "base_revision": config["revision"], "suffix_layers": 8,
        "image_processing": {"min_pixels": 65536, "max_pixels": 262144},
        "files": {name: sha(run / "adapter" / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
        "training_manifest_sha256": sha(run / "dataset_manifest.json"),
        "training_settings_sha256": sha(run / "training_settings.json"),
        "optimizer_steps": 2, "actual_training_examples_consumed": 2,
        "adapter_parameter_hashes": selected_hashes, "verification": {"fresh_base_reload_verified": False}}
    put(run, "adapter/grounding_adapter_manifest.json", selected)
    raw = {"status": "public_destination_finetuning_completed", "optimizer_steps": 4, "examples_consumed": 4,
        "epochs": 2, "epochs_completed": 2, "training_examples_unique": 2,
        "actual_training_case_ids": ["train-a", "train-b", "train-b", "train-a"],
        "selected_checkpoint": {"epoch": 1, "token_weighted_nll": 1.2},
        "adapter_hashes_initial": initial, "selected_adapter_parameter_hashes": selected_hashes,
        "selected_adapter_files": selected["files"], "fresh_reload_pending": True,
        "free_spatial_comparison_pending": True, "local_clarification_training_complete": False}
    put(run, "training_result.json", raw)
    trained_config = {**config, "adapter_dir": str((run / "adapter").resolve())}
    put(run, "inference_config.json", trained_config)
    protocol = put(tmp_path, "protocol.json", {"schema_version": 1, "kind": "public_same_image_language_contrast",
        "development_source_sha256": dataset["development_sha256"], "case_count": 2, "final_test": False,
        "groups": [{"image_sha256": "b" * 64, "case_ids": ["dev-a", "dev-b"],
                    "all_reference_masks_pairwise_disjoint": True}]})
    monkeypatch.setattr(package, "committed_protocol", lambda path, content:
        {"commit": "f" * 40, "path": "synthetic/frozen-protocol.json", "sha256": package.digest(content)})
    for prefix in ("", "contrast-"):
        for condition in ("pretrained", "trained"):
            name = prefix + condition
            rows = [{"case_id": identity, "ok": True, "point_in_mask": False, "coarse_box_iou": 0.1,
                "normalized_point_error": 0.2, "request_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256((identity + "prompt").encode()).hexdigest(),
                "scene_context_sha256": "c" * 64, "result": {"destination_point_xy": [10 + index * 2, 10]}}
                for index, identity in enumerate(dataset["development_case_ids"])]
            result_path = blob(run, "evaluation/" + name + ".jsonl",
                "".join(json.dumps(row) + "\n" for row in rows).encode())
            put(run, "evaluation/" + name + ".summary.json", {"results_sha256": sha(result_path), "requested_cases": 2,
                "case_ids": dataset["development_case_ids"], "development_source_sha256": dataset["development_sha256"],
                "runtime": config if condition == "pretrained" else trained_config,
                "contrast_protocol_sha256": sha(protocol) if prefix else None})
            if condition == "trained": put(run, "evaluation/" + name + ".adapter-hashes.json", selected_hashes)
        report = compare(run / f"evaluation/{prefix}pretrained.jsonl", run / f"evaluation/{prefix}trained.jsonl",
                         protocol=protocol if prefix else None)
        put(run, "evaluation/" + ("contrast-comparison.json" if prefix else "comparison.json"), report)
    request = {"request_id": "train-a", "text": "Drop me near the entrance", "input_mode": "typed", "dialogue": [],
        "scene": {"source": "predicted", "tracks": [], "provenance": {"mode": "image_only_public_training"},
            "frame": {"recording_id": "train-a", "frame_id": 0, "timestamp_s": 0.0,
                      "width": 100, "height": 100, "image_path": "probe-image.png"}}}
    put(probe, "reload_probe.json", {"schema_version": 1, "case_id": "train-a", "partition": "train",
                                     "request": request, "image_sha256": image_sha})
    actual = deepcopy(request)
    actual["scene"]["frame"]["image_path"] = str((probe / "probe-image.png").resolve())
    raw_text = json.dumps({"status": "proposed", "destination_point_xy": [500, 500],
                          "destination_box_xyxy": [400, 400, 600, 600], "clarification_question": None})
    result = {"status": "proposed", "destination_point_xy": [50, 50],
              "destination_box_xyxy": [40, 40, 60, 60], "clarification_question": None}
    provenance = {"backend_kind": "transformers_local_streamed", "injected": False, "requested_condition": "fine_tuned",
        "model_id": config["model_id"], "revision": config["revision"],
        "request_sha256": package.digest(package.canonical(actual)), "image": {"sha256": image_sha},
        "adapter": {"manifest_sha256": sha(run / "adapter/grounding_adapter_manifest.json"),
            "base_model_id": config["model_id"], "base_revision": config["revision"],
            "files": {name: {"sha256": value} for name, value in selected["files"].items()}},
        "backend_details": {"native_fault_isolation": "spawn_process", "worker_pid": 123,
            "runtime": "streamed", "dtype": "bfloat16", "device": "cuda:0",
            "model_id": config["model_id"], "revision": config["revision"],
            "processor_model_id": BASE_MODEL_ID, "processor_revision": BASE_REVISION,
            "suffix_layers": 8, "effective_resident_layers": 20, "image_processing": selected["image_processing"],
            "weight_identity": {"model_id": config["model_id"], "revision": config["revision"],
                "verified_weights": True, "verified_metadata": True,
                "files": {name: {"sha256": value} for name, value in {**WEIGHT_SHA256, **METADATA_SHA256}.items()}}}}
    receipt = {"request_id": "train-a", "request": actual, "status": "completed", "raw_text": raw_text,
               "result": result, "provenance": provenance}
    outcome = {"ok": True, "request_id": "train-a", "result": result, "provenance": provenance}
    for name, row in (("receipts.jsonl", receipt), ("model-outcomes.jsonl", outcome),
                      ("resolved-requests.jsonl", actual), ("input-snapshot.jsonl", actual)):
        blob(production, name, (json.dumps(row) + "\n").encode())
    summary = {"configuration": trained_config, "requested_rows": 1, "semantic_results": 1, "operational_errors": 0}
    for key, name in (("requests_sha256", "input-snapshot.jsonl"), ("resolved_requests_sha256", "resolved-requests.jsonl"),
                      ("outcomes_sha256", "model-outcomes.jsonl"), ("receipts_sha256", "receipts.jsonl")):
        summary[key] = sha(production / name)
    put(production, "run.json", summary)
    return {"run": run, "production": production, "probe": probe, "protocol": protocol, "out": tmp_path / "staged"}


def stage(evidence):
    return package.stage_run(evidence["run"], evidence["production"], evidence["probe"], evidence["protocol"], evidence["out"])


def bind_encoding_fixture(evidence, receipt):
    """Rebind synthetic proof files so corpus tests cannot fail on stale hashes."""
    root=evidence['run'];put(root,'target-encoding.json',receipt)
    binding=sha(root/'target-encoding.json')
    settings=json.loads((root/'training_settings.json').read_text())
    settings.update(target_encoding=receipt['encoding'],target_encoding_receipt_sha256=binding)
    put(root,'training_settings.json',settings)
    selected=json.loads((root/'adapter/grounding_adapter_manifest.json').read_text())
    selected.update(training_manifest_sha256=sha(root/'dataset_manifest.json'),
        training_settings_sha256=sha(root/'training_settings.json'),target_encoding=receipt['encoding'],
        target_encoding_receipt_sha256=binding)
    put(root,'adapter/grounding_adapter_manifest.json',selected)
    raw=json.loads((root/'training_result.json').read_text())
    raw.update(target_encoding=receipt['encoding'],target_encoding_receipt_sha256=binding)
    put(root,'training_result.json',raw)
    production=evidence['production']
    for name in ('receipts.jsonl','model-outcomes.jsonl'):
        value=json.loads((production/name).read_text())
        value['provenance']['adapter']['manifest_sha256']=sha(root/'adapter/grounding_adapter_manifest.json')
        put(production,name,value)
    summary=json.loads((production/'run.json').read_text())
    summary.update(receipts_sha256=sha(production/'receipts.jsonl'),outcomes_sha256=sha(production/'model-outcomes.jsonl'))
    put(production,'run.json',summary)


@pytest.fixture
def encoded_evidence(evidence):
    root=evidence['run'];dataset=json.loads((root/'dataset_manifest.json').read_text())
    probe_image=dataset['case_artifacts'][0]['image_sha256']
    identities=dataset['train_case_ids']+dataset['development_case_ids']
    dataset['case_artifacts']=[{'case_id':identity,'image_sha256':probe_image if identity=='train-a'
        else hashlib.sha256(('synthetic-image-'+identity).encode()).hexdigest()} for identity in identities]
    put(root,'dataset_manifest.json',dataset)
    receipt={'schema_version':1,'kind':'teacher_target_encoding','encoding':'point_first_mask_preserving_integer_v1',
        'source_bindings':{key:dataset[key] for key in ('train_sha256','development_sha256')},
        'case_count':len(identities),'cases':{}}
    for row in dataset['case_artifacts']:
        identity=row['case_id']
        receipt['cases'][identity]={'partition':'train' if identity in dataset['train_case_ids'] else 'development',
            'image_sha256':row['image_sha256'],**{key:hashlib.sha256((identity+':'+key).encode()).hexdigest()
                for key in ('original_supervision_sha256','input_sha256','transformed_target_sha256','mask_sha256')}}
    bind_encoding_fixture(evidence,receipt)
    return evidence


def test_stages_declared_encoding_with_consistent_split_case_and_image_bindings(encoded_evidence):
    root=encoded_evidence['run']
    before={str(path):path.read_bytes() for path in root.rglob('*') if path.is_file()}
    stage(encoded_evidence)
    target=encoded_evidence['out']
    assert (target/'target-encoding.json').read_bytes()==(root/'target-encoding.json').read_bytes()
    manifest=json.loads((target/'adapter/grounding_adapter_manifest.json').read_text())
    assert manifest['target_encoding_receipt_sha256']==sha(target/'target-encoding.json')
    assert all(path.read_bytes()==before[str(path)] for path in root.rglob('*') if path.is_file())


@pytest.mark.parametrize('mutation,match',[
    ('source_hash','source split hashes'),('missing_case','receipt cases/count'),
    ('extra_case','receipt cases/count'),('wrong_count','receipt cases/count'),
    ('duplicate_split_id','split IDs are duplicate'),('overlapping_split_id','split IDs are duplicate'),
    ('duplicate_artifact','case artifacts are duplicate'),('missing_artifact','case artifacts are duplicate'),
    ('extra_artifact','case artifacts are duplicate'),('wrong_partition','wrong case partition'),
    ('wrong_image','image binding differs'),
])
def test_rejects_rehashed_receipt_for_wrong_corpus(encoded_evidence,mutation,match):
    evidence=encoded_evidence;root=evidence['run']
    receipt=json.loads((root/'target-encoding.json').read_text())
    dataset=json.loads((root/'dataset_manifest.json').read_text())
    if mutation=='source_hash':receipt['source_bindings']['train_sha256']=dataset['development_sha256']
    elif mutation=='missing_case':receipt['cases'].pop('dev-b')
    elif mutation=='extra_case':receipt['cases']['extra-case']=deepcopy(receipt['cases']['dev-b'])
    elif mutation=='wrong_count':receipt['case_count']+=1
    elif mutation=='duplicate_split_id':dataset['development_case_ids'].append('dev-a')
    elif mutation=='overlapping_split_id':dataset['development_case_ids'][0]='train-a'
    elif mutation=='duplicate_artifact':dataset['case_artifacts'].append(deepcopy(dataset['case_artifacts'][0]))
    elif mutation=='missing_artifact':dataset['case_artifacts'].pop()
    elif mutation=='extra_artifact':dataset['case_artifacts'].append({'case_id':'extra-case','image_sha256':'e'*64})
    elif mutation=='wrong_partition':receipt['cases']['dev-a']['partition']='train'
    else:receipt['cases']['dev-a']['image_sha256']='f'*64
    put(root,'dataset_manifest.json',dataset)
    bind_encoding_fixture(evidence,receipt)
    with pytest.raises(package.PackageError,match=match):stage(evidence)
    assert not evidence['out'].exists()


@pytest.mark.parametrize('field',['original_supervision_sha256','input_sha256','transformed_target_sha256','mask_sha256'])
@pytest.mark.parametrize('invalid',[None,'short','G'*64,True])
def test_integer_receipt_requires_well_formed_label_input_and_mask_hashes(encoded_evidence,field,invalid):
    evidence=encoded_evidence;root=evidence['run']
    receipt=json.loads((root/'target-encoding.json').read_text())
    receipt['cases']['dev-a'][field]=invalid
    bind_encoding_fixture(evidence,receipt)
    with pytest.raises(package.PackageError,match='Invalid integer target-encoding'):
        stage(evidence)
    assert not evidence['out'].exists()


def test_stages_selected_checkpoint_separately_from_complete_schedule_without_touching_raw(evidence):
    raw_before = {str(path): path.read_bytes() for path in evidence["run"].rglob("*") if path.is_file()}
    result = stage(evidence)
    staged = evidence["out"]
    training = json.loads((staged / "training_result.json").read_text())
    manifest = json.loads((staged / "adapter/grounding_adapter_manifest.json").read_text())
    assert result["full_schedule"]["optimizer_steps"] == training["optimizer_steps"] == 4
    assert training["counter_scope"] == "entire_completed_training_schedule"
    assert training["packaged_checkpoint"]["optimizer_steps"] == manifest["optimizer_steps"] == 2
    assert len(training["actual_training_case_ids"]) == 4
    assert all(path.read_bytes() == raw_before[str(path)] for path in evidence["run"].rglob("*") if path.is_file())
    inference = json.loads((staged / "inference_config.json").read_text())
    assert inference["adapter_dir"] == "adapter"
    assert inference["model_dir"] == "models/qwen3-vl-8b-local-nf4"
    with zipfile.ZipFile(staged / "training_source.zip") as archive:
        assert set(archive.namelist()) == package.SOURCE_FILES
    source_manifest = json.loads((staged / "training_source_manifest.json").read_text())
    assert source_manifest["purpose"] == "training_reproduction_only_in_separate_copy"
    assert source_manifest["production_overlay_allowed"] is False
    archive = staged.parent / "adapter.zip"
    bundle_artifacts(staged, archive)
    assert verify_bundle(archive)["metadata"]["model_quality_verified"] is False
    extracted = staged.parent / "extracted"
    with zipfile.ZipFile(archive) as source:
        source.extractall(extracted)
    from tools.dropoff_predict import read_requests
    portable = json.loads((extracted / "probe-requests.jsonl").read_text())
    assert portable["scene"]["frame"]["image_path"] == "probe-image.png"
    assert "target" not in portable and "image_sha256" not in portable
    loaded = read_requests(extracted / "probe-requests.jsonl")
    assert loaded[0]["scene"]["frame"]["image_path"] == str((extracted / "probe-image.png").resolve())


def test_rejects_optional_encoding_receipt_binding_mismatch(evidence):
    root=evidence['run']
    put(root,'target-encoding.json',{'encoding':'synthetic_teacher_codec'})
    settings=json.loads((root/'training_settings.json').read_text())
    settings.update(target_encoding='synthetic_teacher_codec',target_encoding_receipt_sha256=sha(root/'target-encoding.json'))
    put(root,'training_settings.json',settings)
    manifest=json.loads((root/'adapter/grounding_adapter_manifest.json').read_text())
    manifest.update(training_settings_sha256=sha(root/'training_settings.json'),target_encoding='synthetic_teacher_codec',
                    target_encoding_receipt_sha256='0'*64)
    put(root,'adapter/grounding_adapter_manifest.json',manifest)
    with pytest.raises(package.PackageError,match='target-encoding receipt/settings/adapter'):
        stage(evidence)
    assert not evidence['out'].exists()


@pytest.mark.parametrize('missing',['sha_field','empty_sha','receipt_file','all_settings_fields'])
def test_declared_integer_encoding_cannot_skip_required_receipt(evidence,missing):
    root=evidence['run'];encoding='point_first_mask_preserving_integer_v1'
    put(root,'target-encoding.json',{'encoding':encoding})
    binding=sha(root/'target-encoding.json')
    settings=json.loads((root/'training_settings.json').read_text())
    settings.update(target_encoding=encoding,target_encoding_receipt_sha256=binding)
    if missing=='sha_field':settings.pop('target_encoding_receipt_sha256')
    elif missing=='empty_sha':settings['target_encoding_receipt_sha256']=''
    elif missing=='all_settings_fields':
        settings.pop('target_encoding');settings.pop('target_encoding_receipt_sha256')
    else:(root/'target-encoding.json').unlink()
    put(root,'training_settings.json',settings)
    manifest=json.loads((root/'adapter/grounding_adapter_manifest.json').read_text())
    manifest.update(training_settings_sha256=sha(root/'training_settings.json'),target_encoding=encoding,
                    target_encoding_receipt_sha256=binding)
    put(root,'adapter/grounding_adapter_manifest.json',manifest)
    result=json.loads((root/'training_result.json').read_text())
    result.update(target_encoding=encoding,target_encoding_receipt_sha256=binding)
    put(root,'training_result.json',result)
    with pytest.raises(package.PackageError,match='Declared target encoding requires'):
        stage(evidence)
    assert not evidence['out'].exists()


@pytest.mark.parametrize("mutation,match", [
    ("unfinished", "has not completed"), ("selected_tensors", "selected tensor hashes"),
    ("fresh_tensors", "Fresh evaluation tensor hashes"), ("adapter_bytes", "adapter file changed"),
    ("source_bytes", "Captured training source hash"), ("stale_comparison", "Saved comparison differs"),
    ("inference_profile", "Saved inference configuration"),
])
def test_rejects_incomplete_or_stale_selected_evidence(evidence, mutation, match):
    root = evidence["run"]
    if mutation in ("unfinished", "selected_tensors"):
        path = root / "training_result.json"
        value = json.loads(path.read_text())
        if mutation == "unfinished": value["status"] = "training"
        else: value["selected_adapter_parameter_hashes"]["lora-a"] = "0" * 64
        put(root, path.name, value)
    elif mutation == "fresh_tensors":
        value = json.loads((root / "evaluation/trained.adapter-hashes.json").read_text())
        value["lora-a"] = "0" * 64
        put(root, "evaluation/trained.adapter-hashes.json", value)
    elif mutation == "adapter_bytes": blob(root, "adapter/adapter_model.safetensors", b"changed")
    elif mutation == "source_bytes": blob(root, "source/tools/dropoff_train.py", b"changed")
    elif mutation == "stale_comparison":
        value = json.loads((root / "evaluation/comparison.json").read_text())
        value["trained"]["coverage"] = 0
        put(root, "evaluation/comparison.json", value)
    else:
        value = json.loads((root / "inference_config.json").read_text())
        value["runtime"] = "resident"
        put(root, "inference_config.json", value)
    with pytest.raises(package.PackageError, match=match): stage(evidence)
    assert not evidence["out"].exists()


@pytest.mark.parametrize("mutation,match", [
    ("manifest", "different selected manifest"), ("injected", "genuine fine-tuned"),
    ("worker", "native worker"), ("partial_result", "required interpretation fields"),
])
def test_rejects_stale_or_substituted_production_probe(evidence, mutation, match):
    root = evidence["production"]
    receipt = json.loads((root / "receipts.jsonl").read_text())
    if mutation == "manifest": receipt["provenance"]["adapter"]["manifest_sha256"] = "0" * 64
    elif mutation == "injected": receipt["provenance"]["injected"] = True
    elif mutation == "worker": receipt["provenance"]["backend_details"]["native_fault_isolation"] = "fixture"
    else: receipt["raw_text"] = '{"destination_point_xy":[500,500],"destination_box_xyxy":[400,400,600,600]}'
    put(root, "receipts.jsonl", receipt)
    outcome = json.loads((root / "model-outcomes.jsonl").read_text())
    outcome["provenance"] = receipt["provenance"]
    put(root, "model-outcomes.jsonl", outcome)
    summary = json.loads((root / "run.json").read_text())
    summary.update(receipts_sha256=sha(root / "receipts.jsonl"), outcomes_sha256=sha(root / "model-outcomes.jsonl"))
    put(root, "run.json", summary)
    with pytest.raises(ValueError, match=match): stage(evidence)
    assert not evidence["out"].exists()


@pytest.mark.parametrize("field,value", [("max_new_tokens", 1), ("max_time_s", 0.001)])
def test_rejects_saved_generation_budget_that_differs_from_successful_probe(evidence, field, value):
    root = evidence["run"]
    inference = json.loads((root / "inference_config.json").read_text())
    inference[field] = value
    put(root, "inference_config.json", inference)
    with pytest.raises(package.PackageError, match="Saved/production inference profile differs"):
        stage(evidence)
    assert not evidence["out"].exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for committed-protocol verification")
def test_committed_protocol_checks_exact_git_blob_and_rejects_unstaged_or_crlf_change(tmp_path):
    root = tmp_path / "protocol-repository"
    root.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    git("init", "--quiet")
    git("config", "user.name", "Protocol Test Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "core.autocrlf", "true")
    hooks = root / "empty-hooks"
    hooks.mkdir()
    git("config", "core.hooksPath", str(hooks))
    (root / ".gitattributes").write_bytes(b"protocol.json text eol=lf\n")
    path = root / "protocol.json"
    frozen = b'{"frozen":true}\n'
    path.write_bytes(frozen)
    git("add", ".gitattributes", "protocol.json")
    git("-c", "commit.gpgSign=false", "commit", "--quiet", "-m", "Freeze synthetic protocol")
    proof = package.committed_protocol(path, path.read_bytes())
    assert proof["sha256"] == hashlib.sha256(frozen).hexdigest()
    assert proof["commit"] == git("rev-parse", "HEAD").stdout.decode().strip()
    assert proof["path"] == "protocol.json"
    for changed in (b'{"frozen":false}\n', frozen.replace(b"\n", b"\r\n")):
        path.write_bytes(changed)
        with pytest.raises(package.PackageError, match="differs from its committed bytes"):
            package.committed_protocol(path, path.read_bytes())
