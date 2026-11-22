"""Generated source/artifact fixtures; no cloud GPU/model execution."""

from copy import deepcopy
import ast
import hashlib
import io
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import warnings
import zipfile

import pytest

from tools.dropoff_bundle import (
    BundleError, MANIFEST, REQUIRED_SOURCE, TEST_SUPPORT_NOTEBOOK, _regular_bytes,
    bundle_artifacts, bundle_source, source_allowed, verify_bundle,
)


def source_fixture(tmp_path):
    root = tmp_path / "repo"
    for name in REQUIRED_SOURCE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# generated software fixture\n", encoding="utf-8")
    for name in (
        ".env", ".venv-models/Lib/private.py", "data/private.json", "models/base.safetensors",
        "runs/private.json", "trackrag/answering.py", "tools/unrelated.py",
        "trackrag/dropoff/.venv/private.py", "trackrag/dropoff/models/cache.py",
        "trackrag/dropoff/.secret.py", "configs/dropoff/.env.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("EXCLUDED_SECRET_FIXTURE", encoding="utf-8")
    return root


def rewritten_zip(path, *, replace=None, extra=None, duplicate=None, symlink=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(path) as original, zipfile.ZipFile(buffer, "w") as output:
        for entry in original.infolist():
            content = original.read(entry.filename)
            if replace and entry.filename in replace:
                content = replace[entry.filename]
            output.writestr(entry, content)
        if extra:
            for name, content in extra.items():
                output.writestr(name, content)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                output.writestr(duplicate, b"duplicate")
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            output.writestr(info, b"/outside")
    return buffer.getvalue()


def test_minimal_source_bundle_omits_secrets_venvs_models_and_runs(tmp_path):
    root = source_fixture(tmp_path)
    output = tmp_path / "source.zip"
    result = bundle_source(root, output)
    assert result["verified"] is True
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == REQUIRED_SOURCE | {MANIFEST}
        manifest = json.loads(archive.read(MANIFEST))
        assert manifest["metadata"]["actual_8b_gpu_execution_verified"] is False
        for name in REQUIRED_SOURCE:
            assert manifest["files"][name]["sha256"] == hashlib.sha256(archive.read(name)).hexdigest()
    assert verify_bundle(output)["files"] == len(REQUIRED_SOURCE)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        bundle_source(root, output)
    assert output.read_bytes() == before


def test_optional_docs_and_tests_are_explicit(tmp_path):
    root = source_fixture(tmp_path)
    for name in ("docs/dropoff-cloud-quickstart.md", "tests/test_dropoff_model.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated fixture")
    output = tmp_path / "source.zip"
    bundle_source(root, output, include_tests=True)
    with zipfile.ZipFile(output) as archive:
        assert "docs/dropoff-cloud-quickstart.md" in archive.namelist()
        assert "tests/test_dropoff_model.py" in archive.namelist()


def support_notebook():
    return {"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{
        "id":"generated-test-support","cell_type":"code","metadata":{},
        "source":["value = 1\n"],"execution_count":None,"outputs":[]}]}


def test_exact_unexecuted_notebook_is_optional_test_support_only(tmp_path):
    root=source_fixture(tmp_path);path=root/TEST_SUPPORT_NOTEBOOK
    path.parent.mkdir();path.write_text(json.dumps(support_notebook()),encoding='utf-8')
    (path.parent/'unrelated-notebook.ipynb').write_text('not exported')
    without=tmp_path/'without-tests.zip';with_tests=tmp_path/'with-tests.zip'
    bundle_source(root,without)
    with zipfile.ZipFile(without) as archive:assert TEST_SUPPORT_NOTEBOOK not in archive.namelist()
    bundle_source(root,with_tests,include_tests=True)
    with zipfile.ZipFile(with_tests) as archive:
        assert archive.read(TEST_SUPPORT_NOTEBOOK)==path.read_bytes()
        assert 'notebooks/unrelated-notebook.ipynb' not in archive.namelist()
    assert source_allowed(TEST_SUPPORT_NOTEBOOK)
    assert not source_allowed('notebooks/unrelated-notebook.ipynb')
    extracted=tmp_path/'with-test-support'
    cloud_validator()(with_tests.read_bytes(),extracted)
    assert (extracted/TEST_SUPPORT_NOTEBOOK).read_bytes()==path.read_bytes()


@pytest.mark.parametrize('mutation',['output','execution_count','invalid_json','invalid_shape','invalid_python'])
def test_notebook_export_rejects_saved_execution_or_invalid_content(tmp_path,mutation):
    root=source_fixture(tmp_path);path=root/TEST_SUPPORT_NOTEBOOK
    path.parent.mkdir();value=support_notebook()
    if mutation=='output':value['cells'][0]['outputs']=[{'output_type':'stream','name':'stdout','text':['saved runtime output']}]
    elif mutation=='execution_count':value['cells'][0]['execution_count']=1
    elif mutation=='invalid_shape':value['cells']='invalid cells'
    elif mutation=='invalid_python':value['cells'][0]['source']=['def broken(\n']
    path.write_text('not JSON' if mutation=='invalid_json' else json.dumps(value),encoding='utf-8')
    output=tmp_path/'must-not-export.zip'
    with pytest.raises(BundleError,match='notebook'):
        bundle_source(root,output,include_tests=True)
    assert not output.exists()


def test_rehashed_executed_notebook_fails_verification_before_extraction(tmp_path):
    root=source_fixture(tmp_path);path=root/TEST_SUPPORT_NOTEBOOK
    path.parent.mkdir();path.write_text(json.dumps(support_notebook()),encoding='utf-8')
    archive=tmp_path/'source.zip';bundle_source(root,archive,include_tests=True)
    bad=support_notebook();bad['cells'][0]['execution_count']=7
    content=json.dumps(bad).encode()
    with zipfile.ZipFile(archive) as source:manifest=json.loads(source.read(MANIFEST))
    manifest['files'][TEST_SUPPORT_NOTEBOOK]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    payload=rewritten_zip(archive,replace={TEST_SUPPORT_NOTEBOOK:content,MANIFEST:json.dumps(manifest).encode()})
    tampered=tmp_path/'executed-source.zip';tampered.write_bytes(payload)
    with pytest.raises(BundleError,match='notebook'):verify_bundle(tampered)
    destination=tmp_path/'must-not-extract'
    with pytest.raises(ValueError,match='notebook'):cloud_validator()(payload,destination)
    assert not destination.exists()


def test_local_source_requirements_are_packaged_without_environment_files(tmp_path):
    root=source_fixture(tmp_path)
    (root/'requirements-grounding-local.txt').write_bytes(b'torch==2.6.0\n')
    (root/'requirements-grounding-format.txt').write_bytes(b'interegular==0.3.3\n')
    protocol=root/'experiments/dropoff/public-contrast-protocol-20261121.json'
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes(b'{"kind":"fixture_protocol"}\n')
    output=tmp_path/'local-source.zip'
    bundle_source(root,output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read('requirements-grounding-local.txt')==b'torch==2.6.0\n'
        assert archive.read('requirements-grounding-format.txt')==b'interegular==0.3.3\n'
        assert archive.read(protocol.relative_to(root).as_posix())==protocol.read_bytes()
        assert not any(name.startswith('models/') or name.startswith('.venv') for name in archive.namelist())
    extracted=tmp_path/'validated-source'
    cloud_validator()(output.read_bytes(),extracted)
    assert (extracted/protocol.relative_to(root)).read_bytes()==protocol.read_bytes()
    assert (extracted/'requirements-grounding-format.txt').read_bytes()==b'interegular==0.3.3\n'


def test_minimal_training_bundle_excludes_full_emt_checkout_entrypoint(tmp_path):
    root = source_fixture(tmp_path)
    for name in ("tools/dropoff_prepare_scene.py", "tests/test_dropoff_prepare_scene.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("import trackrag.emt  # full-checkout-only generated fixture\n")
    output = tmp_path / "source.zip"
    bundle_source(root, output, include_tests=True)
    with zipfile.ZipFile(output) as archive:
        assert "tools/dropoff_prepare_scene.py" not in archive.namelist()
        assert "tests/test_dropoff_prepare_scene.py" not in archive.namelist()
        assert REQUIRED_SOURCE <= set(archive.namelist())


@pytest.mark.parametrize("with_initializer", (False, True))
def test_namespace_layout_and_optional_top_level_initializer(tmp_path, with_initializer):
    root = source_fixture(tmp_path)
    initializer = root / "trackrag/__init__.py"
    assert not initializer.exists()
    if with_initializer:
        initializer.write_text("# optional generated package initializer\n", encoding="utf-8")
    output = tmp_path / "source.zip"
    bundle_source(root, output)
    verify_bundle(output)
    extracted = tmp_path / "cloud-project"
    cloud_validator()(output.read_bytes(), extracted)
    assert (extracted / "trackrag/contracts.py").is_file()
    assert (extracted / "trackrag/dropoff/__init__.py").is_file()
    assert (extracted / "trackrag/__init__.py").is_file() == with_initializer


def test_source_symlink_is_not_followed(tmp_path):
    root = source_fixture(tmp_path)
    target = tmp_path / "outside.py"
    target.write_text("outside")
    link = root / "trackrag/dropoff/linked.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("This platform/account cannot create a real symlink fixture")
    with pytest.raises(BundleError, match="Symlink"):
        bundle_source(root, tmp_path / "source.zip")


def test_individual_file_limit_is_checked_before_reading(tmp_path, monkeypatch):
    path = tmp_path / "large.py"
    path.write_bytes(b"12345")
    original_read = Path.read_bytes
    def forbidden_read(candidate):
        if candidate == path:
            pytest.fail("Over-budget file must not be loaded into memory")
        return original_read(candidate)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(BundleError, match="byte ceiling"):
        _regular_bytes(path, tmp_path, max_bytes=4)


@pytest.mark.parametrize("mutation", ("changed", "traversal", "duplicate", "symlink", "foreign", "missing"))
def test_bundle_verification_rejects_unbound_or_unsafe_contents(tmp_path, mutation):
    root = source_fixture(tmp_path)
    original = tmp_path / "source.zip"
    bundle_source(root, original)
    if mutation == "changed":
        blob = rewritten_zip(original, replace={"trackrag/contracts.py": b"changed"})
    elif mutation == "traversal":
        blob = rewritten_zip(original, extra={"../outside.py": b"bad"})
    elif mutation == "duplicate":
        blob = rewritten_zip(original, duplicate="trackrag/contracts.py")
    elif mutation == "symlink":
        blob = rewritten_zip(original, symlink="trackrag/dropoff/linked.py")
    elif mutation == "foreign":
        blob = rewritten_zip(original, extra={".env": b"secret"})
    else:
        with zipfile.ZipFile(original) as archive:
            manifest = json.loads(archive.read(MANIFEST))
        manifest["files"].pop("trackrag/contracts.py")
        blob = rewritten_zip(original, replace={MANIFEST: json.dumps(manifest).encode()})
    path = tmp_path / "corrupted.zip"
    path.write_bytes(blob)
    with pytest.raises(BundleError):
        verify_bundle(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def artifact_fixture(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    model = "Qwen/Qwen3-VL-8B-Instruct"
    revision = "a" * 40
    write_json(root / "training_config.json", {
        "schema_version": 1, "run_kind": "pilot", "base_model_id": model, "base_revision": revision,
    })
    write_json(root / "training_result.json", {
        "schema_version": 1, "run_kind": "pilot", "optimizer_steps": 2,
        "nonzero_updates_verified": True, "changed_adapter_tensors": 1,
    })
    for name in ("dataset_manifest.json", "inference_config.json"):
        write_json(root / name, {"generated_fixture": True})
    (root / "adapter").mkdir()
    (root / "adapter/adapter_model.safetensors").write_bytes(b"generated fixture; not real tensor weights")
    write_json(root / "adapter/adapter_config.json", {
        "peft_type": "LORA", "base_model_name_or_path": model, "revision": revision,
    })
    (root / "probe-image.jpg").write_bytes(b"generated public-probe fixture")
    write_json(root / "reload_probe.json", {
        "request": {"scene": {"frame": {"image_path": "probe-image.jpg"}}},
        "image_sha256": hashlib.sha256((root / "probe-image.jpg").read_bytes()).hexdigest(),
    })
    write_json(root / "completion.json", {"status": "pilot_completed"})
    write_json(root / "reload_verification.json", {"verified": True})
    manifest = {
        "schema_version": 1, "artifact_kind": "lora_adapter", "run_kind": "pilot",
        "base_model_id": model, "base_revision": revision,
        "files": {
            name: hashlib.sha256((root / "adapter" / name).read_bytes()).hexdigest()
            for name in ("adapter_model.safetensors", "adapter_config.json")
        },
        "verification": {
            name: True for name in ("nonzero_updates_verified", "fresh_base_reload_verified",
                                   "generation_smoke_completed", "structured_generation_valid",
                                   "spatial_proposal_generated")
        },
    }
    for field, name in (
        ("training_config_sha256", "training_config.json"),
        ("dataset_manifest_sha256", "dataset_manifest.json"),
        ("training_result_sha256", "training_result.json"),
    ):
        manifest[field] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    write_json(root / "adapter/grounding_adapter_manifest.json", manifest)
    return root


def refresh_artifact_hashes(root):
    path = root / "adapter/grounding_adapter_manifest.json"
    manifest = json.loads(path.read_text())
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        manifest["files"][name] = hashlib.sha256((root / "adapter" / name).read_bytes()).hexdigest()
    for key, name in (
        ("training_config_sha256", "training_config.json"),
        ("dataset_manifest_sha256", "dataset_manifest.json"),
        ("training_result_sha256", "training_result.json"),
    ):
        manifest[key] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    write_json(path, manifest)


def test_artifact_bundle_preserves_bound_adapter_and_probe_excludes_base_cache(tmp_path):
    root = artifact_fixture(tmp_path)
    for name in ("model.safetensors", "cache/base.bin", "checkpoints/optimizer.pt", ".env", "adapter/private.txt"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded fixture")
    output = tmp_path / "adapter.zip"
    result = bundle_artifacts(root, output)
    assert result["bundle_kind"] == "adapter"
    verification = verify_bundle(output)
    assert verification["metadata"]["model_quality_verified"] is False
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "probe-image.jpg" in names
        assert "adapter/adapter_model.safetensors" in names
        assert not names & {".env", "model.safetensors", "cache/base.bin", "checkpoints/optimizer.pt", "adapter/private.txt"}


def test_public_destination_handoff_is_distinct_from_full_adaptation(tmp_path):
    root=artifact_fixture(tmp_path)
    for name in ['training_config.json','training_result.json','adapter/grounding_adapter_manifest.json']:
        path=root/name;value=json.loads(path.read_text());value['run_kind']='public_destination'
        write_json(path,value)
    write_json(root/'completion.json',{'status':'public_destination_completed'})
    evidence=root/'evaluation/trained.jsonl';evidence.parent.mkdir()
    evidence.write_bytes(b'{"case_id":"fixture","ok":false}\n')
    refresh_artifact_hashes(root)
    output=tmp_path/'public-destination.zip'
    assert bundle_artifacts(root,output)['bundle_kind']=='adapter'
    with zipfile.ZipFile(output) as archive:
        assert json.loads(archive.read('completion.json'))['status']=='public_destination_completed'
        assert json.loads(archive.read('adapter/grounding_adapter_manifest.json'))['run_kind']=='public_destination'
        assert archive.read('evaluation/trained.jsonl')==evidence.read_bytes()
    write_json(root/'completion.json',{'status':'adaptation_completed'})
    with pytest.raises(BundleError,match='fresh-reload'):
        bundle_artifacts(root,tmp_path/'mislabelled.zip')


@pytest.mark.parametrize("mutation", ("adapter", "probe", "reload", "training_record"))
def test_unverified_or_changed_artifact_cannot_be_labelled_completed(tmp_path, mutation):
    root = artifact_fixture(tmp_path)
    if mutation == "adapter":
        (root / "adapter/adapter_model.safetensors").write_bytes(b"changed")
    elif mutation == "probe":
        (root / "probe-image.jpg").write_bytes(b"changed")
    elif mutation == "reload":
        write_json(root / "reload_verification.json", {"verified": False})
    else:
        write_json(root / "training_result.json", {"changed": True})
    with pytest.raises(BundleError):
        bundle_artifacts(root, tmp_path / "completed.zip")
    result = bundle_artifacts(root, tmp_path / "diagnostic.zip", allow_incomplete=True)
    assert result["bundle_kind"] == "diagnostic"
    assert verify_bundle(tmp_path / "diagnostic.zip")["metadata"]["run_artifacts_complete"] is False


@pytest.mark.parametrize("mutation", ("missing_schema", "wrong_kind", "base_model", "peft_revision", "peft_type", "zero_steps", "false_updates", "zero_changed"))
def test_completed_artifact_requires_realistic_consistent_training_records(tmp_path, mutation):
    root = artifact_fixture(tmp_path)
    if mutation in ("missing_schema", "wrong_kind"):
        path = root / "adapter/grounding_adapter_manifest.json"
        value = json.loads(path.read_text())
        if mutation == "missing_schema":
            value.pop("schema_version")
        else:
            value["artifact_kind"] = "base_model"
        write_json(path, value)
    elif mutation == "base_model":
        path = root / "training_config.json"
        value = json.loads(path.read_text())
        value["base_model_id"] = "another/model"
        write_json(path, value)
    elif mutation in ("peft_revision", "peft_type"):
        path = root / "adapter/adapter_config.json"
        value = json.loads(path.read_text())
        value["revision" if mutation == "peft_revision" else "peft_type"] = "b" * 40 if mutation == "peft_revision" else "OTHER"
        write_json(path, value)
    else:
        path = root / "training_result.json"
        value = json.loads(path.read_text())
        value[{"zero_steps": "optimizer_steps", "false_updates": "nonzero_updates_verified", "zero_changed": "changed_adapter_tensors"}[mutation]] = False if mutation == "false_updates" else 0
        write_json(path, value)
    refresh_artifact_hashes(root)  # Reject semantics, not just a stale file hash.
    with pytest.raises(BundleError):
        bundle_artifacts(root, tmp_path / "complete.zip")
    assert bundle_artifacts(root, tmp_path / "diagnostic.zip", allow_incomplete=True)["bundle_kind"] == "diagnostic"


@pytest.mark.parametrize("fresh", (
    {"status": "failed", "exit_code": -9, "verified": False},
    {"status": "running", "exit_code": None, "verified": False},
    {"status": "passed", "exit_code": 1, "verified": True},
    {"status": "passed", "exit_code": 0, "verified": False},
))
def test_stale_inner_training_success_cannot_override_cloud_reload_failure(tmp_path, fresh):
    root = artifact_fixture(tmp_path)  # Inner completion/reload flags are all true.
    write_json(root / "cloud_handoff.json", {"training_exit_code": 0, "fresh_process_reload": fresh})
    with pytest.raises(BundleError, match="stale inner"):
        bundle_artifacts(root, tmp_path / "complete.zip")
    result = bundle_artifacts(root, tmp_path / "diagnostic.zip", allow_incomplete=True)
    assert result["bundle_kind"] == "diagnostic"


def test_successful_cloud_reload_can_complete_and_malformed_records_can_export_diagnostics(tmp_path):
    root = artifact_fixture(tmp_path)
    write_json(root / "cloud_handoff.json", {
        "training_exit_code": 0,
        "fresh_process_reload": {"status": "passed", "exit_code": 0, "verified": True},
    })
    assert bundle_artifacts(root, tmp_path / "complete.zip")["bundle_kind"] == "adapter"
    (root / "cloud_handoff.json").write_text("invalid JSON")
    assert bundle_artifacts(root, tmp_path / "diagnostic.zip", allow_incomplete=True)["bundle_kind"] == "diagnostic"


def notebook():
    path = Path(__file__).resolve().parents[1] / "notebooks/dropoff_gpu_pilot.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


def cloud_validator():
    source = next(
        "".join(cell["source"]) for cell in notebook()["cells"]
        if cell["cell_type"] == "code" and "def verify_and_extract_source_bundle" in "".join(cell["source"])
    )
    namespace = {}
    exec(compile(source, "<cloud-source-validator>", "exec"), namespace)
    return namespace["verify_and_extract_source_bundle"]


def cloud_reload_helper():
    source = next(
        "".join(cell["source"]) for cell in notebook()["cells"]
        if cell["cell_type"] == "code" and "def run_fresh_process_reload" in "".join(cell["source"])
    )
    tree = ast.parse(source)
    definitions = ast.Module(
        body=[node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(definitions, "<cloud-reload-helper>", "exec"), namespace)
    return namespace["run_fresh_process_reload"]


def test_cloud_reload_nonzero_exit_persists_failure_and_stale_receipt_becomes_diagnostic(tmp_path):
    root = artifact_fixture(tmp_path)
    write_json(root / "cloud_handoff.json", {"training_exit_code": 0})
    runner = lambda *args, **kwargs: SimpleNamespace(returncode=-9, stdout="load started", stderr="CUDA OOM")
    with pytest.raises(RuntimeError, match="code -9"):
        cloud_reload_helper()(["fixture-command"], root, cwd=tmp_path, env={}, runner=runner)
    fresh = json.loads((root / "cloud_handoff.json").read_text())["fresh_process_reload"]
    assert fresh["status"] == "failed" and fresh["verified"] is False and fresh["exit_code"] == -9
    assert (root / "cloud_reload_stderr.txt").read_text() == "CUDA OOM"
    with pytest.raises(BundleError):
        bundle_artifacts(root, tmp_path / "complete.zip")
    assert bundle_artifacts(root, tmp_path / "diagnostic.zip", allow_incomplete=True)["bundle_kind"] == "diagnostic"


def test_cloud_reload_launch_failure_is_persisted(tmp_path):
    root = artifact_fixture(tmp_path)
    write_json(root / "cloud_handoff.json", {"training_exit_code": 0})
    def fail(*args, **kwargs):
        raise OSError("fixture process could not launch")
    with pytest.raises(OSError):
        cloud_reload_helper()(["fixture-command"], root, cwd=tmp_path, env={}, runner=fail)
    fresh = json.loads((root / "cloud_handoff.json").read_text())["fresh_process_reload"]
    assert fresh["status"] == "failed" and fresh["verified"] is False
    assert fresh["error_type"] == "OSError"


def test_notebook_is_unexecuted_python_and_cloud_validator_matches_real_bundle_format(tmp_path):
    nb = notebook()
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    root = source_fixture(tmp_path)
    archive = tmp_path / "source.zip"
    bundle_source(root, archive)
    output = tmp_path / "extracted"
    result = cloud_validator()(archive.read_bytes(), output)
    assert result["files_verified"] == len(REQUIRED_SOURCE)
    assert (output / "trackrag/contracts.py").read_bytes() == (root / "trackrag/contracts.py").read_bytes()


@pytest.mark.parametrize("mutation", ("hash", "path", "duplicate", "symlink"))
def test_cloud_upload_validator_rejects_before_creating_project(tmp_path, mutation):
    root = source_fixture(tmp_path)
    archive = tmp_path / "source.zip"
    bundle_source(root, archive)
    if mutation == "hash":
        payload = rewritten_zip(archive, replace={"trackrag/contracts.py": b"changed"})
    elif mutation == "path":
        payload = rewritten_zip(archive, extra={"../outside": b"bad"})
    elif mutation == "duplicate":
        payload = rewritten_zip(archive, duplicate="trackrag/contracts.py")
    else:
        payload = rewritten_zip(archive, symlink="trackrag/dropoff/linked.py")
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        cloud_validator()(payload, destination)
    assert not destination.exists()
