from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from threading import Barrier
from zipfile import ZipFile

import pytest

from tools import dropoff_restore_evaluator as restore


def zip_release(path, name, files, *, member, native=False):
    if native:
        inventory = {"files": {name: member}}
        manifest = "manifest.json"
    else:
        inventory = {name: member}
        manifest = "inventory.json"
    with ZipFile(path, "w") as archive:
        archive.writestr(name, files)
        archive.writestr(manifest, json.dumps(inventory))
    return sha256(path.read_bytes()).hexdigest(), manifest


@pytest.mark.parametrize("native", [True, False])
def test_both_release_formats_check_archive_and_members(tmp_path, native):
    path = tmp_path / "release.zip"
    name, data = "adapter/model.bin", b"sample bytes"
    digest = sha256(data).hexdigest()
    member = digest if native else {"bytes": len(data), "sha256": digest}
    good_sha, manifest = zip_release(path, name, data, member=member, native=native)
    assert restore.checked_members(path, good_sha, manifest, only=[name]) == {name: data}
    assert manifest in restore.checked_members(path, good_sha, manifest)
    with pytest.raises(ValueError, match="Release SHA256"):
        restore.checked_members(path, "0" * 64, manifest)
    bad_sha, manifest = zip_release(path, name, data, member="old", native=native)
    with pytest.raises(ValueError, match="Release member"):
        restore.checked_members(path, bad_sha, manifest)


def test_member_and_git_paths_are_limited_to_published_locations(tmp_path):
    path = tmp_path / "release.zip"
    data = b"sample"
    ref, manifest = zip_release(path, "../outside", data, member=sha256(data).hexdigest(), native=True)
    with pytest.raises(ValueError, match="Invalid release member"):
        restore.checked_members(path, ref, manifest)
    recording = sorted(restore.FINAL)[0]
    expected = f"images/test/{recording}/00003.jpg"
    binding = {"frame_id": 3, "recording_id": recording, "original_git_image": f"{restore.PACKAGE}/{expected}"}
    assert str(restore.sources(binding)) == expected
    for change in ({"frame_id": True}, {"frame_id": -1}, {"recording_id": "../wrong"},
                   {"original_git_image": "../labels.jsonl"}):
        with pytest.raises(ValueError):
            restore.sources({**binding, **change})


def test_existing_input_tree_rejects_any_extras_missing_images_and_links(tmp_path):
    files = {"inputs/requests.jsonl": b"requests", "inputs/source/sample.jpg": b"pixels"}
    restore.write_members(tmp_path, files)
    restore.check_input_tree(tmp_path, files)
    extra = tmp_path / "inputs/reference.jsonl"
    extra.write_bytes(b"private answers not to read")
    with pytest.raises(ValueError, match="Unexpected evaluator input"):
        restore.check_input_tree(tmp_path, files)
    assert extra.read_bytes() == b"private answers not to read"
    extra.unlink()
    photo = tmp_path / "inputs/source/sample.jpg"
    photo.unlink()
    with pytest.raises(ValueError, match="missing"):
        restore.check_input_tree(tmp_path, files)
    try:
        photo.symlink_to(tmp_path / "inputs/requests.jsonl")
    except OSError:
        return  # Windows may require developer mode or a privileged account for links.
    with pytest.raises(ValueError, match="Linked path"):
        restore.check_input_tree(tmp_path, files)


def test_existing_adapter_can_have_its_original_receipts(tmp_path):
    required = {"adapter/weights": b"registered", "inference_config.json": b"configuration"}
    restore.write_members(tmp_path, {**required, "receipts.jsonl": b"original release receipts"})
    restore.check_members(tmp_path, required)
    (tmp_path / "adapter/weights").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Installed file differs"):
        restore.check_members(tmp_path, required)
    assert (tmp_path / "adapter/weights").read_bytes() == b"changed"


def test_failed_restore_and_concurrent_retries_publish_one_verified_copy(tmp_path):
    out = tmp_path / "installed"
    files = {"a": b"first", "b": b"second"}

    def check(directory):
        restore.check_members(directory, files)
        return "valid"

    with pytest.raises(ValueError, match="Missing installed file"):
        restore.install_directory(out, lambda staged: restore.write_members(staged, {"a": files["a"]}), check)
    assert not out.exists()
    ready = Barrier(2)

    def build(staged):
        restore.write_members(staged, files)
        ready.wait(timeout=15)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = [pool.submit(restore.install_directory, out, build, check) for _ in range(2)]
        assert first.result() == second.result() == "valid"
    assert restore.install_directory(out, lambda _: pytest.fail("rewrote an existing install"), check) == "valid"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["installed"]
