from hashlib import sha256
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import pytest
from PIL import Image

from tools import passenger_local_photo_install as release
from trackrag.passenger_app import load_scenes


def test_published_photo_release_contains_only_bound_derived_outputs():
    manifest = release.PUBLIC_MANIFEST
    data = manifest.read_bytes().replace(b"\r\n", b"\n")
    assert sha256(data).hexdigest() == release.MANIFEST_SHA
    metadata = json.loads(data)
    archive = manifest.with_name(release.ARCHIVE_NAME)
    assert release.file_hash(archive) == metadata["archive"]["sha256"]
    with ZipFile(archive) as packed:
        assert set(packed.namelist()) == set(metadata["archive"]["files"])
        assert all(name.endswith((".json", ".jsonl")) for name in packed.namelist())
        for name, spec in metadata["archive"]["files"].items():
            content = packed.read(name)
            assert len(content) == spec["size_bytes"] and sha256(content).hexdigest() == spec["sha256"]
            assert b"PycharmProjects" not in content and b"AppData" not in content
        rows = [json.loads(line)["scene"] for line in packed.read(release.BANK).splitlines()]
        assert len(rows) == len(metadata["sources"]) == 6 and sum(len(s["tracks"]) for s in rows) == 131
        assert all(s["provenance"]["network_policy"] == "local_only" for s in rows)


def test_local_install_fails_atomically_on_missing_or_modified_original(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "download", lambda *args: pytest.fail("Offline install contacted the network."))
    out = tmp_path / "new-install"
    with pytest.raises(FileNotFoundError, match="offline install"):
        release.install(out, tmp_path, offline=True)
    assert not out.exists()
    row = json.loads(release.PUBLIC_MANIFEST.read_bytes())["sources"][0]
    (tmp_path / (row["sha256"] + ".jpg")).write_bytes(b"incorrect original image")
    with pytest.raises(ValueError, match="declared image hash"):
        release.install(out, tmp_path, offline=True)
    assert not out.exists()


def test_install_rejects_altered_archive_before_fetch(tmp_path, monkeypatch):
    source = tmp_path / release.PUBLIC_MANIFEST.name
    shutil.copyfile(release.PUBLIC_MANIFEST, source)
    (tmp_path / release.ARCHIVE_NAME).write_bytes(b"incorrect archive")
    monkeypatch.setattr(release, "download", lambda *args: pytest.fail("A damaged release was allowed to download photos."))
    with pytest.raises(ValueError, match="archive failed verification"):
        release.install(tmp_path / "new", manifest=source)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("title", ["Sample licensed photo", None])
def test_offline_scene_installs_and_revalidates_with_actual_app_reader(tmp_path, monkeypatch, title):
    run = tmp_path / "original-machine"
    photos = run / "images"
    outputs = run / "predictions"
    photos.mkdir(parents=True)
    outputs.mkdir()
    photo = tmp_path / "original.jpg"
    Image.new("RGB", (128, 72), "navy").save(photo)
    sha = release.file_hash(photo)
    photo.rename(photos / (sha + ".jpg"))
    recording = "kartaview_123"
    identity = "1"*64
    checkpoint = "2"*64
    source = dict(recording_id=recording, frame_id=8, photo_id=19, width=128,
                  height=72, sha256=sha, source_url="https://kartaview.org/details/123/8/",
                  processed_original_url="https://cdn.kartaview.org/pr:sharp/sample", original_blur_status="BLURRED",
                  attribution="Sample contributor", license="CC BY-SA 4.0", license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                  source_terms="https://kartaview.org/terms")
    if title is not None:
        source["title"] = title
    inputs = tmp_path / "input.json"
    inputs.write_bytes(release.json_bytes({"sources": [source]}))
    scene = {"frame": {"recording_id": recording, "frame_id": 8, "timestamp_s": 0., "width": 128, "height": 72,
                       "image_path": "images/" + sha + ".jpg"}, "tracks": [], "source": "predicted",
             "provenance": {"network_policy": "local_only", "mode": "current_frame_task1", "detector_run": True,
                            "image_sha256": sha, "vision_identity_sha256": identity}}
    bank = run / release.BANK
    bank.write_bytes((json.dumps({"scene": scene})+"\n").encode())
    tracks = outputs / (recording + ".tracks.jsonl")
    tracks.write_bytes(b"")
    detections = outputs / (recording + ".detections.jsonl")
    detections.write_bytes(b"")
    provenance = outputs / (recording + ".provenance.json")
    provenance.write_bytes(release.json_bytes({"config_source": "C:/private/example", "identity_hash": identity,
        "detector": {"checkpoint": {"file": "weights.pth", "path": "C:/private/weights.pth", "sha256": checkpoint}},
        "tracks_sha256": release.file_hash(tracks)}))
    (run / "sources.json").write_bytes(release.json_bytes({"vision_identity_sha256": identity, "checkpoint_sha256": checkpoint,
        "source_manifest_sha256": release.file_hash(inputs), "scene_bank_sha256": release.file_hash(bank),
        "sources": [dict(recording_id=recording, frame_id=8, original_sha256=sha, tracks_sha256=release.file_hash(tracks),
                         detections_sha256=release.file_hash(detections), vision_provenance_sha256=release.file_hash(provenance))]}))
    public = tmp_path / "published.json"
    digest = release.pack_release(inputs, run, public)
    assert json.loads(public.read_bytes())["sources"][0].get("title") == title
    archive_digest = release.file_hash(public.with_name(release.ARCHIVE_NAME))
    assert release.pack_release(inputs, run, public) == digest
    assert release.file_hash(public.with_name(release.ARCHIVE_NAME)) == archive_digest
    other = tmp_path / "different-sources.json"
    other.write_bytes(release.json_bytes({"sources": []}))
    with pytest.raises(ValueError, match="Original source list differs"):
        release.pack_release(other, run, public)
    assert release.file_hash(public.with_name(release.ARCHIVE_NAME)) == archive_digest and release.file_hash(public) == digest
    # An OS error during the final swap must also retain the prior complete release.
    recorded = provenance.read_bytes()
    source_receipt = (run / "sources.json").read_bytes()
    changed = json.loads(recorded)
    changed["seconds"] = 5
    provenance.write_bytes(release.json_bytes(changed))
    info = json.loads(source_receipt)
    info["sources"][0]["vision_provenance_sha256"] = release.file_hash(provenance)
    (run / "sources.json").write_bytes(release.json_bytes(info))
    replace = release.os.replace
    def fail_final(src, dst):
        if Path(dst) == public:
            raise OSError("simulated manifest publication failure")
        return replace(src, dst)
    with monkeypatch.context() as m:
        m.setattr(release.os, "replace", fail_final)
        with pytest.raises(OSError, match="simulated manifest"):
            release.pack_release(inputs, run, public)
    assert release.file_hash(public.with_name(release.ARCHIVE_NAME)) == archive_digest and release.file_hash(public) == digest
    provenance.write_bytes(recorded)
    (run / "sources.json").write_bytes(source_receipt)
    monkeypatch.setattr(release, "MANIFEST_SHA", digest)
    monkeypatch.setattr(release, "download", lambda *args: pytest.fail("Offline install contacted the network."))
    fresh = tmp_path / "fresh"
    restored = release.install(fresh, photos, offline=True, manifest=public)
    assert release.file_hash(restored) == release.file_hash(bank)
    assert release.install(fresh, offline=True, manifest=public) == restored
    assert release.install(run, offline=True, manifest=public) == bank  # machine-local path spellings can differ
    loaded = next(iter(load_scenes([restored]).values()))
    assert Path(loaded["frame"]["image_path"]).parent == fresh / "images"
    (fresh / "predictions" / tracks.name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Saved Task 1 predictions disagree"):
        release.install(fresh, offline=True, manifest=public)
    assert (fresh / "predictions" / tracks.name).read_bytes() == b"changed"
