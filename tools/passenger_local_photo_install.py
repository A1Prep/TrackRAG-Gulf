"""Install source-verified local-only KartaView stills and their saved Task 1 predictions."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from tools.passenger_release import ROOT, download, file_hash


PUBLIC_MANIFEST = ROOT / "experiments/dropoff/passenger-local-photos-20261124.json"
ARCHIVE_NAME = "passenger-local-photo-predictions-20261124.zip"
BANK = "passenger-requests.jsonl"
MANIFEST_SHA = "1179cde63934f3c624a3394271164074fea5567a9a6c9ed19dd589073b5ddb46"
SOURCE_FIELDS = (
    "recording_id", "frame_id", "photo_id", "title", "width", "height", "sha256", "source_url",
    "processed_original_url", "original_blur_status", "attribution", "license", "license_url", "source_terms",
)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf8")


def portable_files(folder):
    """Retain original predictions and normalise two machine-dependent paths in model provenance."""
    source_info = json.loads((folder / "sources.json").read_bytes())
    files = {BANK: (folder / BANK).read_bytes()}
    if sha256(files[BANK]).hexdigest() != source_info["scene_bank_sha256"]:
        raise ValueError("The local photo scene bank differs from its source receipt.")
    for row in source_info["sources"]:
        recording = row["recording_id"]
        if not re.fullmatch(r"kartaview_[1-9]\d*", recording):
            raise ValueError("Invalid KartaView recording identity.")
        prefix = "predictions/" + recording
        for suffix, field in ((".tracks.jsonl", "tracks_sha256"), (".detections.jsonl", "detections_sha256"),
                              (".provenance.json", "vision_provenance_sha256")):
            name = prefix + suffix
            data = (folder / name).read_bytes()
            if sha256(data).hexdigest() != row[field]:
                raise ValueError(f"Saved Task 1 predictions disagree with the source receipt for {recording}.")
            if suffix == ".provenance.json":
                run = json.loads(data)
                if (run["identity_hash"] != source_info["vision_identity_sha256"]
                        or run["detector"]["checkpoint"]["sha256"] != source_info["checkpoint_sha256"]
                        or run["tracks_sha256"] != row["tracks_sha256"]):
                    raise ValueError(f"Task 1 provenance disagrees with its source scene for {recording}.")
                run["config_source"] = "configs/vision/finetuned.json"
                run["detector"]["checkpoint"]["path"] = "models/task1/" + run["detector"]["checkpoint"]["file"]
                data = json_bytes(run)
                row[field] = sha256(data).hexdigest()
            files[name] = data
    files["sources.json"] = json_bytes(source_info)
    return files


def pack_release(source_manifest, folder, destination):
    """Recreate the public metadata and compact prediction archive from the measured local run."""
    raw = json.loads(source_manifest.read_bytes())
    files = portable_files(folder)
    archive = destination.parent / ARCHIVE_NAME
    sources = []
    for row in sorted(raw["sources"], key=lambda r: (r["recording_id"], r["frame_id"])):
        source = {key: row[key] for key in SOURCE_FIELDS}
        source["original_path"] = "images/" + row["sha256"] + ".jpg"
        sources.append(source)
    identity = json.loads(files["sources.json"])
    if (identity["source_manifest_sha256"] != file_hash(source_manifest) or
            {(r["recording_id"], r["frame_id"], r["sha256"]) for r in sources} !=
            {(r["recording_id"], r["frame_id"], r["original_sha256"]) for r in identity["sources"]}):
        raise ValueError("Original source list differs from the local run.")
    public = {
        "schema": "trackrag-local-public-photo-release-v1",
        "scope": "Provider-blurred KartaView source photographs with locally computed Task 1 coordinates. Model processing stays local.",
        "attribution": "© Grab and KartaView Contributors; derived Task 1 coordinates are shared under CC BY-SA 4.0.",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "original_source_manifest_sha256": file_hash(source_manifest),
        "model_identity_sha256": identity["vision_identity_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "archive": {"file": archive.name,
                    "files": {name: {"sha256": sha256(data).hexdigest(), "size_bytes": len(data)}
                              for name, data in sorted(files.items())}},
        "sources": sources,
    }
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged_archive = Path(temporary) / archive.name
        with ZipFile(staged_archive, "w") as output:
            for name, data in sorted(files.items()):
                entry = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                output.writestr(entry, data, compress_type=ZIP_DEFLATED, compresslevel=9)
        # Preserve the release's stable serialization and only publish fully prepared files.
        public["archive"] = {"file": archive.name, "sha256": file_hash(staged_archive),
                             "files": public["archive"]["files"]}
        staged_manifest = Path(temporary) / destination.name
        staged_manifest.write_bytes(json_bytes(public))
        previous = None
        if archive.is_file():
            previous = Path(temporary) / "previous.zip"
            shutil.copyfile(archive, previous)
        os.replace(staged_archive, archive)
        try:
            os.replace(staged_manifest, destination)
        except OSError:
            if previous is not None:
                os.replace(previous, archive)
            else:
                archive.unlink()
            raise
    return sha256(destination.read_bytes()).hexdigest()


def verify_install(folder, release):
    from trackrag.passenger_app import load_scenes

    files = portable_files(folder)
    actual = {name: {"sha256": sha256(data).hexdigest(), "size_bytes": len(data)} for name, data in files.items()}
    if actual != release["archive"]["files"]:
        raise ValueError("Local prediction files differ from the published release.")
    catalog = load_scenes([folder / BANK])
    expected = {(row["recording_id"], row["frame_id"]): row for row in release["sources"]}
    if len(catalog) != len(expected):
        raise ValueError("The installed catalogue has the wrong number of source photos.")
    for scene in catalog.values():
        frame, origin = scene["frame"], scene["provenance"]
        row = expected[(frame["recording_id"], frame["frame_id"])]
        if (origin["network_policy"] != "local_only" or row["sha256"] != origin["image_sha256"]
                or (row["width"], row["height"]) != (frame["width"], frame["height"])):
            raise ValueError("Installed scene differs from its local-only source photo.")
    return folder / BANK


def install(folder, originals=None, *, offline=False, manifest=PUBLIC_MANIFEST):
    folder = Path(folder).resolve()
    manifest = Path(manifest)
    raw = manifest.read_bytes().replace(b"\r\n", b"\n")
    if sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("The published local photo manifest failed verification.")
    release = json.loads(raw)
    archive = manifest.parent / ARCHIVE_NAME
    if release["archive"]["file"] != ARCHIVE_NAME or file_hash(archive) != release["archive"]["sha256"]:
        raise ValueError("The local Task 1 prediction archive failed verification.")
    if folder.exists():
        return verify_install(folder, release)

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder.parent, prefix="." + folder.name + "-") as temporary:
        stage = Path(temporary) / "bundle"
        images = stage / "images"
        images.mkdir(parents=True)
        with ZipFile(archive) as compressed:
            members = compressed.infolist()
            expected = release["archive"]["files"]
            if len(members) != len(expected) or {m.filename for m in members} != set(expected):
                raise ValueError("The Task 1 archive contains unexpected files.")
            for member in members:
                spec = expected[member.filename]
                if member.file_size != spec["size_bytes"]:
                    raise ValueError("The Task 1 archive contains an unexpected file size.")
                data = compressed.read(member)
                if sha256(data).hexdigest() != spec["sha256"]:
                    raise ValueError("A saved Task 1 prediction failed verification.")
                target = stage / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for row in release["sources"]:
            address = urlsplit(row["processed_original_url"])
            if (address.scheme != "https" or address.netloc != "cdn.kartaview.org" or
                    not address.path.startswith("/pr:sharp/") or row["original_blur_status"] != "BLURRED"):
                raise ValueError("Expected a verified privacy-processed original from KartaView.")
            name = row["sha256"] + ".jpg"
            target = images / name
            existing = Path(originals) / name if originals is not None else None
            if existing and existing.is_file():
                if file_hash(existing) != row["sha256"]:
                    raise ValueError(f"The local original differs from its declared image hash: {name}")
                try:
                    os.link(existing, target)
                except OSError:
                    shutil.copyfile(existing, target)
            elif offline:
                raise FileNotFoundError(f"The offline install needs this original from KartaView: {name}")
            else:
                download(row["processed_original_url"], target, row["sha256"], 40 * 1024**2)
        verify_install(stage, release)
        try:
            stage.rename(folder)
        except OSError:
            if folder.is_dir():
                return verify_install(folder, release)
            raise
    return folder / BANK


def install_main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/uae-parking-local")
    parser.add_argument("--originals", type=Path, help="Directory with original KartaView JPEGs named by SHA-256.")
    parser.add_argument("--offline", action="store_true", help="Install from existing JPEG files without a network connection.")
    args = parser.parse_args(argv)
    bank = install(args.out, args.originals, offline=args.offline)
    print(f"Verified local-only KartaView photographs and their Task 1 detections: {bank}")
