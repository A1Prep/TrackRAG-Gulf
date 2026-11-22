"""Restore the registered Task 5 model and Task 1 scene inputs from the private release."""

import argparse
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from evaluation.grounding import load_scenes
from evaluation.grounding_inputs import find_scene
from tools.dropoff_final_input_inventory import FINAL, PACKAGE, ROOT, TASK2
from tools.dropoff_predict import read_requests
from trackrag.dropoff.data import file_sha256, inference_input_sha256
from trackrag.dropoff.inputs import _inspect_image
from trackrag.dropoff.model import GroundingConfig, verify_adapter


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_members(path, expected, manifest, *, only=None):
    """Check the published complete archive before reading its selected regular files."""
    payload = path.read_bytes()
    require(sha256(payload).hexdigest() == expected, f"Release SHA256 differs for {path.name}")
    with ZipFile(BytesIO(payload)) as archive:
        inventory = json.loads(archive.read(manifest))
        if manifest == "manifest.json":
            inventory = inventory["files"]
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(inventory) | {manifest},
                f"Archive inventory differs for {path.name}")
        result = {}
        selected = names if only is None else only
        for name in selected:
            relative = PurePosixPath(name)
            require(relative.parts and not relative.is_absolute() and ".." not in relative.parts
                    and "\\" not in name and ":" not in name and name == relative.as_posix(),
                    f"Invalid release member {name!r}")
            data = archive.read(name)
            if name != manifest:
                checksum = sha256(data).hexdigest()
                expected_member = checksum if manifest == "manifest.json" else {"bytes": len(data), "sha256": checksum}
                require(inventory[name] == expected_member, f"Release member differs: {name}")
            result[name] = data
    return result


def write_members(directory, members):
    for name, payload in members.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def require_plain(path):
    """Reject links and Windows junctions before reading an existing handoff."""
    info = path.lstat()
    attributes = getattr(info, "st_file_attributes", 0)
    linked = stat.S_ISLNK(info.st_mode) or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    require(not linked, f"Linked path in evaluator installation: {path}")


def check_members(directory, members):
    require_plain(directory)
    for name, payload in members.items():
        target = directory / name
        require(target.is_file(), f"Missing installed file: {target}")
        for part in (target, *target.parents):
            if part == directory:
                break
            require_plain(part)
        require(file_sha256(target) == sha256(payload).hexdigest(), f"Installed file differs: {target}")


def check_input_tree(directory, expected):
    """Only the registered inputs belong in this folder, including on retries."""
    allowed = {directory / name for name in expected}
    parents = {parent for file in allowed for parent in file.parents if directory in parent.parents}
    require_plain(directory)
    pending = [directory]
    found = set()
    while pending:
        for path in pending.pop().iterdir():
            require_plain(path)
            if path in parents and path.is_dir():
                pending.append(path)
            else:
                require(path in allowed and path.is_file(), f"Unexpected evaluator input: {path}")
                found.add(path)
    require(found == allowed, "The evaluator input tree is missing registered files")


def install_directory(destination, build, verify):
    """Publish a fully validated directory once. Retries check an existing installation."""
    if destination.exists() or destination.is_symlink():
        require_plain(destination)
        return verify(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "ready"
        staged.mkdir()
        build(staged)
        result = verify(staged)
        try:
            staged.rename(destination)
        except OSError:
            if destination.is_dir():
                return verify(destination)
            raise
        return result


def sources(binding):
    recording, frame = binding["recording_id"], binding["frame_id"]
    require(recording in FINAL and type(frame) is int and frame >= 0, "Unexpected Task 2 recording or frame")
    image = PurePosixPath("images", "test", recording, f"{frame:05d}.jpg")
    require(binding["original_git_image"] == f"{PACKAGE}/{image}", "Unexpected Task 2 image path")
    return image


def restore_photos(folder, root, bindings):
    for binding in bindings:
        image = sources(binding)
        obj = f"{TASK2}:{PACKAGE}/{image}"
        try:
            data = subprocess.check_output(["git", "cat-file", "blob", obj], cwd=root, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"Cannot read {obj}. Fetch origin Task-2-arjun-part-2 in this checkout.") from exc
        require(sha256(data).hexdigest() == binding["image_sha256"], f"Original photo differs: {image}")
        write_members(folder / "inputs" / "source", {str(image): data})


def verify_inputs(folder, members, bindings, release):
    photos = [f"inputs/source/{sources(binding)}" for binding in bindings]
    check_input_tree(folder, [*members, *photos])
    check_members(folder, members)
    registered = {r["case_id"]: r for r in bindings}
    scenes = load_scenes(folder / "canonical" / "scenes")
    index = json.loads(members["canonical/scenes/index.json"])
    indexed = {row["file"]: row["scene_sha256"] for row in index}
    stored = {
        name.removeprefix("canonical/scenes/"): json.loads(content)["scene_sha256"]
        for name, content in members.items()
        if name.startswith("canonical/scenes/") and name != "canonical/scenes/index.json"
    }
    require(len(indexed) == len(index) == len(scenes) and indexed == stored,
            "The Task 1 scene index must match every registered scene")
    inputs = folder / "inputs"
    detector = {q["request_id"]: q for q in read_requests(inputs / "requests-with-saved-tracks.jsonl")}
    reviewed = {q["request_id"]: q for q in read_requests(inputs / "requests-as-human-reviewed.jsonl")}
    total = release["source_asset_verification"]["canonical_scenes"]
    require(len(registered) == len(bindings) == len(scenes) == total and registered.keys() == detector.keys() == reviewed.keys(),
            "Inputs and canonical scenes must cover the complete registered cohort")
    matched = set()
    observations = 0
    for case, binding in registered.items():
        current, original = detector[case], reviewed[case]
        frame = current["scene"]["frame"]
        path, photo = _inspect_image(frame)
        require(path == (inputs / "source" / sources(binding)).resolve() and photo == binding["image_sha256"],
                f"Wrong original image for {case}")
        require(original["scene"]["frame"]["image_path"] == str(path)
                and inference_input_sha256(original, photo) == binding["reviewed_input_sha256"]
                and inference_input_sha256(current, photo) == binding["saved_detector_input_sha256"],
                f"Reviewed or detector input changed for {case}")
        tracks = current["scene"]["tracks"]
        scene, problems = find_scene(scenes, frame, tracks, photo)
        require(scene is not None and not problems
                and scene["model_input"]["timestamp_s"] == frame["timestamp_s"]
                and scene["scene_sha256"] == binding["saved_detector_conditions"]["finetuned"]["scene_sha256_from_original_hashes"],
                f"Detector observations or canonical scene differ for {case}")
        matched.add(scene["scene_sha256"])
        observations += len(tracks)
    require(matched == set(scenes) and observations == release["saved_detector_condition"]["observations"],
            "Scene coverage or total number of observations differs")
    return {"cases": total, "recordings": len({b["recording_id"] for b in bindings}),
            "detector_observations": observations,
            "reviewed_input_observations": sum(len(q["scene"]["tracks"]) for q in reviewed.values()),
            "original_photo_bytes": sum((inputs / "source" / sources(b)).stat().st_size for b in bindings)}


def restore(archives, out, *, root=ROOT):
    output_path = Path(out)
    if output_path.exists() or output_path.is_symlink():
        require_plain(output_path)
    root, archives, out = Path(root).resolve(), Path(archives).resolve(), output_path.resolve()
    records = root / "experiments" / "dropoff"
    release = json.loads((records / "passenger-final-input-readiness-20261122.json").read_bytes())
    conditions = json.loads((records / "passenger-final-model-conditions-20261122.json").read_bytes())
    canonical = release["canonical_archive"]
    sparse = checked_members(archives / "passenger-44-inputs-only-20261122.zip",
                             release["sparse_input_archive_sha256"], "inventory.json")
    scenes = checked_members(archives / canonical["filename"], canonical["sha256"], "inventory.json")
    trained, pretrained = conditions["trained"], conditions["pretrained"]
    artifact = checked_members(archives / trained["zip"], trained["archive_sha256"], "manifest.json",
                               only=("inference_config.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                                     "adapter/grounding_adapter_manifest.json"))
    pretrained_config = root / pretrained["config_path"]
    require(file_sha256(pretrained_config) == pretrained["config_sha256"]
            and sha256(artifact["inference_config.json"]).hexdigest() == trained["config_sha256"],
            "A model configuration differs from the registered comparison")
    before = json.loads(pretrained_config.read_bytes())
    after = json.loads(artifact["inference_config.json"])
    model_dir = (root / trained["config_path"]).parent
    require(before["adapter_dir"] is None and (root / after["adapter_dir"]).resolve() == model_dir / "adapter"
            and before == {**after, "adapter_dir": None}, "The frozen models differ in settings beyond the trained adapter")
    require(out != model_dir and model_dir not in out.parents and out not in model_dir.parents,
            "Choose separate model and evaluation-data directories")

    def check_model(directory):
        check_members(directory, artifact)
        config = replace(GroundingConfig.from_mapping(after), adapter_dir=str(directory / "adapter"))
        loaded = verify_adapter(config)
        require(loaded["manifest_sha256"] == trained["adapter_manifest_sha256"]
                and loaded["files"]["adapter_model.safetensors"]["sha256"] == trained["adapter_weights_sha256"],
                "Installed adapter differs from the registered trained model")

    data = {f"inputs/{name}": content for name, content in sparse.items()}
    data.update({f"canonical/{name}": content for name, content in scenes.items()})
    bindings = [json.loads(line) for line in sparse["input-bindings.jsonl"].splitlines()]

    def build(folder):
        write_members(folder, data)
        restore_photos(folder, root, bindings)

    install_directory(model_dir, lambda directory: write_members(directory, artifact), check_model)
    summary = install_directory(out, build, lambda directory: verify_inputs(directory, data, bindings, release))
    return {**summary, "trained_config": str(root / trained["config_path"]), "pretrained_config": str(pretrained_config),
            "model_working_directory": str(root), "base_model_and_processor_checked": False,
            "adapter_sha256": trained["adapter_weights_sha256"], "source_commit": TASK2,
            "canonical_scenes": str(out / "canonical" / "scenes"),
            "reviewed_requests": str(out / "inputs" / "requests-as-human-reviewed.jsonl"),
            "detector_requests": str(out / "inputs" / "requests-with-saved-tracks.jsonl"),
            "final_model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True, help="Folder containing the three private release ZIPs")
    parser.add_argument("--out", type=Path, required=True, help="Location for the private scenes, input banks and original photographs")
    args = parser.parse_args()
    print(json.dumps(restore(args.archives, args.out), indent=2))
