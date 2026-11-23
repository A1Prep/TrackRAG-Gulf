"""Restore the signed Task 2 review package for the evaluation owner and verify its original manifest."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import TemporaryDirectory

from tools.dropoff_import_task2_review import REVISION, git_blob


def original_bytes(name, data, receipt):
    def matches(candidate):
        return (len(candidate) == receipt["bytes"]
                and hashlib.sha256(candidate).hexdigest() == receipt["sha256"])

    if matches(data):
        return data
    if Path(name).suffix in {".csv", ".json", ".jsonl", ".md", ".py"}:
        # The signed Windows CSVs and the other text files use different line endings.
        lf = data.replace(b"\r\n", b"\n")
        for candidate in (lf, lf.replace(b"\n", b"\r\n")):
            if matches(candidate):
                return candidate
    raise ValueError(f"Cannot restore the signed bytes for {name}")


def restore(repo, destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new evaluation output folder: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = git_blob(repo, "manifest.json")
    with TemporaryDirectory(dir=destination.parent, prefix=".restore-reviews-") as staging:
        folder = Path(staging) / "package"
        folder.mkdir()
        (folder / "manifest.json").write_bytes(manifest)
        for name, receipt in json.loads(manifest)["files"].items():
            relative = PurePosixPath(name)
            if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                    or "\\" in name or ":" in name or relative.as_posix() != name):
                raise ValueError(f"Invalid package path {name!r}")
            local = folder.joinpath(*relative.parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(original_bytes(name, git_blob(repo, name), receipt))
        result = subprocess.run([sys.executable, "-B", str(folder / "tools/validate_package.py")],
                                capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        if report.get("status") != "passed" or report.get("manifest", {}).get("verified") is not True:
            raise ValueError("Task 2's full package validation did not pass")
        folder.rename(destination)
    return {"source_revision": REVISION, **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True, help="New private directory for the evaluator")
    args = parser.parse_args()
    print(json.dumps(restore(args.repo, args.out), indent=2))


if __name__ == "__main__":
    main()
