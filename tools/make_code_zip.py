"""Zip the working tree for Colab (no git clone, no tokens): python -m tools.make_code_zip <out.zip>

Entries use forward slashes (PowerShell 5.1 Compress-Archive writes backslashes, which Linux
unzip turns into odd file names). Caches, virtual environments and local data are skipped.
Prints the sha256 to record in provenance.
"""

import hashlib
import sys
import zipfile
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", ".venv-vision", "runs", "data", "models",
             "checkpoints", "outputs", ".idea", ".vscode", "wandb"}
SKIP_SUFFIXES = {".pyc", ".pth", ".pt", ".ckpt", ".onnx", ".safetensors", ".tmp", ".pem", ".key"}


def _is_secret(name: str) -> bool:
    return name == ".env" or name.startswith(".env.")


def build(target: Path) -> str:
    root = Path(__file__).resolve().parents[1]
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"{target} exists; choose a new name")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (any(part in SKIP_DIRS for part in relative.parts) or path.suffix in SKIP_SUFFIXES
                    or _is_secret(path.name) or path == target):
                continue
            if path.is_file():
                archive.write(path, relative.as_posix())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return digest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tools.make_code_zip <out.zip>")
    print(build(Path(sys.argv[1])))
