"""Report the environment of the PC running this script."""
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def probe(label, command):
    if not shutil.which(command[0]):
        print(f"{label}: {command[0]} unavailable")
        return
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"{label}: {error}")
        return
    output = (result.stdout or result.stderr).strip()
    print(f"{label}: {output or 'no output'}")
    if result.returncode:
        print(f"  command exited with status {result.returncode}")


def main():
    project = Path(__file__).resolve().parents[1]
    free_gib = shutil.disk_usage(project).free / (1024 ** 3)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")
    print(f"Free on project volume: {free_gib:.1f} GiB")
    probe("Git", ["git", "--version"])
    probe("GPU (name, total MiB, driver)", [
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name == "torch":
            print("PyTorch: not installed in this interpreter")
        else:
            print(f"PyTorch import failed: {error}")
        return
    except (ImportError, OSError, RuntimeError) as error:
        print(f"PyTorch import failed: {error}")
        return
    print(f"PyTorch: {torch.__version__}, CUDA build {torch.version.cuda}")
    try:
        if torch.cuda.is_available():
            # Ensure the installed build can execute a kernel on this GPU.
            value = torch.tensor([1.0], device="cuda") + 1
            print(f"CUDA calculation on {torch.cuda.get_device_name(0)}: {value.item()}")
        else:
            print("CUDA unavailable to this interpreter")
    except (RuntimeError, OSError) as error:
        print(f"CUDA calculation failed: {error}")


if __name__ == "__main__":
    main()
