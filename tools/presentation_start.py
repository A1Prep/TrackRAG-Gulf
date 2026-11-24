"""Open the prepared demo, starting a single local server if needed."""
import argparse
import json
import os
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import webbrowser

from tools.passenger_release import ROOT


def ready(port):
    url = f"http://127.0.0.1:{port}/api/footage-catalog"
    try:
        with urlopen(url, timeout=5) as response:
            catalog = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Port {port} answered, but did not serve the TrackRAG footage catalogue.") from error
    except URLError as error:
        if isinstance(error.reason, ConnectionRefusedError):
            return False
        raise RuntimeError(f"The service on port {port} could not be checked: {error.reason}") from error
    except (ValueError, TimeoutError) as error:
        raise RuntimeError(f"The service on port {port} did not return a readable TrackRAG catalogue.") from error
    scenes = catalog.get("scenes", []) if isinstance(catalog, dict) else []
    recordings = {row.get("recording_id") for row in scenes if isinstance(row, dict) and row.get("recording_id")}
    if len(recordings) < 2 or not catalog.get("footage_available"):
        raise RuntimeError(f"The service on port {port} does not have the two searchable recordings.")
    return True


def open_presentation(port=8775, *, open_browser=True):
    if not ready(port):
        working = ROOT / "runs" / "presentation"
        working.mkdir(parents=True, exist_ok=True)
        path = working / f"start-{time.time_ns()}.log"
        args = [sys.executable, "-m", "tools.passenger_release", "start", "--require-footage", "2",
                "--port", str(port), "--low-memory", "--parking-backend", "modal", "--public-places"]
        road_model = ROOT / "runs/model-cache/airport-segformer-b0"
        if road_model.is_dir():
            args.extend(["--vision-cache", str(road_model)])
        scratch = working / "temp"
        scratch.mkdir(exist_ok=True)
        with path.open("ab") as log:
            process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                       env={**os.environ, "TEMP":str(scratch), "TMP":str(scratch)},
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        print("Opening the two recordings and the passenger photo gallery…", flush=True)
        deadline = time.monotonic() + 90
        while not ready(port):
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"The presentation server is not ready yet. Startup details are in {path}")
            time.sleep(.5)
    page = f"http://127.0.0.1:{port}/footage"
    print(f"TrackRAG is ready. {page}", flush=True)
    if open_browser:
        webbrowser.open(page)
    return page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--no-open", action="store_true", help="Check or start the presentation without opening a tab.")
    args = parser.parse_args()
    try:
        open_presentation(args.port, open_browser=not args.no_open)
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
