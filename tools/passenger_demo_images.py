"""Fetch the 300 original EMT images used by the passenger demo from the pinned public release."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile
import time
import urllib.error
import urllib.request
import zlib

from tools.prepare_emt_clip import BASE_URL, REVISION
from trackrag.dropoff.data import file_sha256


SOURCE = BASE_URL + "/train_images.tar.gz"
SOURCE_BYTES = 21_848_135_924
REGISTRY = Path(__file__).resolve().parents[1] / "experiments/vision/emt_frame_sha256_manifest.json"
CHUNK = 8 * 1024**2
FRAME_COUNT = 150
# Each entry is (compressed start, exclusive stop, initial decoded bytes to skip).
# The second window begins at a DEFLATE stored block in the preceding recording.
WINDOWS = {"video_054604": (0, 146_259_232, 0),
           "video_054907": (1_772_550_448, 1_923_149_088, 1_243_894)}


def fetch_range(start, stop):
    """Request one inclusive range. Reject proxies that return a different segment or the full archive."""
    request = urllib.request.Request(SOURCE + f"?passenger_demo={start}-{stop}", headers={
        "Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity", "User-Agent": "TrackRAG-Gulf/1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if (response.status != 206 or
                        response.headers.get("Content-Range") != f"bytes {start}-{stop}/{SOURCE_BYTES}"):
                    raise ValueError("Official archive returned the wrong byte range")
                data = response.read(stop - start + 2)
                if len(data) != stop - start + 1:
                    raise OSError("Incomplete archive range")
                return data
        except (TimeoutError, OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


class ChunkReader(io.RawIOBase):
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.buffer = memoryview(b"")

    def readable(self):
        return True

    def readinto(self, target):
        while not self.buffer:
            chunk = next(self.chunks, None)
            if chunk is None:
                return 0
            self.buffer = memoryview(chunk)
        count = min(len(self.buffer), len(target))
        target[:count] = self.buffer[:count]
        self.buffer = self.buffer[count:]
        return count


def decoded_chunks(bounds, counts, fetch):
    start, stop, skip = bounds
    # At the pinned stored-block length, the original bit-aligned prefix is replaced
    # by a zero stored-block header. The skipped bytes exceed the 32 KiB history
    # window. tarfile checks the next header and every image has its own SHA-256.
    stream = zlib.decompressobj(-15, zdict=bytes(32768)) if start else zlib.decompressobj(16 + zlib.MAX_WBITS)
    for offset in range(start, stop, CHUNK):
        data = fetch(offset, min(offset + CHUNK, stop) - 1)
        counts["bytes_downloaded"] += len(data)
        decoded = stream.decompress((b"\0" if start and offset == start else b"") + data, 4 * CHUNK)
        if stream.unconsumed_tail:
            raise ValueError("Unexpected expansion in pinned EMT image archive")
        if skip:
            discard = min(skip, len(decoded))
            decoded = decoded[discard:]
            skip -= discard
        if decoded:
            yield decoded


def prepare_recording(output, recording, expected, bounds, *, fetch=fetch_range):
    directory = output / "images" / "train" / recording
    directory.mkdir(parents=True, exist_ok=True)
    remaining = {}
    counts = {"recording": recording, "requested": len(expected), "already_verified": 0,
              "newly_verified": 0, "bytes_downloaded": 0}
    for frame, sha in expected.items():
        name = f"{frame:05d}.jpg"
        dest = directory / name
        if dest.exists():
            if not dest.is_file() or file_sha256(dest) != sha:
                raise ValueError(f"Existing file differs from its original EMT hash: {dest}")
            counts["already_verified"] += 1
        else:
            remaining[f"images/train/images/{recording}/{name}"] = (dest, sha)
    if not remaining:
        return counts
    chunks = decoded_chunks(bounds, counts, fetch)
    try:
        with io.BufferedReader(ChunkReader(chunks)) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                if member.name not in remaining:
                    continue
                dest, expected_sha = remaining[member.name]
                if not member.isfile() or not 0 < member.size < 8 * 1024**2:
                    raise ValueError(f"Unexpected image archive entry: {member.name}")
                image = archive.extractfile(member).read()
                if hashlib.sha256(image).hexdigest() != expected_sha:
                    raise ValueError(f"Image failed its original SHA-256: {member.name}")
                with NamedTemporaryFile(dir=directory, prefix=".emt-", delete=False) as f:
                    tmp = Path(f.name)
                    try:
                        f.write(image)
                    except BaseException:
                        f.close()
                        tmp.unlink(missing_ok=True)
                        raise
                try:
                    tmp.replace(dest)
                finally:
                    tmp.unlink(missing_ok=True)
                del remaining[member.name]
                counts["newly_verified"] += 1
                if not remaining:
                    return counts
    finally:
        chunks.close()
    raise ValueError(f"Official source window did not supply {len(remaining)} requested original frames")


def prepare(out):
    registry = json.loads(REGISTRY.read_bytes())["recordings"]
    work = [(name, {i: registry[name][str(i)] for i in range(1, FRAME_COUNT + 1)}, bounds)
            for name, bounds in WINDOWS.items()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare_recording, out, *entry) for entry in work]
        records = [task.result() for task in futures]
    return {"source_revision": REVISION, "source": SOURCE, "full_archive_bytes": SOURCE_BYTES,
            "original_registry_sha256": file_sha256(REGISTRY), "recordings": records,
            "verified_original_frames": sum(r["already_verified"] + r["newly_verified"] for r in records)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="EMT root; each verified prefix is written under images/train")
    print(json.dumps(prepare(parser.parse_args().out), indent=2))
