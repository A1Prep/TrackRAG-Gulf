"""Reuse identical original-frame evidence in local, immutable footage review journals."""
from hashlib import file_digest, sha256
from io import BytesIO
from pathlib import Path
from threading import Lock


def _sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


class FootageImages:
    def __init__(self, folder):
        self.folder = Path(folder)
        self._images = {}
        self._lock = Lock()

    def save(self, frame, box, destination, render):
        # Always bind the current source bytes. A video frame number can be reused in another clip.
        raw = Path(frame["image_path"]).read_bytes()
        key = (sha256(raw).digest(), frame["width"], frame["height"], tuple(box) if box is not None else None)
        with self._lock:
            digest = self._images.get(key)
        if digest:
            shared = self.folder / (digest + ".png")
            try:
                if _sha(shared) == digest:
                    destination.hardlink_to(shared)
                    return
            except OSError:
                pass
        with BytesIO(raw) as source:
            render({**frame, "image_path":source}, box, destination)
        digest = _sha(destination)
        shared = self.folder / (digest + ".png")
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            try:
                shared.hardlink_to(destination)
            except FileExistsError:
                if _sha(shared) != digest:
                    return
                # Another request or a previous run already rendered these exact bytes.
                twin = destination.with_suffix(".link")
                try:
                    twin.hardlink_to(shared)
                    twin.replace(destination)
                finally:
                    twin.unlink(missing_ok=True)
        except OSError:
            # Filesystems without hard links still get the ordinary standalone PNG.
            return
        with self._lock:
            self._images[key] = digest
