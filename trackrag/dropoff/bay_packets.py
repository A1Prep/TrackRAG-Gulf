"""Read the current photo observations from the small release archives."""
import json
import re
from zipfile import BadZipFile, ZipFile


def read_photo_packet(path, photo_sha, max_bytes):
    if not isinstance(photo_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", photo_sha):
        return None
    name = photo_sha + ".json"
    try:
        with ZipFile(path) as archive:
            entries = [entry for entry in archive.infolist() if entry.filename == name]
            if len(entries) != 1 or entries[0].file_size > max_bytes:
                return None
            return json.loads(archive.read(entries[0]))
    except (OSError, KeyError, ValueError, BadZipFile):
        return None
