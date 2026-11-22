import hashlib
import io
import tarfile
import zlib

from PIL import Image
import pytest

from tools import passenger_demo_images as demo


def original_archive():
    images, raw = {}, io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for frame in (1, 2):
            source = io.BytesIO()
            Image.new("RGB", (20, 10), (frame * 80, 3, 21)).save(source, "JPEG")
            images[frame] = source.getvalue()
            member = tarfile.TarInfo(f"images/train/images/example/{frame:05d}.jpg")
            member.size = len(images[frame])
            archive.addfile(member, io.BytesIO(images[frame]))
    return images, raw.getvalue()


@pytest.mark.parametrize("resume_block", [False, True])
def test_fetch_normal_and_indexed_windows_with_byte_verified_resume(tmp_path, monkeypatch, resume_block):
    images, archive = original_archive()
    expected = {k: hashlib.sha256(v).hexdigest() for k, v in images.items()}
    if resume_block:
        # Same DEFLATE layout as the official mid-archive index, with known leading partial member bytes.
        prefix = b"partial preceding member" * 18
        compressor = zlib.compressobj(level=0, wbits=-15)
        compressed = compressor.compress(prefix + archive) + compressor.flush(zlib.Z_SYNC_FLUSH)
        assert compressed[:1] == b"\0"
        packed = b"prior bytes" + compressed
        bounds = (len(b"prior bytes") + 1, len(packed), len(prefix))
    else:
        compressor = zlib.compressobj(level=0, wbits=31)
        packed = compressor.compress(archive) + compressor.flush()
        bounds = (0, len(packed), 0)
    monkeypatch.setattr(demo, "CHUNK", 357)
    fetch = lambda start, end: packed[start:end + 1]
    result = demo.prepare_recording(tmp_path, "example", expected, bounds, fetch=fetch)
    assert result["newly_verified"] == 2
    for frame, original in images.items():
        assert (tmp_path / f"images/train/example/{frame:05d}.jpg").read_bytes() == original
    result = demo.prepare_recording(tmp_path, "example", expected, bounds, fetch=lambda *_: pytest.fail("network used"))
    assert result["already_verified"] == 2 and result["bytes_downloaded"] == 0
    (tmp_path / "images/train/example/00002.jpg").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Existing file differs"):
        demo.prepare_recording(tmp_path, "example", expected, bounds, fetch=fetch)


def test_untrusted_source_image_cannot_be_installed(tmp_path):
    images, archive = original_archive()
    compressor = zlib.compressobj(level=0, wbits=31)
    packed = compressor.compress(archive) + compressor.flush()
    with pytest.raises(ValueError, match="failed its original SHA"):
        demo.prepare_recording(tmp_path, "example", {1: "0" * 64}, (0, len(packed), 0),
                               fetch=lambda start, end: packed[start:end + 1])
    assert not list(tmp_path.rglob("*.jpg"))
