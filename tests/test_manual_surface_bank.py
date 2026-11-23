"""Verify the exact distributable maps independently of a local model cache or cloud account."""
import json
from zipfile import ZipFile

from tools.passenger_release import PUBLIC_PHOTO_LISTS, ROOT
from trackrag.dropoff import manual_surfaces
from trackrag.dropoff.parking_evidence import map_shape


def test_distributed_maps_cover_every_original_and_replay_without_private_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(manual_surfaces, "CACHE", tmp_path)
    known = {
        photo["sha256"]
        for manifest in PUBLIC_PHOTO_LISTS
        for photo in json.loads((ROOT/"experiments/dropoff"/manifest).read_bytes())["photos"]
    }
    with ZipFile(manual_surfaces.BANK) as bank:
        assert len(bank.namelist()) == len(known)
        assert set(bank.namelist()) == {sha+".json" for sha in known}
    for sha in known:
        packet = manual_surfaces.get_packet(sha)
        width, height = packet["body"]["size"]
        frame = {"width":width, "height":height}
        assert manual_surfaces.decode(packet, frame, sha).shape == map_shape(frame)
    assert manual_surfaces.get_packet("0"*64) is None
    # Corruption of a local cached copy must allow recovery from the verified distribution.
    sample = next(iter(known))
    good = manual_surfaces.get_packet(sample)
    corrupted = {**good, "signature":"00"*64}
    (tmp_path/(sample+".json")).write_text(json.dumps(corrupted), encoding="utf-8")
    assert manual_surfaces.get_packet(sample) == good
