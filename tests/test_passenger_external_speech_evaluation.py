"""Synthetic stereo PCM and a local range reader; these tests use no corpus or recognizer."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from threading import Barrier
import wave

import pytest

from tools import passenger_external_speech_evaluation as study


def recording(value):
    stream = BytesIO()
    with wave.open(stream, "wb") as f:
        f.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(value.to_bytes(2, "little") * 2 * 16000)
    return stream.getvalue()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    audio = {f"fixture{i}": recording(i) for i in (1, 2)}
    study.write_new(tmp_path / "protocol.json", {"purpose": "software test only"})
    plan = study.hu((tmp_path / "protocol.json").read_bytes())
    monkeypatch.setattr(study, "PLAN_SHA", plan)
    source = {"protocol_sha256": plan, "annotation_sha256": study.ANNOTATIONS_SHA,
              "selected": [{"move": name + ".move", "dialogue": name, "start": .20, "end": .60}
                           for name in audio]}
    study.write_new(tmp_path / "cohort.json", source)
    selection = study.hu((tmp_path / "cohort.json").read_bytes())

    def read_range(url, start, end):
        name = url.rsplit("/", 1)[-1].removesuffix(".mix.wav")
        payload = audio[name]
        assert 0 <= start <= end < len(payload)
        return payload[start:end+1], len(payload)

    monkeypatch.setattr(study, "source_range", read_range)
    return tmp_path, selection, read_range


def interrupt_after_first(corpus, monkeypatch):
    work, cohort, read_range = corpus

    def interrupted(url, start, end):
        if "fixture2.mix.wav" in url:
            raise KeyboardInterrupt
        return read_range(url, start, end)

    monkeypatch.setattr(study, "source_range", interrupted)
    with pytest.raises(KeyboardInterrupt):
        study.acquire(work, cohort)
    first = work / "wavs" / "fixture1.move.wav"
    assert first.is_file() and not (work / "bound.json").exists()
    return first


def test_resume_verifies_and_reuses_byte_identical_wav(corpus, monkeypatch):
    work, cohort, read_range = corpus
    first = interrupt_after_first(corpus, monkeypatch)
    before = first.read_bytes(), first.stat().st_mtime_ns
    verified = []

    def tracked(url, start, end):
        verified.append(url.rsplit("/", 1)[-1])
        return read_range(url, start, end)

    monkeypatch.setattr(study, "source_range", tracked)
    summary = study.acquire(work, cohort)
    assert (first.read_bytes(), first.stat().st_mtime_ns) == before
    assert verified.count("fixture1.mix.wav") == 2
    assert summary == {"cases": 2, "audio_bound": 2, "bound_sha256": study.hu((work / "bound.json").read_bytes())}


def test_conflicting_partial_clip_is_preserved_and_study_is_not_sealed(corpus):
    work, cohort, _ = corpus
    (work / "wavs").mkdir()
    conflict = work / "wavs" / "fixture1.move.wav"
    conflict.write_bytes(b"interrupted or foreign file")
    with pytest.raises(ValueError, match="conflicts with the reconstructed"):
        study.acquire(work, cohort)
    assert conflict.read_bytes() == b"interrupted or foreign file"
    assert not (work / "bound.json").exists()
    assert not (work / "wavs" / "fixture2.move.wav").exists()


def test_unreachable_original_for_partial_clip_preserves_and_waits_for_retry(corpus, monkeypatch):
    work, cohort, _ = corpus
    first = interrupt_after_first(corpus, monkeypatch)
    original = first.read_bytes()

    def offline(*args):
        raise OSError("simulated offline source")

    monkeypatch.setattr(study, "source_range", offline)
    with pytest.raises(ValueError, match="Cannot verify an existing WAV"):
        study.acquire(work, cohort)
    assert first.read_bytes() == original and not (work / "bound.json").exists()


def test_completed_study_verifies_all_wavs_offline_without_rewriting(corpus, monkeypatch):
    work, cohort, _ = corpus
    expected = study.acquire(work, cohort)
    files = [work / "bound.json", *sorted((work / "wavs").glob("*.wav"))]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    monkeypatch.setattr(study, "source_range", lambda *args: pytest.fail("contacted source after sealing"))
    assert study.acquire(work, cohort) == expected
    assert before == [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]


@pytest.mark.parametrize("damage", ["wav", "missing_wav", "pcm_binding", "renamed_wav"])
def test_corrupted_or_missing_previously_bound_clip_is_rejected(corpus, monkeypatch, damage):
    work, cohort, _ = corpus
    study.acquire(work, cohort)
    monkeypatch.setattr(study, "source_range", lambda *args: pytest.fail("attempted to replace sealed evidence"))
    bound = work / "bound.json"
    data = json.loads(bound.read_bytes())
    wav = work / "wavs" / data["rows"][0]["wav"]
    if damage == "wav":
        wav.write_bytes(b"damaged test WAV")
    elif damage == "missing_wav":
        wav.unlink()
    elif damage == "pcm_binding":
        data["rows"][0]["source_pcm_sha256"] = "0" * 64
        bound.write_text(json.dumps(data))
    else:
        data["rows"][0]["wav"] = "../unrelated-file"
        bound.write_text(json.dumps(data))
    before = bound.read_bytes()
    with pytest.raises(ValueError):
        study.acquire(work, cohort)
    assert bound.read_bytes() == before
    if damage == "wav":
        assert wav.read_bytes() == b"damaged test WAV"
    if damage == "missing_wav":
        assert not wav.exists()


def test_first_acquisition_error_stays_in_sealed_denominator(corpus, monkeypatch):
    work, cohort, read_range = corpus

    def one_missing(url, start, end):
        if "fixture2.mix.wav" in url:
            raise OSError("synthetic response unavailable")
        return read_range(url, start, end)

    monkeypatch.setattr(study, "source_range", one_missing)
    original = study.acquire(work, cohort)
    assert original["cases"] == 2 and original["audio_bound"] == 1
    assert "acquisition_error" in study.load(work / "bound.json")["rows"][1]
    monkeypatch.setattr(study, "source_range", lambda *args: pytest.fail("reselected a failed item"))
    assert study.acquire(work, cohort) == original


def test_concurrent_acquisitions_publish_one_consistent_binding(corpus, monkeypatch):
    work, cohort, read_range = corpus
    gate = Barrier(2)

    def overlapping(url, start, end):
        if start == 0 and "fixture1.mix.wav" in url:
            gate.wait(timeout=10)
        return read_range(url, start, end)

    monkeypatch.setattr(study, "source_range", overlapping)
    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = [pool.submit(study.acquire, work, cohort) for _ in range(2)]
        assert left.result() == right.result()
    assert len(list((work / "wavs").glob("*.wav"))) == 2
    assert len(study.load(work / "bound.json")["rows"]) == 2
