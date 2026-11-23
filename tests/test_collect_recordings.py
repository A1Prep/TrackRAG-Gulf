"""Software-only collector checks. No microphone, network, or participant data."""

import hashlib
import io
import json
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

from tools import collect_recordings as collector
from trackrag import speech


def source_wav(rate):
    # A nonzero signal makes an added header or a changed sample detectable.
    pcm = struct.pack("<5h", -12000, -1, 0, 1, 12000) * 200
    data = io.BytesIO()
    with wave.open(data, "wb") as source:
        source.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        source.writeframes(pcm)
    return data.getvalue(), pcm


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_microphone_container_and_pcm_are_preserved(rate, tmp_path):
    original, pcm = source_wav(rate)
    dest = tmp_path / "new" / "take.wav"
    metadata = collector.save_wav(original, dest)
    assert dest.read_bytes() == original
    assert metadata["audio_sha256"] == hashlib.sha256(original).hexdigest()
    with wave.open(str(dest), "rb") as stored:
        assert stored.getframerate() == metadata["audio_sample_rate_hz"] == rate
        assert stored.getnframes() == metadata["audio_frame_count"] == 1000
        assert stored.getparams()[:2] == (1, 2)
        assert stored.readframes(1000) == pcm
    assert metadata["audio_duration_s"] == pytest.approx(1000 / rate)


def test_rejected_capture_does_not_replace_an_existing_take(tmp_path):
    dest = tmp_path / "take.wav"
    data, _ = source_wav(48000)
    with pytest.raises(ValueError, match="truncated"):
        collector.save_wav(data[:-16], dest)
    assert not dest.exists()
    collector.save_wav(data, dest)
    different, _ = source_wav(16000)
    with pytest.raises(FileExistsError):
        collector.save_wav(different, dest)
    assert dest.read_bytes() == data


@pytest.mark.parametrize("rate", [16000, 44100])
def test_existing_whisper_reader_receives_headerless_samples_at_its_expected_rate(rate, tmp_path, monkeypatch):
    numpy = pytest.importorskip("numpy")
    original, _ = source_wav(rate)
    dest = tmp_path / "take.wav"
    collector.save_wav(original, dest)
    decoder = Mock()
    decoder.transcribe.return_value = {"text": " software output "}
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=lambda _: decoder))

    assert speech._transcribe_with_whisper(dest) == "software output"
    samples = decoder.transcribe.call_args.args[0]
    assert len(samples) == int(1000 * 16000 / rate)
    assert samples[0] == pytest.approx(-12000 / 32768)
    assert samples[-1] == pytest.approx(12000 / 32768)
    assert numpy.abs(samples).max() <= 12000 / 32768


@pytest.mark.parametrize(
    "machine,answer,expected_source,expected_final,expected_error",
    [
        ("two pedestrians", "", "reviewed_asr_unchanged", "two pedestrians", None),
        ("two pedestrians", "three pedestrians", "reviewed_asr_edited", "three pedestrians", None),
        (None, "my spoken wording", "manual_after_asr_failure", "my spoken wording", "RuntimeError"),
        (None, "", "unverified_prompt_fallback", "printed question", "RuntimeError"),
        ("", "", "unverified_prompt_fallback", "printed question", "empty_transcript"),
    ],
)
def test_real_collector_entry_appends_provenance_and_preserves_old_rows(
    machine, answer, expected_source, expected_final, expected_error, tmp_path, monkeypatch
):
    wav, _ = source_wav(44100)
    folder = tmp_path / "data" / "recordings"
    manifest = folder / "manifest.jsonl"
    monkeypatch.setattr(collector, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collector, "RECORDINGS_DIR", folder)
    monkeypatch.setattr(collector, "MANIFEST_PATH", manifest)
    folder.mkdir(parents=True)
    legacy = {"recording_id": "legacy-fixture", "speaker_id": "SOFTWARE", "audio_path": "previous.wav",
              "prompt": "older question", "transcript": "older words", "duration_s": 5, "collected_at": "earlier"}
    legacy_line = json.dumps(legacy).encode() + b"\n"
    manifest.write_bytes(legacy_line)
    monkeypatch.setattr(speech, "record_from_microphone", lambda duration_s: wav)
    decoder = Mock(return_value=(machine, "software-test"))
    if machine is None:
        decoder.side_effect = RuntimeError("software fixture offline")
    monkeypatch.setattr(speech, "transcribe_file_with_backend", decoder)
    responses = iter(["", answer])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    collector.collect_one("SOFTWARE", "printed question", 5, 1)

    assert manifest.read_bytes().startswith(legacy_line)
    first, take = [json.loads(line) for line in manifest.read_bytes().splitlines()]
    assert first == legacy
    dest = tmp_path / take["audio_path"]
    decoder.assert_called_once_with(dest)
    assert dest.read_bytes() == wav
    assert take["raw_asr_transcript"] == machine
    assert take["asr_backend"] == (None if machine is None else "software-test")
    assert take["asr_error"] == expected_error
    assert take["transcript_source"] == expected_source
    assert take["transcript"] == expected_final
    assert take["duration_s"] == 5
    assert take["audio_duration_s"] == pytest.approx(1000 / 44100)


@pytest.mark.parametrize("backend", ["gemini", "openai-whisper:base", "google-web-speech"])
def test_transcription_remembers_the_actual_backend_without_a_second_decode(backend, tmp_path, monkeypatch):
    path = tmp_path / "source.wav"
    path.write_bytes(source_wav(16000)[0])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    if backend == "gemini":
        monkeypatch.setenv("GEMINI_API_KEY", "software-fixture")
        monkeypatch.setenv("GEMINI_MODEL", "software-model")
        decoder = Mock(return_value="initial words")
        monkeypatch.setattr(speech, "_transcribe_with_gemini", decoder)
        label = "gemini:software-model"
    elif backend == "openai-whisper:base":
        decoder = Mock(return_value="initial words")
        monkeypatch.setattr(speech, "_transcribe_with_whisper", decoder)
        label = backend
    else:
        monkeypatch.setattr(speech, "_transcribe_with_whisper", Mock(side_effect=ImportError))
        decoder = Mock(return_value="initial words")
        monkeypatch.setattr(speech, "_transcribe_with_speech_recognition", decoder)
        label = backend
    assert speech.transcribe_file_with_backend(path) == ("initial words", label)
    assert decoder.call_count == 1
