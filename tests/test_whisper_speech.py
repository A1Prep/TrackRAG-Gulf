"""Software-only contract checks. Public-voice evidence is recorded separately."""
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import sys
from threading import Lock
from types import ModuleType, SimpleNamespace
import wave

import pytest

from tools import passenger_asr_comparison as comparison
from tools.passenger_speech_study import decode
from tools import passenger_speech_study as team_study
from trackrag import passenger_app as app
from trackrag import whisper_speech as whisper
from trackrag.local_speech import SpeechInputError


def wav(samples=b"\x01\x00" * 3200, rate=16000):
    target = BytesIO()
    with wave.open(target, "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(samples)
    return target.getvalue()


def fake_native(transcribe):
    model = object.__new__(whisper.WhisperSpeechRecognizer)
    model._model = SimpleNamespace(transcribe=transcribe)
    model._decode = lambda stream, sampling_rate: (stream.read(), sampling_rate)
    model._lock = Lock()
    return model


def test_verified_sources_and_distinct_failure_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper, "FILES", {"model.bin": (3, sha256(b"abc").hexdigest())})
    for content, expected in ((None, "model_unavailable"), (b"a", "model_unavailable"), (b"bad", "model_mismatch")):
        if content:
            (tmp_path / "model.bin").write_bytes(content)
        with pytest.raises(SpeechInputError) as error:
            whisper.verify_model(tmp_path)
        assert error.value.code == expected
    (tmp_path / "model.bin").write_bytes(b"abc")
    assert whisper.verify_model(tmp_path) == tmp_path
    (tmp_path / "preprocessor_config.json").write_text("{}")
    with pytest.raises(SpeechInputError) as error:
        whisper.verify_model(tmp_path)
    assert error.value.code == "model_mismatch"


def test_model_initialization_enforces_local_cpu_and_engine_versions(monkeypatch, tmp_path):
    calls = []
    pkg = ModuleType("faster_whisper")
    audio = ModuleType("faster_whisper.audio")
    pkg.WhisperModel = lambda path, **kw: calls.append((path, kw))
    audio.decode_audio = object()
    monkeypatch.setitem(sys.modules, "faster_whisper", pkg)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", audio)
    monkeypatch.setattr(whisper, "verify_model", lambda _: tmp_path)
    monkeypatch.setattr(whisper, "version", lambda k: whisper.VERSIONS[k])
    whisper.WhisperSpeechRecognizer(tmp_path)
    assert calls == [(str(tmp_path), {"device":"cpu", "compute_type":"int8", "cpu_threads":4,
                                      "num_workers":1, "local_files_only":True})]
    monkeypatch.setattr(whisper, "version", lambda k: "unverified")
    with pytest.raises(SpeechInputError) as mismatch:
        whisper.WhisperSpeechRecognizer(tmp_path)
    assert mismatch.value.code == "backend_version_mismatch" and len(calls) == 1


def test_transcript_uses_wav_identity_and_frozen_settings():
    source = wav()
    def transcribe(decoded, **kwargs):
        assert decoded == (source, 16000)
        assert kwargs == {k: comparison.protocol()["candidate"][k] for k in whisper.OPTIONS}
        return iter(SimpleNamespace(text=x) for x in (" parked ", "", "cars")), None
    model = fake_native(transcribe)
    receipt = asdict(model.transcribe_wav(source))
    assert receipt["transcript"] == "parked cars" and receipt["audio_sha256"] == sha256(source).hexdigest()
    assert receipt["engine"] == "faster-whisper" and receipt["model_id"] == whisper.MODEL_ID
    assert receipt["duration_s"] == .2 and receipt["model_revision"] == whisper.REVISION and "vosk_version" not in receipt
    result = decode(model, source, expected_model=whisper.MODEL_ID, expected_digest=whisper.MODEL_CONTENT_SHA256)
    assert result["receipt"]["transcript"] == "parked cars" and result["error"] is None
    with pytest.raises(ValueError, match="pinned recognizer"):
        decode(model, source)  # The separate Vosk study cannot silently ingest the optional engine.


@pytest.mark.parametrize("raw,code", [(b"other","malformed_audio"), (wav(b"\0"*3200),"no_speech"),
                                     (wav(rate=8000),"unsupported_audio")])
def test_invalid_and_digital_silence_never_start_a_decoder(raw, code):
    def fail(*_args, **_options):
        pytest.fail("native decode should not have been invoked")
    with pytest.raises(SpeechInputError) as error:
        fake_native(fail).transcribe_wav(raw)
    assert error.value.code == code


def test_failure_and_vad_returning_no_words_are_separate():
    def broken(_input, **_options): raise RuntimeError("native decoding failed")
    for fn, code in ((broken, "asr_failed"), (lambda *args, **kw: (iter([]), None), "no_speech")):
        with pytest.raises(SpeechInputError) as error:
            fake_native(fn).transcribe_wav(wav())
        assert error.value.code == code


def test_explicit_opt_in_and_text_only_preserve_existing_release(monkeypatch):
    monkeypatch.setattr(whisper, "WhisperSpeechRecognizer", lambda path: ("whisper",path))
    monkeypatch.setattr(app, "LocalSpeechRecognizer", lambda path: ("vosk",path))
    monkeypatch.delenv("TRACKRAG_PASSENGER_WHISPER", raising=False)
    assert app.load_speech_recognizer("baseline") == ("vosk","baseline")
    monkeypatch.setenv("TRACKRAG_PASSENGER_WHISPER","candidate")
    assert app.load_speech_recognizer("baseline") == ("vosk","baseline")
    assert app.load_speech_recognizer(None) is None
    assert app.load_speech_recognizer(None, "explicit") == ("whisper","explicit")
    assert app.load_speech_recognizer("baseline","explicit") == ("whisper","explicit")


def test_pair_measurement_requires_the_same_audio_and_same_reference():
    row = {"real_filename":"file.flac", "recognizer_wav_sha256":"A", "source_sentence_sha256":"B",
           "reference_words":3, "place_mentions":1, "slurp_id":1, "speaker_slot":1, "intent":"transport_taxi",
           "microphone":"headset","word_edits":2,"word_exact":False,"ordered_place_mentions_recovered":0,
           "decode_s":1.0,"error":None, "raw_transcript_sha256":"C"}
    other = {**row, "word_edits":0,"word_exact":True,"ordered_place_mentions_recovered":1}
    result = comparison.compare([row],[other])
    assert result["fewer_word_edits"] == 1 and result["whisper"]["word_error_rate"] == 0
    assert result["vosk"]["word_error_rate"] == 2 / 3 and result["whisper"]["place_mention_recall"] == 1
    for key in ("recognizer_wav_sha256", "source_sentence_sha256"):
        with pytest.raises(ValueError, match="different input"):
            comparison.compare([row], [{**other, key:"changed"}])


def test_blank_same_wav_volunteer_supplement_has_zero_returns_and_separate_model_id(tmp_path, monkeypatch):
    folder = tmp_path / "blank"
    team_study.prepare(folder)
    monkeypatch.setattr(whisper, "WhisperSpeechRecognizer", lambda _: pytest.fail("no model needed until a consenting person records"))
    result = team_study.score(folder, whisper_model=tmp_path / "unused")
    assert result["kind"] == "human_passenger_speech_study_whisper_supplement"
    assert result["model_id"] == whisper.MODEL_ID and result["initial_display_model_id"] == team_study.MODEL_ID
    assert result["overall"]["recorded"] == result["speakers_with_returns"] == 0
    assert result["overall"]["word_error_rate"] is None and result["overall"]["missing"] == 30
    with pytest.raises(ValueError, match="one replay recognizer"):
        team_study.score(folder, model="baseline", whisper_model="candidate")
