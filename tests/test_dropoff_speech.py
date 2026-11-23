"""Native ASR wiring with controlled recognizers, not speech-quality evidence."""

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from trackrag.dropoff.app import LocalApplication, main
from trackrag.dropoff.speech import NativeSpeechAdapter
from trackrag.local_speech import SpeechInputError, SpeechTranscript, _decode_wav
from test_dropoff_app import corpus, idle, saved_config, select, wav_bytes


class Recognizer:
    def __init__(self, directory):
        self.directory = directory
        self.inputs = []

    def transcribe_wav(self, content):
        self.inputs.append(content)
        _, rate, channels, duration = _decode_wav(content)
        return SpeechTranscript("beside the wide car", sha256(content).hexdigest(), duration,
                                .01, rate, channels, "controlled-recognizer", "a" * 64, "test")


@pytest.fixture
def recognizers(monkeypatch):
    monkeypatch.setattr("trackrag.local_speech.LocalSpeechRecognizer", Recognizer)
    monkeypatch.setattr("trackrag.whisper_speech.WhisperSpeechRecognizer", Recognizer)


@pytest.mark.parametrize("option,engine", [("asr_model", "vosk_local"), ("whisper_model", "whisper_local")])
def test_native_bridge_selects_requested_recognizer_and_retains_audio_receipt(tmp_path, recognizers, option, engine):
    adapter = NativeSpeechAdapter(**{option: tmp_path / "speech-model"})
    path = tmp_path / "utterance.wav"
    content = wav_bytes()
    path.write_bytes(content)
    result = adapter.transcribe_audio(path)
    assert adapter.recognizer.inputs == [content]
    assert result["text"] == "beside the wide car"
    assert result["provenance"]["audio_sha256"] == sha256(content).hexdigest()
    assert result["provenance"]["model_id"] == "controlled-recognizer"
    assert result["provenance"]["backend_kind"] == engine
    assert result["provenance"]["sample_rate_hz"] == 16000


def test_native_bridge_rejects_receipt_from_another_audio(tmp_path, recognizers):
    adapter = NativeSpeechAdapter(asr_model=tmp_path)
    original = adapter.recognizer.transcribe_wav
    adapter.recognizer.transcribe_wav = lambda content: replace(original(content), audio_sha256="b" * 64)
    path = tmp_path / "utterance.wav"
    path.write_bytes(wav_bytes())
    with pytest.raises(SpeechInputError) as error:
        adapter.transcribe_audio(path)
    assert error.value.code == "invalid_speech_receipt"


@pytest.mark.parametrize("options", [{}, {"asr_model": "vosk", "whisper_model": "whisper"}])
def test_native_bridge_requires_exactly_one_model(options):
    with pytest.raises(ValueError, match="exactly one"):
        NativeSpeechAdapter(**options)


def test_real_native_missing_model_is_explicit_without_hosted_fallback(tmp_path):
    with pytest.raises(SpeechInputError) as error:
        NativeSpeechAdapter(asr_model=tmp_path / "missing")
    assert error.value.code == "model_unavailable"


def test_normal_app_preserves_native_transcript_correction_and_source_bytes(corpus, tmp_path, recognizers):
    config = replace(saved_config(corpus, tmp_path), asr_model=tmp_path / "speech-model")
    app = LocalApplication(config)
    try:
        token = select(app)["selected"]["token"]
        content = wav_bytes()
        app.audio(token, "utterance.wav", content)
        shown = idle(app)
        assert shown["condition"]["development_adapters"] is None
        assert shown["condition"]["speech"] == "vosk_local"
        audio = shown["audio"]
        assert audio["text"] == "beside the wide car"
        assert audio["provenance"]["audio_sha256"] == sha256(content).hexdigest()
        app.interpret({"scene_token": token, "audio_id": audio["audio_id"], "text": "beside the white car"})
        state = idle(app)["state"]
        assert state["request"]["original_transcript"] == "beside the wide car"
        assert state["request"]["text"] == "beside the white car"
        assert state["transcript_corrections"][0]["corrected_text"] == "beside the white car"
        assert state["confirmation"] is None
        saved = json.loads((app.output_dir / "session.json").read_text())
        source = app.output_dir / saved["audio_calls"][0]["audio_path"]
        assert source.read_bytes() == content
    finally:
        app.close()


def test_app_keeps_native_decode_failure_without_inventing_a_transcript(corpus, tmp_path, recognizers):
    app = LocalApplication(replace(saved_config(corpus, tmp_path), asr_model=tmp_path))
    try:
        token = select(app)["selected"]["token"]
        content = b"webm is not PCM WAV"
        app.audio(token, "browser.webm", content)
        assert idle(app)["audio"] is None
        call = app.audit()["audio_calls"][0]
        assert call["error"]["code"] == "malformed_audio"
        assert (app.output_dir / call["audio_path"]).read_bytes() == content
        assert app.controller.snapshot()["counts"]["requested_rows"] == 0
    finally:
        app.close()


def test_explicit_wrong_speech_module_fails_at_startup(corpus, tmp_path):
    with pytest.raises(ValueError, match="transcribe_audio must be callable"):
        LocalApplication(replace(saved_config(corpus, tmp_path), speech_adapter="trackrag.speech"))


def test_programmatic_speech_options_cannot_silently_choose_a_backend(corpus, tmp_path):
    with pytest.raises(ValueError, match="select only one"):
        LocalApplication(replace(saved_config(corpus, tmp_path), speech_adapter="trackrag.speech", asr_model=tmp_path))


def test_cli_refuses_multiple_speech_conditions(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--data-root", "unused", "--asr-model", "vosk", "--whisper-model", "whisper"])
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_cli_reports_missing_native_model_as_configuration_error(corpus, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--data-root", str(corpus), "--asr-model", str(tmp_path / "missing"),
              "--adapters", "tools.dropoff_app_demo"])
    assert error.value.code == 2
    assert "English speech model folder is missing" in capsys.readouterr().err


def test_custom_adapter_cannot_attach_a_foreign_audio_receipt(corpus, tmp_path):
    app = LocalApplication(replace(saved_config(corpus, tmp_path)))
    try:
        token = select(app)["selected"]["token"]
        app._transcribe = lambda path: {"text": "a different recording", "provenance": {"audio_sha256": "b" * 64}}
        app.audio(token, "utterance.wav", wav_bytes())
        assert idle(app)["audio"] is None
        assert "different audio bytes" in app.audit()["audio_calls"][0]["error"]["message"]
        assert app.controller.snapshot()["counts"]["requested_rows"] == 0
    finally:
        app.close()
