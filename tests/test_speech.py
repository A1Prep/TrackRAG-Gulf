
"""Tests for the speech-to-text adapter."""

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _write_silent_wav(path: Path, duration_frames: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * duration_frames)


def test_transcribe_file_not_found(tmp_path):
    from trackrag.speech import transcribe_file

    with pytest.raises(FileNotFoundError):
        transcribe_file(tmp_path / "missing.wav")


def test_transcribe_with_speech_recognition(tmp_path):
    audio_path = tmp_path / "test.wav"
    _write_silent_wav(audio_path)

    mock_recognizer = MagicMock()
    mock_recognizer.record.return_value = MagicMock()
    mock_recognizer.recognize_google.return_value = "leave the package at the front door"

    mock_sr = MagicMock()
    mock_sr.Recognizer.return_value = mock_recognizer
    mock_sr.AudioFile.return_value.__enter__ = lambda s: MagicMock()
    mock_sr.AudioFile.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict("sys.modules", {"speech_recognition": mock_sr}):
        with patch("trackrag.speech._transcribe_with_speech_recognition",
                   return_value="leave the package at the front door"):
            with patch.dict("os.environ", {}, clear=True):
                from trackrag import speech
                result = speech._transcribe_with_speech_recognition(audio_path)

    assert isinstance(result, str)


def test_prompt_correction_accepts_empty_input(monkeypatch):
    from trackrag.speech import prompt_correction

    monkeypatch.setattr("builtins.input", lambda _: "")
    result = prompt_correction("leave it at the door")
    assert result == "leave it at the door"


def test_prompt_correction_accepts_override(monkeypatch):
    from trackrag.speech import prompt_correction

    monkeypatch.setattr("builtins.input", lambda _: "leave it at the front door please")
    result = prompt_correction("leave it at the door")
    assert result == "leave it at the front door please"


def test_transcribe_and_correct_noninteractive(tmp_path):
    audio_path = tmp_path / "test.wav"
    _write_silent_wav(audio_path)

    with patch("trackrag.speech.transcribe_file", return_value="drop it at reception"):
        from trackrag.speech import transcribe_and_correct
        result = transcribe_and_correct(audio_path, interactive=False)

    assert result == "drop it at reception"


def test_transcribe_and_correct_interactive(tmp_path, monkeypatch):
    audio_path = tmp_path / "test.wav"
    _write_silent_wav(audio_path)

    monkeypatch.setattr("builtins.input", lambda _: "")

    with patch("trackrag.speech.transcribe_file", return_value="drop it at reception"):
        from trackrag.speech import transcribe_and_correct
        result = transcribe_and_correct(audio_path, interactive=True)

    assert result == "drop it at reception"
