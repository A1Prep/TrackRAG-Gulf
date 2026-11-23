"""Bridge the existing fixed local recognizers to the Task 4 audio interface."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from trackrag.local_speech import MAX_WAV_BYTES, SpeechInputError


class NativeSpeechAdapter:
    """Select one recognizer explicitly and retain its exact audio/model receipt."""

    def __init__(self, *, asr_model=None, whisper_model=None):
        if (asr_model is None) == (whisper_model is None):
            raise ValueError("select exactly one of --asr-model and --whisper-model")
        if whisper_model is not None:
            from trackrag.whisper_speech import WhisperSpeechRecognizer
            self.recognizer = WhisperSpeechRecognizer(whisper_model)
            self.engine = "whisper_local"
            self.model_path = Path(whisper_model).resolve()
        else:
            from trackrag.local_speech import LocalSpeechRecognizer
            self.recognizer = LocalSpeechRecognizer(asr_model)
            self.engine = "vosk_local"
            self.model_path = Path(asr_model).resolve()

    def transcribe_audio(self, audio_path):
        with Path(audio_path).open("rb") as source:
            content = source.read(MAX_WAV_BYTES + 1)
        if len(content) > MAX_WAV_BYTES:
            raise SpeechInputError("audio_too_large", "The WAV upload exceeds 12 MiB.")
        receipt = asdict(self.recognizer.transcribe_wav(content))
        if receipt.get("audio_sha256") != sha256(content).hexdigest():
            raise SpeechInputError("invalid_speech_receipt", "The recognizer receipt does not match the saved audio.")
        return {"text": receipt["transcript"],
                "provenance": {**receipt, "backend_kind": self.engine}}
