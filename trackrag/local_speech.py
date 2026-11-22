"""Offline, fixed-model speech recognition for the passenger dialogue."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import wave
from array import array
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


MODEL_ID = "vosk-model-small-en-us-0.15"
MODEL_ARCHIVE_SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
MODEL_CONTENT_SHA256 = "38655871e595540ec3bd39927a7e4b9f80ab7cf53abc34b3780fb05bd4cf94b6"
MAX_WAV_BYTES = 12 * 1024 * 1024
MAX_DURATION_S = 60


class SpeechInputError(Exception):
    """A failed speech input cannot generate a passenger instruction."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SpeechTranscript:
    transcript: str
    audio_sha256: str
    duration_s: float
    elapsed_s: float
    sample_rate_hz: int
    channels: int
    model_id: str
    model_content_sha256: str
    vosk_version: str


def _verify_model(model_dir: Path):
    if not model_dir.is_dir():
        raise SpeechInputError("model_unavailable", f"English speech model folder is missing: {model_dir}")
    digest = hashlib.sha256()
    try:
        files = sorted(model_dir.rglob("*"), key=lambda p: p.relative_to(model_dir).as_posix())
        for path in files:
            relative = path.relative_to(model_dir).as_posix()
            if not path.is_file() or relative == "README":
                continue
            with path.open("rb") as source:
                file_hash = hashlib.file_digest(source, "sha256").hexdigest()
            digest.update(f"{relative}\0{file_hash}\n".encode())
    except OSError as error:
        raise SpeechInputError("model_unavailable", "Could not read the English speech model.") from error
    if digest.hexdigest() != MODEL_CONTENT_SHA256:
        raise SpeechInputError("model_mismatch", f"The selected folder does not match the {MODEL_ID} release.")


def _decode_wav(data: bytes) -> tuple[bytes, int, int, float]:
    if not isinstance(data, bytes) or not data:
        raise SpeechInputError("missing_audio", "Record or upload a WAV file.")
    if len(data) > MAX_WAV_BYTES:
        raise SpeechInputError("audio_too_large", "The WAV upload exceeds 12 MiB.")
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            channels = source.getnchannels()
            rate = source.getframerate()
            frames = source.getnframes()
            if source.getcomptype() != "NONE" or source.getsampwidth() != 2 or channels not in (1, 2):
                raise SpeechInputError("unsupported_audio", "Use mono or stereo uncompressed 16-bit PCM WAV.")
            if not 16000 <= rate <= 48000:
                raise SpeechInputError("unsupported_audio", "Use a WAV sample rate between 16 and 48 kHz.")
            if not frames:
                raise SpeechInputError("empty_audio", "The WAV file contains no samples.")
            if frames > MAX_DURATION_S * rate:
                raise SpeechInputError("audio_too_long", f"Record at most {MAX_DURATION_S} seconds per turn.")
            samples = source.readframes(frames + 1)
            if len(samples) != frames * channels * 2:
                raise SpeechInputError("malformed_audio", "The WAV recording ends before its declared length.")
    except (wave.Error, EOFError, OSError, ValueError) as error:
        raise SpeechInputError("malformed_audio", "Could not read this PCM WAV recording.") from error
    if channels == 2:
        stereo = array("h")
        stereo.frombytes(samples)
        if sys.byteorder != "little":
            stereo.byteswap()
        mono = array("h", ((left + right) // 2 for left, right in zip(stereo[::2], stereo[1::2])))
        if sys.byteorder != "little":
            mono.byteswap()
        samples = mono.tobytes()
    return samples, rate, channels, frames / rate


class LocalSpeechRecognizer:
    """Load verified weights once and create a separate decoder for each utterance."""

    def __init__(self, model_dir: str | Path):
        root = Path(model_dir).resolve()
        _verify_model(root)
        try:
            from vosk import KaldiRecognizer, Model
            self._model = Model(str(root))
            self._recognizer = KaldiRecognizer
            self.vosk_version = version("vosk")
        except (ImportError, OSError, RuntimeError) as error:
            raise SpeechInputError("asr_unavailable", "Install requirements-speech.txt in the application environment.") from error
        except Exception as error:
            # The Vosk Python wrapper raises a plain Exception on model initialization failures.
            raise SpeechInputError("asr_unavailable", "The local English decoder could not start.") from error

    def transcribe_wav(self, audio_bytes: bytes) -> SpeechTranscript:
        """Return recognized text and its audio/model receipt. No request leaves the machine."""
        start = perf_counter()
        samples, rate, channels, duration = _decode_wav(audio_bytes)
        segments = []
        try:
            decoder = self._recognizer(self._model, rate)
            for offset in range(0, len(samples), 8000):
                if decoder.AcceptWaveform(samples[offset:offset + 8000]):
                    segments.append(json.loads(decoder.Result())["text"].strip())
            segments.append(json.loads(decoder.FinalResult())["text"].strip())
        except Exception as error:
            raise SpeechInputError("asr_failed", "The local decoder failed to transcribe this recording.") from error
        text = " ".join(segment for segment in segments if segment)
        if not text:
            raise SpeechInputError("no_speech", "No words were recognized. Record again or type the instruction.")
        return SpeechTranscript(
            transcript=text, audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            duration_s=duration, elapsed_s=perf_counter() - start, sample_rate_hz=rate,
            channels=channels, model_id=MODEL_ID, model_content_sha256=MODEL_CONTENT_SHA256,
            vosk_version=self.vosk_version,
        )
