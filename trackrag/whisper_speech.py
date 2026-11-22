"""Optional English-only Whisper Small recognizer. All inference uses verified local files and CPU."""
from dataclasses import dataclass
from hashlib import file_digest, sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter

from trackrag.local_speech import SpeechInputError, _decode_wav


MODEL_ID = "Systran/faster-whisper-small.en"
REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
FILES = {
    "config.json": (2657, "666a9605530ac1f61fa8177f3702b4dacec9966749e42610839fcc32661d5fae"),
    "model.bin": (483545366, "62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a"),
    "tokenizer.json": (2128466, "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df"),
    "vocabulary.txt": (422309, "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"),
}
MODEL_CONTENT_SHA256 = sha256("".join(f"{name}\0{digest}\n" for name, (_,digest) in sorted(FILES.items())).encode()).hexdigest()
VERSIONS = {"faster-whisper":"1.2.1", "ctranslate2":"4.6.1", "av":"15.1.0"}
OPTIONS = {"language":"en", "task":"transcribe", "beam_size":5, "temperature":0.0,
           "condition_on_previous_text":False, "initial_prompt":None, "hotwords":None,
           "vad_filter":True, "vad_parameters":{"min_silence_duration_ms":500}}


def verify_model(directory):
    folder = Path(directory)
    # faster-whisper reads this optional file automatically. The fixed release has no such override.
    if (folder / "preprocessor_config.json").exists():
        raise SpeechInputError("model_mismatch", "The selected model contains an unpinned feature-extraction configuration.")
    for name, (size, expected) in FILES.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != size:
            raise SpeechInputError("model_unavailable", f"Missing or incomplete pinned Whisper model file: {name}")
        with path.open("rb") as file:
            if file_digest(file, "sha256").hexdigest() != expected:
                raise SpeechInputError("model_mismatch", f"Whisper model file differs from the fixed official revision: {name}")
    return folder.resolve()


@dataclass(frozen=True)
class WhisperTranscript:
    transcript: str
    audio_sha256: str
    duration_s: float
    elapsed_s: float
    sample_rate_hz: int
    channels: int
    model_id: str = MODEL_ID
    model_revision: str = REVISION
    model_content_sha256: str = MODEL_CONTENT_SHA256
    engine: str = "faster-whisper"
    engine_version: str = VERSIONS["faster-whisper"]
    ctranslate2_version: str = VERSIONS["ctranslate2"]
    av_version: str = VERSIONS["av"]
    device: str = "cpu"
    compute_type: str = "int8"


class WhisperSpeechRecognizer:
    def __init__(self, directory):
        folder = verify_model(directory)
        try:
            from faster_whisper import WhisperModel
            from faster_whisper.audio import decode_audio
            installed = {pkg: version(pkg) for pkg in VERSIONS}
        except ImportError as exc:
            raise SpeechInputError("backend_unavailable", "Install requirements-speech-whisper.txt in the local passenger environment.") from exc
        if installed != VERSIONS:
            raise SpeechInputError("backend_version_mismatch", "Install the versions in requirements-speech-whisper.txt.")
        try:
            self._model = WhisperModel(str(folder), device="cpu", compute_type="int8", cpu_threads=4, num_workers=1,
                                       local_files_only=True)
        except Exception as exc:
            raise SpeechInputError("model_unavailable", "Could not load the verified CPU Whisper recognizer.") from exc
        self._decode = decode_audio
        self._lock = Lock()

    def transcribe_wav(self, audio_bytes):
        samples, rate, channels, duration = _decode_wav(audio_bytes)
        if samples.count(b"\0") == len(samples):
            raise SpeechInputError("no_speech", "No words were recognized. Record again or type the instruction.")
        start = perf_counter()
        try:
            with self._lock:
                audio = self._decode(BytesIO(audio_bytes), sampling_rate=16000)
                segments, _ = self._model.transcribe(audio, **OPTIONS)
                text = " ".join(s.text.strip() for s in segments if s.text.strip())
        except Exception as exc:
            raise SpeechInputError("asr_failed", "The offline Whisper recognizer could not transcribe this recording.") from exc
        if not text:
            raise SpeechInputError("no_speech", "No words were recognized. Record again or type the instruction.")
        return WhisperTranscript(text, sha256(audio_bytes).hexdigest(), duration, perf_counter() - start, rate, channels)
