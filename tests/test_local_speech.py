import hashlib
import io
import struct
import wave

import pytest

from tools.dropoff_voice_trial import edit_distance, retains_terms
from trackrag.local_speech import LocalSpeechRecognizer, SpeechInputError, _decode_wav


def wav(frames, *, rate=16000, channels=1, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return buf.getvalue()


class StubDecoder:
    def __init__(self, model, rate):
        self.segments = 0

    def AcceptWaveform(self, piece):
        self.segments += 1
        return self.segments == 1

    def Result(self):
        return '{"text": "park near"}'

    def FinalResult(self):
        return '{"text": "the red bus"}'


def recognizer(decoder=StubDecoder):
    asr = LocalSpeechRecognizer.__new__(LocalSpeechRecognizer)
    asr._model = object()
    asr._recognizer = decoder
    asr.vosk_version = "software-test"
    return asr


def test_reusable_model_decodes_all_segments_with_a_fresh_decoder_and_source_hash():
    data = wav(bytes(32000))
    asr = recognizer()
    for _ in range(2):
        receipt = asr.transcribe_wav(data)
        assert receipt.transcript == "park near the red bus"
        assert receipt.audio_sha256 == hashlib.sha256(data).hexdigest()
        assert receipt.duration_s == 1
        assert receipt.sample_rate_hz == 16000
        assert receipt.channels == 1
        assert receipt.elapsed_s >= 0


def test_stereo_48khz_preserves_time_and_downmixes():
    audio = wav(struct.pack("<hhhh", 20000, 22000, -8000, 6000), rate=48000, channels=2)
    samples, rate, channels, duration = _decode_wav(audio)
    assert (rate, channels, duration) == (48000, 2, 2 / 48000)
    assert struct.unpack("<hh", samples) == (21000, -1000)


@pytest.mark.parametrize(("audio", "code"), [
    (b"", "missing_audio"), (b"123", "malformed_audio"),
    (wav(b""), "empty_audio"), (wav(bytes(200), channels=3), "unsupported_audio"),
    (wav(bytes(100), width=1), "unsupported_audio"), (wav(bytes(100), rate=8000), "unsupported_audio"),
    (wav(bytes(3200))[:-1], "malformed_audio"),
    (wav(bytes(2 * 16000 * 60 + 2)), "audio_too_long"),
    (bytes(12 * 1024 * 1024 + 1), "audio_too_large"),
], ids=["missing", "bad-container", "no-frames", "three-channels", "eight-bit", "sample-rate",
        "truncated", "over-sixty-seconds", "over-byte-budget"])
def test_rejects_invalid_inputs_before_starting_decoder(audio, code):
    def fail(*args):
        pytest.fail("should not start the ASR decoder")
    with pytest.raises(SpeechInputError) as exc:
        recognizer(fail).transcribe_wav(audio)
    assert exc.value.code == code


def test_no_recognized_words_and_decoder_errors_are_distinct():
    class Silence(StubDecoder):
        def Result(self):
            return '{"text": ""}'

        def FinalResult(self):
            return '{"text": ""}'

    class Broken(Silence):
        def FinalResult(self):
            raise RuntimeError("failed")

    for implementation, code in [(Silence, "no_speech"), (Broken, "asr_failed")]:
        with pytest.raises(SpeechInputError) as exc:
            recognizer(implementation).transcribe_wav(wav(bytes(32000)))
        assert exc.value.code == code


def test_missing_or_unexpected_model_cannot_switch_to_network(tmp_path):
    with pytest.raises(SpeechInputError) as missing:
        LocalSpeechRecognizer(tmp_path / "absent")
    assert missing.value.code == "model_unavailable"
    (tmp_path / "wrong-model").write_bytes(b"unexpected")
    with pytest.raises(SpeechInputError) as changed:
        LocalSpeechRecognizer(tmp_path)
    assert changed.value.code == "model_mismatch"


def test_error_metrics_count_deleted_words_wrong_spatial_order_and_false_matches():
    assert edit_distance(["two", "parked", "cars"], ["two", "park", "cars"]) == 1
    assert edit_distance(["left", "of", "the", "bus"], []) == 4
    assert edit_distance(["bus"], ["the", "bus"]) == 1
    assert retains_terms(["between", "the", "two", "cars"], ["between", "two", "cars"])
    assert not retains_terms(["black", "cars", "between"], ["between", "black", "cars"])
    assert not retains_terms(["bright"], ["right"])
