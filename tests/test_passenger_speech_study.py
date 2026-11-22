"""Software-only prompts, PCM and scripted responses. No physical microphone or human participant."""
from io import BytesIO
import json
from pathlib import Path
import wave

import pytest

from tools import passenger_speech_study as study
from trackrag.local_speech import SpeechInputError, SpeechTranscript


def wav(salt=1):
    out = BytesIO()
    with wave.open(out, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(int(salt).to_bytes(2, "little") * 16000)
    return out.getvalue()


class FixtureASR:
    def __init__(self, _path=None):
        pass

    def transcribe_wav(self, raw):
        return SpeechTranscript(transcript="the red car on the left", audio_sha256=study.sha(raw), duration_s=1,
            elapsed_s=0.1, sample_rate_hz=16000, channels=1, model_id=study.MODEL_ID,
            model_content_sha256=study.MODEL_CONTENT_SHA256, vosk_version="software_fixture")


def test_real_protocol_is_bound_to_daytime_data_with_counting_and_negation():
    p = study.protocol()
    assert len(p["prompts"]) == len({r["prompt_id"] for r in p["prompts"]}) == 10
    assert set(r["recording_id"] for r in p["prompts"]) == {"video_054604", "video_054907"}
    assert sum(r["wording"] == "verbatim frozen instruction" for r in p["prompts"]) == 9
    for row in p["prompts"]:
        assert len(row["image_sha256"]) == len(row["source_input_sha256"]) == 64
    scripts = " ".join(r["script"] for r in p["prompts"]).lower().split()
    assert "not" in scripts and "two" in scripts


def test_empty_study_keeps_all_missing_slots_and_no_model_load(tmp_path, monkeypatch):
    root = tmp_path / "study"
    study.prepare(root)
    monkeypatch.setattr(study, "LocalSpeechRecognizer", lambda *_: pytest.fail("no model needed for empty return folder"))
    result = study.score(root)
    assert result["overall"]["word_error_rate"] is None and result["overall"]["missing"] == 30
    assert result["speakers_with_returns"] == 0 and all(len(m) == 10 for m in result["missing_slots"].values())
    assert all(v["critical_terms"]["negation"]["exact_rate"] is None for v in result["by_speaker"].values())
    with pytest.raises(FileExistsError):
        study.prepare(root)
    (root / "recordings/S02").rmdir()
    assert study.score(root)["by_speaker"]["S02"]["missing"] == 10
    with (root / "protocol.json").open("w") as stream:
        json.dump({"wrong": True}, stream)
    with pytest.raises(ValueError, match="protocol"):
        study.score(root)


def test_no_microphone_or_playback_before_explicit_consent(tmp_path, monkeypatch):
    root = tmp_path / "study"
    study.prepare(root)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    monkeypatch.setattr(study, "record_from_microphone", lambda *_: pytest.fail("physical microphone must not be touched"))
    monkeypatch.setattr(study, "listen", lambda *_: pytest.fail("no audio permitted"))
    with pytest.raises(ValueError, match="consent"):
        study.collect(root, "S01", "quiet")
    assert not list((root / "recordings/S01").iterdir())


def test_raw_word_error_and_critical_categories_include_inserted_terms_and_reordering():
    attempt = {"receipt": {"transcript": "two red cars on the left not the white one"}, "error": None}
    got = study.measure("Two white cars on the right, the red one.", attempt)
    assert got["word_edits"] == 4
    assert got["critical_terms"]["negation"] == {"reference": [], "asr": ["not"], "assessable": True, "exact_in_order": False}
    assert got["critical_terms"]["colour"]["reference"] == ["white", "red"]
    assert got["critical_terms"]["colour"]["asr"] == ["red", "white"]
    assert not got["critical_terms"]["direction"]["exact_in_order"]
    assert got["critical_terms"]["count"]["exact_in_order"]
    empty = {"receipt": None, "error": {"code": "no_speech", "message": "fixture"}}
    row = study.measure("Not there", empty)
    total = study.summarize([{**got, "noise": "quiet", "scoring_asr": attempt}, {**row, "noise": "quiet", "scoring_asr": empty}], 10)
    assert total["recognizer_errors"] == 1 and total["word_edits"] == 6
    assert total["missing"] == 8 and total["word_error_rate"] == 6 / (got["reference_words"] + 2)


def test_listenback_frozen_before_asr_and_original_correction_preserved(tmp_path, monkeypatch):
    root = tmp_path / "study"
    study.prepare(root)
    imports = tmp_path / "own-wavs"
    imports.mkdir()
    raw = wav()
    (imports / "a01.wav").write_bytes(raw)
    steps = []
    answers = iter(["I CONSENT", "", "YES", "The blue car on the right.", "the blue car on the right"])
    monkeypatch.setattr("builtins.input", lambda prompt: (steps.append(prompt), next(answers))[1])
    monkeypatch.setattr(study, "listen", lambda *_: steps.append("PLAYBACK_FIXTURE"))
    monkeypatch.setattr(study, "record_from_microphone", lambda *_: pytest.fail("no genuine microphone in software test"))
    class OrderedASR(FixtureASR):
        def transcribe_wav(self, raw):
            steps.append("DECODE_FIXTURE")
            return super().transcribe_wav(raw)
    monkeypatch.setattr(study, "LocalSpeechRecognizer", OrderedASR)
    original_writer = study.write_new
    def mark_software(path, value):
        if path.name == "entry.json":
            value["source"] = "software_fixture"
            value["software_fixture"] = True
        original_writer(path, value)
    monkeypatch.setattr(study, "write_new", mark_software)
    study.collect(root, "S01", "quiet", single="a01", wav_dir=imports, model=Path("fixture"))
    index_ref = next(i for i, s in enumerate(steps) if s.startswith("Type the actual words"))
    assert steps.index("PLAYBACK_FIXTURE") < index_ref < steps.index("DECODE_FIXTURE")
    entry = json.loads((root / "recordings/S01/a01/entry.json").read_bytes())
    assert entry["human_reference"] == "The blue car on the right."
    assert entry["first_asr"]["receipt"]["transcript"] == "the red car on the left"
    assert entry["participant_corrected_text"] == "the blue car on the right"
    assert entry["audio_sha256"] == study.sha(raw) and entry["software_fixture"]
    with pytest.raises(ValueError, match="nonhuman"):
        study.score(root, model=Path("unused"))
    assert study.collect(root, "S01", "quiet", single="a01")["new_clips"] == 0


def test_refusing_listenback_discards_take_without_scoring(tmp_path, monkeypatch):
    root = tmp_path / "study"
    study.prepare(root)
    (tmp_path / "a01.wav").write_bytes(wav())
    answers = iter(["I CONSENT", "", "NO"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr(study, "listen", lambda *_: None)
    with pytest.raises(ValueError, match="discarded"):
        study.collect(root, "S01", "quiet", wav_dir=tmp_path, single="a01")
    assert not list((root / "recordings/S01").iterdir())


def test_scoring_refuses_stale_audio_or_prompt_and_duplicate_voice_bytes(tmp_path, monkeypatch):
    root = tmp_path / "study"
    study.prepare(root)
    spec = study.protocol()
    p = spec["prompts"][0]
    raw = wav()
    # Exercise the genuine schema in memory only. Stored test entries are explicitly fixtures.
    mock = {"version": 1, "source": "human", "speaker_id": "S01", "speaker_consented": True, "consent_statement": study.CONSENT,
            "prompt_id": p["prompt_id"], "protocol_sha256": study.sha(study.encoded(spec)), "prompt_sha256": study.sha(study.encoded(p)),
            "audio_sha256": study.sha(raw), "elicitation": "read", "noise": "quiet", "noise_note": "", "acquisition": "imported_wav",
            "human_reference": "Artificial audio only", "reference_method": "speaker_listenback_before_asr", "first_asr": None,
            "participant_corrected_text": None, **dict.fromkeys(("consented_utc", "reference_verified_utc", "saved_utc"), "2026-01-01T09:00:00Z")}
    study.check_entry(mock, raw, spec, p, "S01")
    with pytest.raises(ValueError, match="binding"):
        study.check_entry(mock, wav(2), spec, p, "S01")
    with pytest.raises(ValueError, match="binding"):
        study.check_entry({**mock, "prompt_sha256": "0" * 64}, raw, spec, p, "S01")
    with pytest.raises(ValueError, match="bound"):
        study.check_entry({**mock, "participant_corrected_text": "Invented correction"}, raw, spec, p, "S01")
    for s in ("S01", "S02"):
        folder = root / "recordings" / s / p["prompt_id"]
        folder.mkdir()
        (folder / "audio.wav").write_bytes(raw)
        study.write_new(folder / "entry.json", {**mock, "speaker_id":s, "source":"software_fixture", "software_fixture": True})
    def only_software(entry, *args):
        assert entry["source"] == "software_fixture" and entry["software_fixture"]
    monkeypatch.setattr(study, "check_entry", only_software)
    monkeypatch.setattr(study, "LocalSpeechRecognizer", FixtureASR)
    with pytest.raises(ValueError, match="same WAV"):
        study.score(root, model=Path("software"))


def test_recognizer_error_remains_explicit():
    class Broken:
        def transcribe_wav(self, _raw):
            raise SpeechInputError("no_speech", "fixture")
    result = study.decode(Broken(), wav())
    assert result == {"receipt": None, "error": {"code": "no_speech", "message": "fixture"}}


def test_original_displayed_wer_is_separate_from_replay_and_correction():
    first = {"receipt": {"transcript": "left"}, "error": None}
    repeat = {"receipt": {"transcript": "right"}, "error": None}
    observations = [{"speaker_id": "S01", "noise": "quiet", "human_reference": "right",
                     "first_asr": first, "scoring_asr": repeat, "participant_corrected_text": "right"},
                    {"speaker_id": "S02", "first_asr": None}]
    display = study.displayed_asr_report(observations, 10)
    assert display["overall"]["word_error_rate"] == 1 and display["by_speaker"]["S01"]["recorded"] == 1
    assert display["collected_without_initial_asr"] == display["different_text_or_success_on_replay"] == 1
    assert study.measure("right", repeat)["word_edits"] == 0
    count = study.measure("ten vehicles", {"receipt": {"transcript": "10 vehicles"}, "error": None})
    assert count["critical_terms"]["count"]["exact_in_order"] and count["word_edits"] == 1
