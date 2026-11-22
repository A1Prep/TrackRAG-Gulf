"""External metric boundaries only. The reported voices come from the actual SLURP archive."""
import pytest

from tools import passenger_external_speech as study


def test_place_names_are_whole_contiguous_spans_in_original_order():
    assert study.place_matches(["New York", "Boston"], "from new york to boston") == [True, True]
    assert study.place_matches(["New York", "Boston"], "from boston to new york") == [True, False]
    assert study.place_matches(["York", "York"], "yorkshire then york") == [True, False]
    assert study.place_matches(["New York"], "new car in york") == [False]


def test_group_wer_is_weighted_by_words_and_missing_attempts_stay_in_it():
    rows = [
        {"speaker_slot": 1, "reference_words": 3, "word_edits": 3, "word_exact": False,
         "place_mentions": 2, "ordered_place_mentions_recovered": 0, "error": "no_speech"},
        {"speaker_slot": 2, "reference_words": 9, "word_edits": 1, "word_exact": False,
         "place_mentions": 1, "ordered_place_mentions_recovered": 1, "error": None},
    ]
    got = study.aggregate(rows)
    assert got["word_error_rate"] == 4 / 12 and got["recognition_errors"] == 1
    assert got["clips_with_all_places"] == 1 and got["place_mention_recall"] == 1 / 3
    assert study.aggregate([])["word_error_rate"] is None


def test_different_private_selection_is_rejected_before_reading_voices(tmp_path, monkeypatch):
    (tmp_path / "selection.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(study, "LocalSpeechRecognizer", lambda *_: pytest.fail("no model after invalid freeze"))
    with pytest.raises(ValueError, match="pre-ASR"):
        study.score(tmp_path, None, tmp_path / "out.json", "wrong-hash")
    assert not (tmp_path / "out.json").exists()
