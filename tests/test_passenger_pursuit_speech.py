from io import BytesIO
from xml.etree import ElementTree
import wave

import pytest

from tools.passenger_pursuit_speech import NITE, clip, measures, span, summary


def test_scoring_preserves_direction_order_repetitions_and_missing_speech():
    vocabulary = {"left", "right"}
    measured = measures("left then right then left", "left left right", vocabulary)
    assert measured["reference_directions"] == 3 and measured["retained_directions"] == 2
    assert measured["extra_direction_words"] == 1 and not measured["all_directions_exact"]
    missing = measures("go left", "", vocabulary)
    result = summary([dict(missing, error="no_speech")])
    assert result["wer"] == 1 and result["direction_recall"] == 0 and result["errors"] == 1


def test_extract_uses_native_samples_and_clamps_padding():
    data = BytesIO()
    with wave.open(data, "wb") as src:
        src.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        src.writeframes(b"\x01\x00" * 16000)
    with wave.open(BytesIO(data.getvalue())) as original:
        segment, facts = clip(original, {"start_s": 0.1, "end_s": 0.7})
    assert facts["source_first_frame"] == 0 and facts["source_end_exclusive_frame"] == 14080
    with wave.open(BytesIO(segment)) as reader:
        assert reader.getframerate() == 16000 and reader.readframes(14080) == b"\x01\x00" * 14080


@pytest.mark.parametrize(("href", "expected"), [("s.words.xml#id(w_7)", (7, 7)),
                                                  ("s.words.xml#id(w_10)..id(w_15)", (10, 15))])
def test_manual_span_uses_inclusive_publisher_word_ids(href, expected):
    element = ElementTree.Element("segment")
    ElementTree.SubElement(element, NITE + "child", href=href)
    assert span(element) == expected
