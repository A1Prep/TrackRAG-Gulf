"""Offline collector checks. All sample recordings are isolated, labelled software fixtures."""
import base64
import copy
from datetime import datetime, timedelta, timezone
import io
import json
import re
from uuid import uuid4
import wave

import pytest

from tools import passenger_speech_study as study
from tools import passenger_voice_form as form


def fixture_audio(value=1):
    data = io.BytesIO()
    with wave.open(data, "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(16000)
        fp.writeframes(value.to_bytes(2, "little") * 8000)
    return data.getvalue()


def fixture_take(index=0, raw=None):
    plan = form.form_spec(True)
    p = plan["prompts"][index]
    audio = raw if raw is not None else fixture_audio(index + 1)
    start = datetime.now(timezone.utc) - timedelta(minutes=5)
    words = "software fixture reading " + str(index)
    return {"prompt_id": p["id"], "prompt_sha256": p["sha256"], "take_id": str(uuid4()),
            "audio_sha256": study.sha(audio), "audio_base64": base64.b64encode(audio).decode(),
            "audio_bytes": len(audio), "original_acquisition": "browser_microphone",
            "consented_utc": start.isoformat(), "selected_utc": (start + timedelta(seconds=1)).isoformat(),
            "listened_utc": (start + timedelta(seconds=3)).isoformat(),
            "verified_utc": (start + timedelta(seconds=4)).isoformat(), "listenback_attestation": form.ATTEST,
            "human_reference": words, "reference_sha256": study.sha(words.encode()), "noise": "quiet", "noise_note": ""}


def fixture_return(path, rows, speaker="S01"):
    spec = form.form_spec(True)
    return_data = {"version": 1, "kind": "offline_passenger_voice_return", "form_id": spec["form_id"],
        "protocol_sha256": spec["protocol_sha256"], "software_fixture": True, "speaker_id": speaker, "consent": True,
        "consent_statement": form.FIXTURE_CONSENT, "consented_utc": rows[0]["consented_utc"],
        "exported_utc": datetime.now(timezone.utc).isoformat(), "entries": rows}
    path.write_text(json.dumps(return_data), encoding="utf8")
    return path


def test_blank_is_reproducible_and_uses_the_original_three_by_ten(tmp_path):
    blank = tmp_path / "blank.html"
    meta = form.build(blank)
    page = blank.read_text(encoding="utf8")
    spec = json.loads(re.search(r'<script type="application/json" id="formData">(.*?)</script>', page, re.S)[1])
    assert spec["form_id"] == meta["form_id"] and spec["software_fixture"] is False
    assert spec["speakers"] == ["S01", "S02", "S03"] and spec["consent"] == study.CONSENT
    assert [p["script"] for p in spec["prompts"]] == [p["script"] for p in study.protocol()["prompts"]]
    assert meta == form.build(tmp_path / "second.html") and meta["scripts_per_speaker"] == 10
    assert form.build(tmp_path / "fixture.html", fixture=True)["form_id"] != spec["form_id"]
    assert "connect-src 'none'" in page and not re.search(r'(?:src|href)="https?://', page)
    with pytest.raises(FileExistsError):
        form.build(blank)


def test_partials_merge_and_repeated_return_is_idempotent(tmp_path):
    root = tmp_path / "software-fixture"
    a, b = fixture_take(0), fixture_take(1)
    b["original_acquisition"] = "participant_selected_original_wav"
    p, q = fixture_return(tmp_path / "a.json", [a]), fixture_return(tmp_path / "ab.json", [a, b])
    result = form.intake(root, [p, q], fixture=True)
    assert (result["new_takes"], result["identical_repeated_takes"], result["occupied_slots"]) == (2, 1, 2)
    assert result["initial_asr_shown"] == result["human_clips_imported"] == 0
    assert form.intake(root, [q, p], fixture=True)["identical_repeated_takes"] == 3
    assert len(list((root / "software-fixture-returns/S01").glob("*.json"))) == 2
    for original in (a, b):
        folder = root / "software-fixture-recordings/S01" / original["prompt_id"]
        entry = json.loads((folder / "entry.json").read_bytes())
        assert entry["source"] == "software_fixture" and entry["speaker_consented"] is False
        assert entry["first_asr"] is None and entry["participant_corrected_text"] is None
        assert entry["original_acquisition"] == original["original_acquisition"] and entry["acquisition"] == "imported_wav"
        assert (folder / "audio.wav").read_bytes() == base64.b64decode(original["audio_base64"])
        with pytest.raises(ValueError, match="nonhuman"):
            study.check_entry(entry, (folder / "audio.wav").read_bytes(), study.protocol(), study.protocol()["prompts"][0], "S01")
    formal = tmp_path / "formal"
    study.prepare(formal)
    with pytest.raises(ValueError, match="fixture|protocol"):
        form.intake(formal, [p])
    assert study.score(formal)["overall"]["missing"] == 30
    with pytest.raises(ValueError, match="separate"):
        form.intake(formal, [q], fixture=True)
    with pytest.raises(ValueError, match="Fixture directory"):
        form.intake(root, [p])


def test_duplicate_wav_and_conflicting_takes_leave_all_new_slots_empty(tmp_path):
    root = tmp_path / "software-fixture"
    a = fixture_take(0)
    p = fixture_return(tmp_path / "a.json", [a])
    form.intake(root, [p], fixture=True)
    fresh, changed = fixture_take(1), fixture_take(0)
    q = fixture_return(tmp_path / "other.json", [fresh, changed])
    with pytest.raises(ValueError, match="Conflicting"):
        form.intake(root, [q], fixture=True)
    assert not (root / "software-fixture-recordings/S01" / fresh["prompt_id"]).exists()
    wrong_speaker = fixture_return(tmp_path / "s02.json", [a], "S02")
    reuse = fixture_return(tmp_path / "dupe.json", [fixture_take(1, fixture_audio(1))])
    for path in (wrong_speaker, reuse):
        with pytest.raises(ValueError, match="single WAV"):
            form.intake(root, [path], fixture=True)
    assert not (root / "software-fixture-recordings/S02").exists()


@pytest.mark.parametrize(("field", "bad", "match"), [
    ("original_acquisition", "synthetic", "original recording"),
    ("reference_sha256", "0" * 64, "spoken words"),
    ("audio_sha256", "0" * 64, "WAV hash"),
    ("audio_bytes", 100000000, "WAV payload"),
    ("listenback_attestation", "", "spoken words"),
    ("listened_utc", "2000-01-01T00:00:00Z", "chronology"),
    ("prompt_sha256", "0" * 64, "Unregistered"),
    ("noise", "real_background", "noise description"),
])
def test_inconsistent_private_payload_rejected(tmp_path, field, bad, match):
    a = fixture_take()
    a[field] = bad
    with pytest.raises(ValueError, match=match):
        form.intake(tmp_path / "software", [fixture_return(tmp_path / "malformed.json", [a])], fixture=True)


def test_consent_and_duplicate_field_validation(tmp_path):
    root = tmp_path / "fixture"
    p = fixture_return(tmp_path / "return.json", [fixture_take()])
    outer = json.loads(p.read_bytes())
    for key, bad in (("consent", False), ("protocol_sha256", "0" * 64), ("software_fixture", False)):
        changed = copy.copy(outer)
        changed[key] = bad
        p.write_text(json.dumps(changed), encoding="utf8")
        with pytest.raises(ValueError, match="Wrong protocol"):
            form.intake(root, [p], fixture=True)
    p.write_text('{"version":1,"version":1}', encoding="utf8")
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        form.intake(root, [p], fixture=True)
