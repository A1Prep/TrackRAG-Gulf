"""Build an offline voice-study form and import explicitly consented, pre-ASR listen-back records."""
import argparse
import base64
from binascii import Error as Base64Error
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID
import shutil

from tools import passenger_speech_study as study
from trackrag.local_speech import _decode_wav


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "passenger_voice_form.html"
WORKLET = HERE.parent / "trackrag/passenger_ui/capture.js"
MAX_CLIP = 6 * 1024 * 1024
MAX_RETURN = 85 * 1024 * 1024
ATTEST = "I listened to this complete take and verified the words I actually said before seeing any automatic transcript."
FIXTURE_CONSENT = "Software test only. No human participant is represented."
RETURN_FIELDS = {"version", "kind", "form_id", "protocol_sha256", "software_fixture", "speaker_id", "consent",
                 "consent_statement", "consented_utc", "exported_utc", "entries"}
TAKE_FIELDS = {"prompt_id", "prompt_sha256", "take_id", "audio_sha256", "audio_base64", "audio_bytes",
               "original_acquisition", "consented_utc", "selected_utc", "listened_utc", "verified_utc",
               "listenback_attestation", "human_reference", "reference_sha256", "noise", "noise_note"}


def source_text(path):
    return path.read_bytes().replace(b"\r\n", b"\n").decode("utf8")


def form_spec(fixture=False):
    spec = study.protocol()
    value = {"version": 1, "protocol_sha256": study.sha(study.encoded(spec)), "speakers": list(study.SPEAKERS),
             "noise": list(study.NOISE), "consent": FIXTURE_CONSENT if fixture else study.CONSENT,
             "attestation": ATTEST, "software_fixture": fixture,
             "prompts": [{"id": row["prompt_id"], "script": row["script"], "sha256": study.sha(study.encoded(row))}
                         for row in spec["prompts"]],
             "max_clip_bytes": MAX_CLIP, "max_return_bytes": MAX_RETURN,
             "microphone_worklet": source_text(WORKLET), "template_sha256": study.sha(source_text(TEMPLATE).encode())}
    value["form_id"] = study.sha(study.encoded(value))
    return value


def build(output, *, fixture=False):
    spec = form_spec(fixture)
    content = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = source_text(TEMPLATE)
    study.require(page.count("{{FORM_DATA}}") == 1, "Missing single offline form data slot")
    with output.open("x", encoding="utf8", newline="\n") as fp:
        fp.write(page.replace("{{FORM_DATA}}", content))
    return {"form_id": spec["form_id"], "blank_html_sha256": study.sha(output.read_bytes()),
            "speaker_slots": len(spec["speakers"]), "scripts_per_speaker": len(spec["prompts"]),
            "software_fixture": fixture}


def moment(text):
    study.require(isinstance(text, str) and 1 <= len(text) <= 48, "Invalid consent or recording time")
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    study.require(value.tzinfo is not None, "Missing timezone on participant record")
    return value


def checked_take(take, speaker, exported, plan, prompts):
    study.require(isinstance(take, dict) and set(take) == TAKE_FIELDS, "Invalid audio take fields")
    prompt = prompts.get(take["prompt_id"])
    study.require(prompt and take["prompt_sha256"] == study.sha(study.encoded(prompt)), "Unregistered script")
    uid = UUID(take["take_id"])
    study.require(uid.version == 4 and str(uid) == take["take_id"], "Unrecognized take identifier")
    method = take["original_acquisition"]
    study.require(method in ("browser_microphone", "participant_selected_original_wav"),
                  "Unknown original recording method")
    times = [moment(take[k]) for k in ("consented_utc", "selected_utc", "listened_utc", "verified_utc")]
    study.require(times == sorted(times) and times[-1] <= exported, "Consent and listen-back chronology differs")
    text = take["human_reference"]
    study.require(take["listenback_attestation"] == ATTEST and isinstance(text, str) and text == text.strip()
                  and 0 < len(text) <= 2000 and study.normalize(text) and study.sha(text.encode()) == take["reference_sha256"],
                  "Missing participant-verified spoken words or reference binding")
    noise, note = take["noise"], take["noise_note"]
    study.require(noise in study.NOISE and isinstance(note, str) and len(note) <= 300 and (noise == "quiet" or note.strip()),
                  "Invalid noise description")
    nbytes, data = take["audio_bytes"], take["audio_base64"]
    study.require(type(nbytes) is int and 44 <= nbytes <= MAX_CLIP and isinstance(data, str)
                  and len(data) == 4 * ((nbytes + 2) // 3), "Oversize or incomplete WAV payload")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, Base64Error) as exc:
        raise ValueError("Malformed WAV payload encoding") from exc
    study.require(len(raw) == nbytes and study.sha(raw) == take["audio_sha256"], "Original WAV hash differs")
    _, _, _, seconds = _decode_wav(raw)
    study.require(.25 <= seconds <= 30, "A portable take must be between 0.25 and 30 seconds")
    metadata = {k: v for k, v in take.items() if k != "audio_base64"}
    fixture = plan["software_fixture"]
    entry = {"version": 1, "source": "software_fixture" if fixture else "human", "software_fixture": fixture,
             "speaker_id": speaker, "prompt_id": prompt["prompt_id"], "protocol_sha256": plan["protocol_sha256"],
             "prompt_sha256": take["prompt_sha256"], "audio_sha256": take["audio_sha256"],
             "speaker_consented": not fixture, "consent_statement": plan["consent"], "consented_utc": take["consented_utc"],
             "elicitation": "read", "acquisition": "imported_wav", "original_acquisition": method,
             "noise": noise, "noise_note": note, "human_reference": text,
             "reference_method": "speaker_listenback_before_asr", "reference_verified_utc": take["verified_utc"],
             "first_asr": None, "participant_corrected_text": None, "saved_utc": take["verified_utc"],
             "portable_form_id": plan["form_id"], "portable_take_sha256": study.sha(study.encoded(metadata)),
             "listenback_completed_utc": take["listened_utc"], "original_audio_selected_utc": take["selected_utc"]}
    return entry, raw


def unique_object(pairs):
    value = dict(pairs)
    study.require(len(value) == len(pairs), "Duplicate JSON field in return")
    return value


@contextmanager
def lock(directory):
    marker = directory / ".portable-voice-import.lock"
    try:
        stream = marker.open("x", encoding="ascii")
    except FileExistsError as exc:
        raise ValueError("Another voice import is active. Check its recorded PID before recovering an interrupted import.") from exc
    with stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        marker.unlink()


def intake(root, paths, *, fixture=False):
    study.require(paths, "Supply at least one private return")
    if fixture:
        root.mkdir(exist_ok=True, parents=True)
        study.require(not (root / "protocol.json").exists() and not (root / "recordings").exists(),
                      "Software fixtures require a separate, non-study directory")
        marker = root / "SOFTWARE-FIXTURES-ONLY"
        marker.touch(exist_ok=True)
        spec, store = study.protocol(), root / "software-fixture-recordings"
    else:
        study.require(not (root / "SOFTWARE-FIXTURES-ONLY").exists(), "Fixture directory cannot accept human study data")
        spec, store = study.check_protocol(root), root / "recordings"
    form = form_spec(fixture)
    prompts = {p["prompt_id"]: p for p in spec["prompts"]}
    now = datetime.now(timezone.utc)
    with lock(root), TemporaryDirectory(prefix=".voice-import-", dir=root) as temporary:
        staged = Path(temporary)
        known, audio_owners, incoming = {}, {}, {}
        def register(slot, entry):
            key = entry["audio_sha256"]
            if key in audio_owners and audio_owners[key] != slot:
                raise ValueError("A single WAV cannot count for multiple scripts or speakers")
            audio_owners[key] = slot
            if slot in known:
                if (known[slot]["portable_take_sha256"] is None or
                        known[slot]["portable_take_sha256"] != entry["portable_take_sha256"]):
                    raise ValueError(f"Conflicting verified takes for {slot[0]}/{slot[1]}; use the participant's chosen return in a fresh study")
                return False
            known[slot] = entry
            return True
        for speaker in study.SPEAKERS:
            for pid, prompt in prompts.items():
                folder = store / speaker / pid
                if folder.exists():
                    entry = json.loads((folder / "entry.json").read_bytes())
                    raw = study.read_audio(folder / "audio.wav")
                    if fixture:
                        study.require(entry.get("software_fixture") is True and entry.get("source") == "software_fixture"
                                      and entry["audio_sha256"] == study.sha(raw), "Invalid existing fixture")
                    else:
                        study.check_entry(entry, raw, spec, prompt, speaker)
                    register((speaker, pid), dict(entry, portable_take_sha256=entry.get("portable_take_sha256")))
        repeated = 0
        for index, path in enumerate(paths):
            study.require(0 < path.stat().st_size <= MAX_RETURN, "Return is missing or too large")
            # Stage immutable original bytes before validation, without retaining every speaker's file in RAM.
            original = staged / f"original-{index}"
            shutil.copyfile(path, original)
            study.require(0 < original.stat().st_size <= MAX_RETURN, "Return changed size during capture")
            payload = original.read_bytes()
            outer_sha = study.sha(payload)
            result = json.loads(payload, object_pairs_hook=unique_object)
            del payload
            study.require(isinstance(result, dict) and set(result) == RETURN_FIELDS and type(result["version"]) is int
                          and result["version"] == 1 and result["kind"] == "offline_passenger_voice_return"
                          and result["software_fixture"] is fixture and result["form_id"] == form["form_id"]
                          and result["protocol_sha256"] == form["protocol_sha256"] and result["consent"] is True
                          and result["consent_statement"] == form["consent"], "Wrong protocol, form, fixture marker or consent")
            speaker = result["speaker_id"]
            exported, latest_consent = moment(result["exported_utc"]), moment(result["consented_utc"])
            study.require(speaker in study.SPEAKERS and latest_consent <= exported <= now + timedelta(minutes=5)
                          and isinstance(result["entries"], list) and 1 <= len(result["entries"]) <= len(prompts),
                          "Invalid speaker, export chronology or number of takes")
            identities = set()
            for take in result["entries"]:
                entry, raw = checked_take(take, speaker, exported, form, prompts)
                slot = speaker, entry["prompt_id"]
                study.require(slot not in identities, "A single return repeats a script")
                identities.add(slot)
                if not fixture:
                    study.check_entry(entry, raw, spec, prompts[slot[1]], speaker)
                if register(slot, entry):
                    folder = staged / speaker / slot[1]
                    folder.mkdir(parents=True)
                    (folder / "audio.wav").write_bytes(raw)
                    study.write_new(folder / "entry.json", entry)
                    incoming[slot] = folder
                else:
                    repeated += 1
            archived_original = staged / f"return-{speaker}-{outer_sha}"
            if archived_original.exists():
                original.unlink()
            else:
                original.rename(archived_original)
        archive = root / ("software-fixture-returns" if fixture else "private-portable-returns")
        archive.mkdir(exist_ok=True)
        for original in staged.glob("return-*"):
            _, speaker, digest = original.name.split("-")
            speaker_archive = archive / speaker
            speaker_archive.mkdir(exist_ok=True)
            target = speaker_archive / (digest + ".json")
            if target.exists():
                study.require(study.sha(target.read_bytes()) == digest, "Existing original return changed")
            else:
                original.rename(target)
        for (speaker, pid), folder in incoming.items():
            dest = store / speaker / pid
            dest.parent.mkdir(exist_ok=True, parents=True)
            study.require(not dest.exists(), "A concurrent collector filled this slot; rerun to verify the duplicate")
            folder.rename(dest)
    return {"new_takes": len(incoming), "identical_repeated_takes": repeated, "occupied_slots": len(known),
            "planned_slots": len(study.SPEAKERS) * len(prompts), "software_fixture": fixture,
            "human_clips_imported": 0 if fixture else len(incoming), "initial_asr_shown": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("build")
    create.add_argument("--out", required=True, type=Path)
    collect = actions.add_parser("import")
    collect.add_argument("--study", required=True, type=Path)
    collect.add_argument("--return", dest="returns", required=True, action="append", type=Path)
    args = parser.parse_args()
    result = build(args.out) if args.action == "build" else intake(args.study, args.returns)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
