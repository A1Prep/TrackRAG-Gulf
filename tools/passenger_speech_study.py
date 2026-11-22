"""Collect and score thirty consented, read English utterances on the fixed daytime scenes."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from tools.dropoff_voice_trial import words, edit_distance, synthesize
from trackrag.speech import LocalSpeechRecognizer, SpeechInputError, record_from_microphone
from trackrag.local_speech import MODEL_ID, MODEL_CONTENT_SHA256, MAX_WAV_BYTES, _decode_wav


BANK = Path(__file__).resolve().parents[1] / "annotations/dropoff/passenger-development"
INPUT_RECEIPT_SHA256 = "8397cce2e7bb77d982d328d7f65422915acbf061472d92fc1bf1dd14e3c852ef"
ENGLISH_READ_PROTOCOL_SHA256 = "1b86207a62479a0f0d90026e17a849911b9508632f216fe15c93ebaa5819ffd5"
SPEAKERS = ("S01", "S02", "S03")
NOISE = ("quiet", "real_background", "added_noise")
CASES = ("a01", "a06", "a10", "b01", "b02", "b06", "b11", "b13", "b16")
COUNT_VARIANT = "Let me out between those two vehicles, behind the white pickup and the black SUV."
CATEGORIES = {
    "colour": "white black red yellow blue green grey gray brown silver orange",
    "direction": "left right behind before after beside between ahead front back near under across",
    "count": "one two three four five six seven eight nine ten both first second third fourth",
    "negation": "not no never don't cannot can't isn't wasn't neither without",
}
CONSENT = ("I voluntarily allow the course team to keep my voice and pseudonymous transcripts privately "
           "for this project and report aggregate speech scores. I can ask the team to remove my recordings.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    raw_receipt = (BANK / "input-bindings-receipt.json").read_bytes()
    require(sha(raw_receipt) == INPUT_RECEIPT_SHA256, "frozen daytime input receipt differs")
    receipt = json.loads(raw_receipt)
    frozen = (BANK / "freeze.json").read_bytes()
    require(sha(frozen) == receipt["freeze_file_sha256"], "frozen daytime instruction manifest differs")
    bindings = (BANK / "input-bindings.jsonl").read_bytes().replace(b"\r\n", b"\n")
    require(sha(bindings) == receipt["bindings_file_sha256"], "frozen input bindings differ")
    keys = {r["request_id"]: r for r in map(json.loads, bindings.splitlines())}
    specs = {}
    for partition in ("train", "development"):
        name = f"{partition}-instructions.jsonl"
        data = (BANK / name).read_bytes().replace(b"\r\n", b"\n")
        require(sha(data) == json.loads(frozen)["files"][name]["sha256"], "frozen daytime instructions differ")
        specs.update((row["request_id"], row) for row in map(json.loads, data.splitlines()))
    prompts = []
    for code in (*CASES, "b06-count"):
        identity = "passenger-" + code.split("-")[0]
        spec, binding = specs[identity], keys[identity]
        require(spec["recording_id"] == binding["recording_id"] and spec["frame_id"] == binding["frame_id"]
                and spec["recording_id"] in ("video_054604", "video_054907"), "scene binding differs")
        variant = code == "b06-count"
        prompts.append({"prompt_id": code, "source_case_id": identity, "recording_id": spec["recording_id"],
                        "frame_id": spec["frame_id"], "image_sha256": binding["image_sha256"],
                        "source_input_sha256": binding["inference_input_sha256"], "scene_dialogue": spec.get("dialogue", []),
                        "script": COUNT_VARIANT if variant else spec["text"],
                        "wording": "predeclared counting paraphrase" if variant else "verbatim frozen instruction"})
    spec = {"version": 1, "kind": "consented_read_english_passenger_speech", "speakers": list(SPEAKERS),
            "elicitation": "read", "critical_vocabularies": CATEGORIES, "model_id": MODEL_ID,
            "model_content_sha256": MODEL_CONTENT_SHA256, "prompts": prompts}
    require(sha(encoded(spec)) == ENGLISH_READ_PROTOCOL_SHA256, "speech prompts or scoring vocabulary differ from the predeclared protocol")
    return spec


def prepare(out):
    spec = protocol()
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "protocol.json", spec)
    for speaker in SPEAKERS:
        (out / "recordings" / speaker).mkdir(parents=True)
    return {"protocol_sha256": sha(encoded(spec)), "planned_human_speakers": 3, "planned_human_recordings": 30}


def check_protocol(root):
    spec = protocol()
    require(json.loads((root / "protocol.json").read_bytes()) == spec, "local protocol differs from the frozen prompts")
    return spec


def read_audio(path):
    require(path.is_file() and path.stat().st_size <= MAX_WAV_BYTES, "WAV is missing or exceeds 12 MiB")
    raw = path.read_bytes()
    _decode_wav(raw)
    return raw


def listen(path):
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    player = shutil.which("afplay") or shutil.which("aplay")
    if player:
        subprocess.run([player, str(path)], check=True)
    else:
        print("Open this local WAV in an audio player for listen-back:", path)


def decode(asr, raw, *, expected_model=MODEL_ID, expected_digest=MODEL_CONTENT_SHA256):
    try:
        response = asdict(asr.transcribe_wav(raw))
        require(response["audio_sha256"] == sha(raw) and response["model_id"] == expected_model
                and response["model_content_sha256"] == expected_digest, "ASR receipt does not match the audio or pinned recognizer")
        return {"receipt": response, "error": None}
    except SpeechInputError as exc:
        return {"receipt": None, "error": {"code": exc.code, "message": str(exc)}}


def collect(root, speaker, noise, *, model=None, wav_dir=None, single=None, seconds=8):
    spec = check_protocol(root)
    require(speaker in SPEAKERS and noise in NOISE, "unknown anonymous speaker code or recording condition")
    require(type(seconds) is int and 1 <= seconds <= 30, "microphone duration must be 1 to 30 seconds")
    require(single is None or single in {p["prompt_id"] for p in spec["prompts"]}, "unknown prompt ID")
    prompts = [p for p in spec["prompts"] if single is None or p["prompt_id"] == single]
    speaker_dir = root / "recordings" / speaker
    todo = [p for p in prompts if not (speaker_dir / p["prompt_id"]).exists()]
    if not todo:
        return {"speaker_id": speaker, "new_clips": 0}
    print(CONSENT, "\nUse only your own voice. Do not say your name or other identifying details.")
    require(input("Type I CONSENT to participate, or press Enter to leave: ").strip() == "I CONSENT",
            "no consent given")
    consent_time = now()
    detail = input("Describe background noise briefly (leave blank for a quiet room): ").strip()
    require((noise == "quiet" or detail) and len(detail) <= 300, "describe real or added noise briefly")
    asr = LocalSpeechRecognizer(model) if model else None
    speaker_dir.mkdir(exist_ok=True)
    for p in todo:
        identity = p["prompt_id"]
        print(f"\n{identity} · {p['recording_id']} · frame {p['frame_id']}\n{p['script']}")
        if wav_dir:
            raw = read_audio(wav_dir / (identity + ".wav"))
        else:
            require(input("Press Enter to start this microphone take, or type STOP: ").strip() == "",
                    "participant stopped before capture")
            raw = record_from_microphone(seconds)
            _decode_wav(raw)
        # The temporary and published folders have the same parent for atomic, one-clip commits.
        with TemporaryDirectory(prefix=".pending-", dir=speaker_dir) as temp:
            stage = Path(temp)
            path = stage / "audio.wav"
            path.write_bytes(raw)
            listen(path)
            require(input("Did you listen and hear your own complete take? Type YES to keep it: ").strip() == "YES",
                    "unverified take was discarded; you can retry")
            reference = input("Type the actual words heard, or press Enter only if they exactly matched the displayed script: ").strip() or p["script"]
            require(reference.strip() and len(reference) <= 2000 and words(reference), "reference must contain spoken words")
            verified_time = now()
            recognized = decode(asr, raw) if asr else None
            corrected = None
            if recognized is not None:
                original = recognized["receipt"]["transcript"] if recognized["receipt"] else ""
                print("Original recognizer text:", original or "(no transcription)")
                corrected = input("Enter the text you would send, or press Enter to use that recognizer text: ").strip() or original or None
                require(corrected is None or len(corrected) <= 2000, "correction exceeds the text limit")
            entry = {"version": 1, "source": "human", "speaker_id": speaker, "prompt_id": identity,
                     "protocol_sha256": sha(encoded(spec)), "prompt_sha256": sha(encoded(p)), "audio_sha256": sha(raw),
                     "speaker_consented": True, "consent_statement": CONSENT, "consented_utc": consent_time,
                     "elicitation": "read", "acquisition": "imported_wav" if wav_dir else "task3_microphone",
                     "noise": noise, "noise_note": detail, "human_reference": reference,
                     "reference_method": "speaker_listenback_before_asr", "reference_verified_utc": verified_time,
                     "first_asr": recognized, "participant_corrected_text": corrected, "saved_utc": now()}
            write_new(stage / "entry.json", entry)
            stage.rename(speaker_dir / identity)
    return {"speaker_id": speaker, "new_clips": len(todo)}


def normalize(text):
    return words(text.replace("’", "'"))


def critical_tokens(tokens, allowed):
    digits = dict(zip("123456789", "one two three four five six seven eight nine".split()))
    digits["10"] = "ten"
    normalized = [digits.get(w, w) for w in tokens]
    return [w for w in normalized if w in allowed]


def measure(reference, raw_result):
    raw_text = raw_result["receipt"]["transcript"] if raw_result["receipt"] else ""
    ref, hyp = normalize(reference), normalize(raw_text)
    require(bool(ref), "verified reference may not be blank")
    terms = {}
    for category, vocabulary in CATEGORIES.items():
        allowed = set(vocabulary.split())
        actual, heard = critical_tokens(ref, allowed), critical_tokens(hyp, allowed)
        terms[category] = {"reference": actual, "asr": heard, "assessable": bool(actual or heard),
                           "exact_in_order": actual == heard}
    return {"reference_words": len(ref), "word_edits": edit_distance(ref, hyp), "word_exact": ref == hyp,
            "critical_terms": terms}


def summarize(rows, planned):
    refs = sum(r["reference_words"] for r in rows)
    edits = sum(r["word_edits"] for r in rows)
    categories = {}
    for key in CATEGORIES:
        values = [r["critical_terms"][key] for r in rows if r["critical_terms"][key]["assessable"]]
        matched = sum(item["exact_in_order"] for item in values)
        categories[key] = {"assessable_clips": len(values), "preserved_in_order": matched,
                           "exact_rate": matched / len(values) if values else None}
    return {"planned": planned, "recorded": len(rows), "missing": planned - len(rows),
            "recognizer_errors": sum(r["scoring_asr"]["error"] is not None for r in rows),
            "reference_words": refs, "word_edits": edits, "word_error_rate": edits / refs if refs else None,
            "word_exact_clips": sum(r["word_exact"] for r in rows),
            "recording_conditions": dict(Counter(r["noise"] for r in rows)), "critical_terms": categories}


def displayed_asr_report(rows, per_speaker):
    """Score the first output the participant could correct, independently of the later re-decode."""
    displayed, differences = [], 0
    for row in rows:
        first = row["first_asr"]
        if first is None:
            continue
        repeat = row["scoring_asr"]
        first_text = first["receipt"]["transcript"] if first["receipt"] else None
        repeat_text = repeat["receipt"]["transcript"] if repeat["receipt"] else None
        differences += first_text != repeat_text
        measured = measure(row["human_reference"], first)
        row["displayed_asr_scores"] = measured
        displayed.append({"speaker_id": row["speaker_id"], "noise": row["noise"], "scoring_asr": first, **measured})
    return {"overall": summarize(displayed, len(SPEAKERS) * per_speaker),
            "by_speaker": {s: summarize([r for r in displayed if r["speaker_id"] == s], per_speaker) for s in SPEAKERS},
            "collected_without_initial_asr": len(rows) - len(displayed), "different_text_or_success_on_replay": differences}


def check_entry(entry, raw, spec, p, speaker):
    require(entry.get("source") == "human" and not entry.get("software_fixture") and entry.get("speaker_consented") is True
            and entry.get("consent_statement") == CONSENT and entry.get("speaker_id") == speaker
            and entry.get("prompt_id") == p["prompt_id"], "missing consent, wrong speaker/prompt or nonhuman source")
    require(entry.get("version") == 1 and entry.get("protocol_sha256") == sha(encoded(spec))
            and entry.get("prompt_sha256") == sha(encoded(p)) and entry.get("audio_sha256") == sha(raw), "clip or prompt binding differs")
    require(entry.get("elicitation") == "read" and entry.get("noise") in NOISE and isinstance(entry.get("noise_note"), str)
            and len(entry["noise_note"]) <= 300
            and (entry["noise"] == "quiet" or entry["noise_note"].strip())
            and entry.get("acquisition") in ("task3_microphone", "imported_wav"), "invalid recording condition")
    require(entry.get("reference_method") == "speaker_listenback_before_asr"
            and isinstance(entry.get("human_reference"), str) and len(entry["human_reference"]) <= 2000
            and normalize(entry["human_reference"]), "actual listen-back reference missing")
    times = [datetime.fromisoformat(entry[k].replace("Z", "+00:00")) for k in ("consented_utc", "reference_verified_utc", "saved_utc")]
    require(all(t.tzinfo is not None for t in times) and times == sorted(times), "invalid consent/reference chronology")
    first, corrected = entry["first_asr"], entry["participant_corrected_text"]
    require(corrected is None or (isinstance(corrected, str) and corrected.strip() and len(corrected) <= 2000),
            "invalid corrected transcript")
    if first is None:
        require(corrected is None, "a participant correction must be bound to the original ASR attempt")
    else:
        receipt, error = first["receipt"], first["error"]
        require((receipt is None) != (error is None), "initial ASR receipt or error required")
        if receipt is not None:
            require(receipt["audio_sha256"] == sha(raw) and receipt["model_content_sha256"] == MODEL_CONTENT_SHA256
                    and receipt["model_id"] == MODEL_ID and isinstance(receipt["transcript"], str), "initial ASR binding differs")
        else:
            require(isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str),
                    "invalid initial recognition error")


def score(root, model=None, *, whisper_model=None):
    require(model is None or whisper_model is None, "choose one replay recognizer per result")
    model_dir, recognizer, model_id, model_hash = model, LocalSpeechRecognizer, MODEL_ID, MODEL_CONTENT_SHA256
    if whisper_model is not None:
        from trackrag import whisper_speech
        model_dir, recognizer = whisper_model, whisper_speech.WhisperSpeechRecognizer
        model_id, model_hash = whisper_speech.MODEL_ID, whisper_speech.MODEL_CONTENT_SHA256
    spec = check_protocol(root)
    path = root / "recordings"
    require(all(p.is_dir() and p.name in SPEAKERS for p in path.iterdir()), "unexpected speaker directories")
    rows, seen, asr = [], set(), None
    for speaker in SPEAKERS:
        folder = path / speaker
        if not folder.exists():
            continue
        known = {p["prompt_id"] for p in spec["prompts"]}
        require(all(p.name in known or p.name.startswith(".pending-") for p in folder.iterdir()),
                "a take in this speaker folder is outside the frozen protocol")
        for p in spec["prompts"]:
            slot = folder / p["prompt_id"]
            if not slot.exists():
                continue
            raw = read_audio(slot / "audio.wav")
            entry_path = slot / "entry.json"
            require(entry_path.is_file() and entry_path.stat().st_size <= 64 * 1024, "missing or oversized clip entry")
            entry_bytes = entry_path.read_bytes()
            entry = json.loads(entry_bytes)
            check_entry(entry, raw, spec, p, speaker)
            require(sha(raw) not in seen, "same WAV cannot count twice or represent two speakers")
            seen.add(sha(raw))
            if asr is None:
                require(model_dir is not None, "provide the pinned offline --model-dir or --whisper-model when genuine clips exist")
                asr = recognizer(model_dir)
            attempt = decode(asr, raw, expected_model=model_id, expected_digest=model_hash)
            rows.append({"speaker_id": speaker, "prompt_id": p["prompt_id"], "source_case_id": p["source_case_id"],
                         "frame_id": p["frame_id"], "recording_id": p["recording_id"], "image_sha256": p["image_sha256"],
                         "audio_sha256": sha(raw), "entry_sha256": sha(entry_bytes), "noise": entry["noise"],
                         "noise_note": entry["noise_note"], "acquisition": entry["acquisition"],
                         "script": p["script"], "human_reference": entry["human_reference"], "script_read_exactly":
                         normalize(p["script"]) == normalize(entry["human_reference"]), "first_asr": entry["first_asr"],
                         "participant_corrected_text": entry["participant_corrected_text"], "scoring_asr": attempt,
                         **measure(entry["human_reference"], attempt)})
    by_speaker = {s: summarize([r for r in rows if r["speaker_id"] == s], len(spec["prompts"])) for s in SPEAKERS}
    missing = {s: [p["prompt_id"] for p in spec["prompts"] if not any(r["speaker_id"] == s and r["prompt_id"] == p["prompt_id"]
                                                                       for r in rows)] for s in SPEAKERS}
    displayed = displayed_asr_report(rows, len(spec["prompts"]))
    kind = "human_passenger_speech_study_whisper_supplement" if whisper_model is not None else "human_passenger_speech_study"
    return {"version": 1, "kind": kind, "scored_utc": now(), "protocol_sha256": sha(encoded(spec)),
            "elicitation": "read", "verified_reference": "participant listened before recognition and self-attested",
            "model_id": model_id, "model_content_sha256": model_hash, "initial_display_model_id": MODEL_ID,
            "intended_human_speakers": len(SPEAKERS),
            "speakers_with_returns": sum(v["recorded"] > 0 for v in by_speaker.values()), "by_speaker": by_speaker,
            "overall": summarize(rows, len(SPEAKERS) * len(spec["prompts"])), "missing_slots": missing,
            "original_displayed_asr": displayed,
            "scoring_basis": "The overall and per-speaker groups use the named replay model. Original displayed ASR always describes the collector's Vosk output when supplied. Both use the spoken reference. A failed recognition counts as an empty transcript.",
            "private_clip_results": rows}


def synthetic_trial(wav_dir, model, *, generate=False):
    spec = protocol()
    prompts = {p["prompt_id"]: p for p in spec["prompts"]}
    if generate:
        synthesize(wav_dir, [("passenger-" + code, p["script"], ()) for code, p in prompts.items()])
    asr = LocalSpeechRecognizer(model)
    rows = []
    for speaker in ("david", "zira"):
        for identity in prompts:
            p = prompts[identity]
            raw = read_audio(wav_dir / f"passenger-{identity}-{speaker}.wav")
            attempt = decode(asr, raw)
            rows.append({"synthetic_voice": speaker, "prompt_id": identity, "audio_sha256": sha(raw),
                         "scripted_reference": p["script"], "noise": "quiet", "scoring_asr": attempt,
                         **measure(p["script"], attempt)})
    return {"kind": "synthetic_windows_speech_scorer_verification", "created_utc": now(), "protocol_sha256": sha(encoded(spec)),
            "model_id": MODEL_ID, "actual_human_speakers": 0, "actual_human_clips": 0,
            "overall": summarize(rows, len(prompts) * 2),
            "by_synthetic_voice": {s: summarize([r for r in rows if r["synthetic_voice"] == s], len(prompts))
                                   for s in ("david", "zira")},
            "synthetic_results": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    initial = actions.add_parser("prepare")
    initial.add_argument("--out", type=Path, required=True)
    gather = actions.add_parser("collect")
    gather.add_argument("--study", type=Path, required=True)
    gather.add_argument("--speaker", choices=SPEAKERS, required=True)
    gather.add_argument("--noise", choices=NOISE, required=True)
    gather.add_argument("--model-dir", type=Path, help="Pinned Vosk directory to offer immediate transcript correction")
    gather.add_argument("--wav-dir", type=Path, help="Import the participant's own <prompt_id>.wav files instead of using a microphone")
    gather.add_argument("--prompt", help="Collect a single prompt, otherwise resume the ten unfinished prompts")
    gather.add_argument("--seconds", type=int, default=8)
    for name in ("score", "synthetic-trial"):
        cmd = actions.add_parser(name)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--model-dir", type=Path, required=name == "synthetic-trial")
        if name == "score":
            cmd.add_argument("--study", type=Path, required=True)
            cmd.add_argument("--whisper-model", type=Path, help="Score the same consented original WAVs with verified offline Whisper Small")
        else:
            cmd.add_argument("--wav-dir", type=Path, required=True)
            cmd.add_argument("--generate", action="store_true", help="Generate both Windows synthetic voices into a fresh audio folder")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        print(json.dumps(prepare(args.out), indent=2))
    elif args.action == "collect":
        print(json.dumps(collect(args.study, args.speaker, args.noise, model=args.model_dir, wav_dir=args.wav_dir,
                                 single=args.prompt, seconds=args.seconds), indent=2))
    else:
        require(not args.out.exists(), "choose a new output JSON so earlier scores are preserved")
        result = (score(args.study, args.model_dir, whisper_model=args.whisper_model) if args.action == "score"
                  else synthetic_trial(args.wav_dir, args.model_dir, generate=args.generate))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_new(args.out, result)
        print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
