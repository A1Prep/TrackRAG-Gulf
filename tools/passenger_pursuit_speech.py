"""Freeze and score paired local speech recognition on original, manually transcribed in-car audio."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
import re
import wave
from xml.etree import ElementTree
from zipfile import ZipFile

from tools.dropoff_voice_trial import edit_distance
from tools.passenger_speech_study import decode, normalize, require, write_new
from trackrag import local_speech, whisper_speech


PLAN = Path(__file__).resolve().parents[1] / "experiments/dropoff/passenger-pursuit-speech-protocol-20261122.json"
PLAN_SHA = "7ac891569494a8e736ff1215c33a5803f8723e0d8b415d598f9f9ca13acf9282"
SELECTED_SHA = "b86050b3be698b2ee9662ad90d6e54849cfd9a607794ce7d076323a3ce15a03c"
NITE = "{http://nite.sourceforge.net/}"
ANNOTATIONS = "PURSUIT-1.0/Data/Annotation/"
SIGNALS = "PURSUIT-1.0/Data/Signal/"


def digest(data):
    return sha256(data).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def protocol():
    require(file_hash(PLAN) == PLAN_SHA, "Protocol changed after registration")
    return json.loads(PLAN.read_bytes())


def span(item):
    href = item.find(NITE + "child").attrib["href"]
    ids = re.fullmatch(r"[^#]+#id\(w_(\d+)\)(?:\.\.id\(w_(\d+)\))?", href)
    require(ids is not None, "Unexpected reference to original words")
    return int(ids[1]), int(ids[2] or ids[1])


def candidates(archive, recording, plan):
    def xml(suffix):
        return ElementTree.fromstring(archive.read(ANNOTATIONS + recording + "." + suffix + ".xml"))

    words = {int(w.attrib[NITE + "id"].removeprefix("w_")): w.attrib["the_word"] for w in xml("words")}
    locations = [span(loc) for loc in xml("locations")]
    directions = set(plan["direction_words"])
    rows = []
    for segment in xml("segments"):
        first, last = span(segment)
        if any(i not in words for i in range(first, last + 1)):
            continue
        text = " ".join(words[i] for i in range(first, last + 1))
        tokens = normalize(text)
        start, end = (float(segment.attrib[NITE + key]) for key in ("start", "end"))
        if not (5 <= len(tokens) <= 23 and 3 <= end - start <= 14) or any(c in text for c in "[]"):
            continue
        identity = segment.attrib[NITE + "id"]
        rows.append({"recording": recording, "segment": identity, "reference": text,
                     "reference_sha256": digest(text.encode()), "start_s": start, "end_s": end,
                     "first_source_word": first, "last_source_word": last,
                     "direction": bool(directions.intersection(tokens)),
                     "landmark": any(first <= a <= b <= last for a, b in locations),
                     "order_sha256": digest(f"{plan['seed']}|{recording}|{identity}".encode())})
    return sorted(rows, key=lambda item: item["order_sha256"])


def choose(cache):
    plan = protocol()
    archive = cache / Path(plan["annotation_url"]).name
    require(file_hash(archive) == plan["annotation_archive_sha256"], "Official word/utterance archive changed")
    selected, shortfalls = [], []
    with ZipFile(archive) as src:
        for recording in plan["recordings"]:
            eligible = candidates(src, recording, plan)
            picked = []
            for label in ("direction", "landmark"):
                row = next((r for r in eligible if r[label] and all(
                    r["end_s"] + .18 <= p["start_s"] - .18 or r["start_s"] - .18 >= p["end_s"] + .18
                    for p in picked)), None)
                if row is None:
                    shortfalls.append({"recording": recording, "slot": label})
                else:
                    picked.append(dict(row, slot=label))
            selected.extend(picked)
    out = cache / "selected-private.json"
    write_new(out, {"protocol_sha256": PLAN_SHA, "rows": selected, "shortfalls": shortfalls})
    require(file_hash(out) == SELECTED_SHA, "Selection differs from the cohort frozen before recognition")
    return {"selected": len(selected), "shortfalls": shortfalls, "annotation_selection_sha256": file_hash(out)}


def clip(reader, row):
    rate, width, channels = reader.getframerate(), reader.getsampwidth(), reader.getnchannels()
    require(width == 2 and channels in (1, 2) and 16000 <= rate <= 48000, "Source PCM format outside release support")
    lo = max(0, round((row["start_s"] - .18) * rate))
    hi = min(reader.getnframes(), round((row["end_s"] + .18) * rate))
    reader.setpos(lo)
    samples = reader.readframes(hi - lo)
    require(len(samples) == (hi - lo) * width * channels, "Incomplete source audio")
    destination = BytesIO()
    with wave.open(destination, "wb") as wav:
        wav.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        wav.writeframes(samples)
    return destination.getvalue(), {"source_first_frame": lo, "source_end_exclusive_frame": hi,
                                   "rate_hz": rate, "channels": channels, "bytes_per_sample": width}


def freeze(cache, selected_sha):
    plan = protocol()
    selection = cache / "selected-private.json"
    require(file_hash(selection) == selected_sha == SELECTED_SHA, "Source selection changed")
    selected = json.loads(selection.read_bytes())
    full = cache / Path(plan["audio_url"]).name
    require(full.stat().st_size == plan["audio_archive_bytes"], "Incomplete original audio archive")
    full_sha = file_hash(full)
    recordings = plan["recordings"]
    clips = cache / "clips"
    clips.mkdir(exist_ok=True)
    bindings = {}
    with ZipFile(full) as src, ZipFile(cache / Path(plan["annotation_url"]).name) as refs:
        for recording in recordings:
            for suffix in ("words", "segments", "locations"):
                name = ANNOTATIONS + recording + "." + suffix + ".xml"
                require(src.read(name) == refs.read(name), "Original audio and annotation distributions disagree")
            member = SIGNALS + recording + ".audio.wav"
            chosen = [r for r in selected["rows"] if r["recording"] == recording]
            with src.open(member) as signal, wave.open(signal) as reader:
                for row in sorted(chosen, key=lambda r: r["start_s"]):
                    raw, meta = clip(reader, row)
                    dest = clips / (row["segment"] + ".wav")
                    if dest.exists():
                        require(file_hash(dest) == digest(raw), "Existing raw audio differs")
                    else:
                        with dest.open("xb") as fp:
                            fp.write(raw)
                    bindings[row["segment"]] = dict(meta, wav_sha256=digest(raw), source_wav_member=member,
                                                     source_wav_crc32=f"{src.getinfo(member).CRC:08x}")
    rows = [dict(row, **bindings[row["segment"]]) for row in selected["rows"]]
    out = cache / "bound-private.json"
    write_new(out, {"protocol_sha256": PLAN_SHA, "source_audio_sha256": full_sha,
                    "selected_sha256": selected_sha, "shortfalls": selected["shortfalls"], "rows": rows})
    return {"selected": len(rows), "source_audio_sha256": full_sha, "clip_binding_sha256": file_hash(out)}


def ordered_matches(reference, prediction):
    """Longest common subsequence counts exact ordered direction tokens, including repeated words."""
    previous = [0] * (len(prediction) + 1)
    for expected in reference:
        current = [0]
        for idx, heard in enumerate(prediction, 1):
            current.append(previous[idx - 1] + 1 if expected == heard else max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def measures(reference, text, directions):
    words, heard = normalize(reference), normalize(text)
    original = [w for w in words if w in directions]
    predicted = [w for w in heard if w in directions]
    hits = ordered_matches(original, predicted)
    return {"reference_words": len(words), "word_edits": edit_distance(words, heard),
            "reference_directions": len(original), "retained_directions": hits,
            "extra_direction_words": len(predicted) - hits, "all_directions_exact": original == predicted,
            "word_exact": words == heard}


def read_bound(cache, expected):
    path = cache / "bound-private.json"
    require(file_hash(path) == expected, "Bound inputs changed after pre-ASR freeze")
    bank = json.loads(path.read_bytes())
    require(bank["protocol_sha256"] == PLAN_SHA and
            file_hash(cache / "selected-private.json") == bank["selected_sha256"] == SELECTED_SHA, "Source selection changed")
    return bank


def run(cache, expected, backend, model):
    selection = read_bound(cache, expected)
    protocol()
    output = cache / (backend + "-private.json")
    require(not output.exists(), "Keep the first result for every registered input")
    selected = whisper_speech if backend == "whisper" else local_speech
    engine = selected.WhisperSpeechRecognizer(model) if backend == "whisper" else selected.LocalSpeechRecognizer(model)
    rows = []
    for row in selection["rows"]:
        raw = (cache / "clips" / (row["segment"] + ".wav")).read_bytes()
        require(digest(raw) == row["wav_sha256"] and digest(row["reference"].encode()) == row["reference_sha256"],
                "Pre-ASR audio or original transcript changed")
        answer = decode(engine, raw, expected_model=selected.MODEL_ID, expected_digest=selected.MODEL_CONTENT_SHA256)
        rows.append({"segment": row["segment"], "wav_sha256": row["wav_sha256"], "output": answer})
    import psutil
    proc = psutil.Process()
    effective_cores = len(proc.cpu_affinity()) if hasattr(proc, "cpu_affinity") else None
    write_new(output, {"model_id": selected.MODEL_ID, "model_sha256": selected.MODEL_CONTENT_SHA256,
                       "effective_cpu_cores": effective_cores,
                       "protocol_sha256": PLAN_SHA, "bound_sha256": expected, "rows": rows})
    return {"backend": backend, "attempted": len(rows), "private_raw_receipts_sha256": file_hash(output)}


def summary(rows):
    total = Counter()
    for row in rows:
        total.update({k: v for k, v in row.items() if type(v) is int})
        total["word_exact_clips"] += int(row["word_exact"])
        total["exact_direction_sequences_with_direction"] += int(row["reference_directions"] > 0 and row["all_directions_exact"])
        total["clips_with_reference_direction"] += int(row["reference_directions"] > 0)
        total["errors"] += int(row["error"] is not None)
    return dict(total, clips=len(rows), wer=total["word_edits"] / total["reference_words"] if total["reference_words"] else None,
                direction_recall=total["retained_directions"] / total["reference_directions"] if total["reference_directions"] else None)


def report(cache, expected, out):
    plan, bank = protocol(), read_bound(cache, expected)
    systems, certificates = {}, {}
    for name, module in (("vosk", local_speech), ("whisper", whisper_speech)):
        source = cache / (name + "-private.json")
        private = json.loads(source.read_bytes())
        require(private["bound_sha256"] == expected and private["protocol_sha256"] == PLAN_SHA and
                private["model_sha256"] == module.MODEL_CONTENT_SHA256, "Wrong input, model or method")
        scores = []
        for entry, ref in zip(private["rows"], bank["rows"], strict=True):
            result = entry["output"]
            require(entry["segment"] == ref["segment"] and entry["wav_sha256"] == ref["wav_sha256"], "Wrong paired clip")
            receipt = result["receipt"]
            if receipt is not None:
                require(receipt["audio_sha256"] == ref["wav_sha256"] and receipt["model_content_sha256"] == module.MODEL_CONTENT_SHA256,
                        "Raw recognizer receipt differs")
            text = receipt["transcript"] if receipt else ""
            scores.append(dict(measures(ref["reference"], text, set(plan["direction_words"])),
                               recording=ref["recording"], slot=ref["slot"], segment=ref["segment"],
                               wav_sha256=ref["wav_sha256"], manual_transcript_sha256=ref["reference_sha256"],
                               returned_transcript_sha256=digest(text.encode()),
                               error=None if receipt else result["error"]["code"]))
        systems[name] = {"overall": summary(scores), "by_recording": {r: summary([s for s in scores if s["recording"] == r])
                         for r in plan["recordings"]}, "rows": scores}
        certificates[name] = {"raw_private_sha256": file_hash(source), "model_sha256": module.MODEL_CONTENT_SHA256,
                              "effective_cpu_cores": private["effective_cpu_cores"],
                              "model_id": module.MODEL_ID}
    data = {"kind": plan["kind"], "measured_utc": datetime.now(timezone.utc).isoformat(),
            "publisher": plan["publisher"], "source_archive_sha256": bank["source_audio_sha256"],
            "protocol_sha256": PLAN_SHA, "pre_asr_source_selection_sha256": bank["selected_sha256"],
            "pre_asr_audio_binding_sha256": expected, "requested_recordings": 13, "source_drives": 7,
            "unique_speakers_counted": None, "selection_shortfalls": bank["shortfalls"],
            "runtime": certificates, "systems": systems, "scope": plan["interpretation"]}
    write_new(out, data)
    return {k: v["overall"] for k, v in systems.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("choose", "freeze", "run", "report"):
        command = actions.add_parser(name)
        command.add_argument("--cache", type=Path, required=True)
        if name == "freeze":
            command.add_argument("--selection-sha", required=True)
        if name in ("run", "report"):
            command.add_argument("--bound-sha", required=True)
        if name == "run":
            command.add_argument("--backend", choices=("vosk", "whisper"), required=True)
            command.add_argument("--model-dir", type=Path, required=True)
        if name == "report":
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "choose":
        result = choose(args.cache)
    elif args.action == "freeze":
        result = freeze(args.cache, args.selection_sha)
    elif args.action == "run":
        import psutil
        proc = psutil.Process()
        if hasattr(proc, "cpu_affinity"):
            proc.cpu_affinity([proc.cpu_affinity()[-1]])
        result = run(args.cache, args.bound_sha, args.backend, args.model_dir)
    else:
        result = report(args.cache, args.bound_sha, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
