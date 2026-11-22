"""Freeze then score a small public SLURP human-speech diagnostic with the passenger recognizer."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import file_digest, sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time
import zlib

from tools.passenger_speech_study import decode, measure, normalize, require, write_new
from trackrag.local_speech import LocalSpeechRecognizer


PROTOCOL = Path(__file__).resolve().parents[1] / "experiments/dropoff/passenger-public-speech-protocol-20261122.json"
PROTOCOL_SHA = "b04ecd6cf04dd5e7f8d4cd6134bc60330fa90643941075cd57ba2fc2b37510fc"
PREFIX = "original-first-160m.tar.gz.prefix"
PREFIX_SHA = "924da6ef738c7fcb1ee123256e2f6bdb56490c914b456cb2797d223d8a58b1b7"
TEXT_SHA = {"devel.jsonl": "00d346b183ecc2d160853cd3dc20e8ccd207d9e1809cf6e2c05182076bd0ecc9",
            "metadata.json": "be7943d576c0b79b34ae32796e5c9474bd3d4d123bb35e38f73497f4a429fb83",
            "test.jsonl": "fe3449af69b42fda7163345482556066c35a80f7e552a6d67e212a0e2f0783cc",
            "train.jsonl": "4a05a3de6c7f028aff298c4a90196e6a3fecd1938d96fb1c125d6fd10d5e301e"}


def sha(data):
    return sha256(data).hexdigest()


def file_hash(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def fixed_protocol():
    raw = PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
    require(sha(raw) == PROTOCOL_SHA, "external evaluation protocol changed")
    return json.loads(raw)


def fetch(cache):
    """Resume just the pinned archive prefix. Verify exact HTTP ranges before appending."""
    import requests
    plan = fixed_protocol()
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in TEXT_SHA.items():
        path = cache / name
        if path.exists():
            require(file_hash(path) == expected, "cached SLURP text differs from the pinned Git revision")
            continue
        base = "https://raw.githubusercontent.com/pswietojanski/slurp/" + plan["text_revision"] + "/dataset/slurp/"
        response = requests.get(base + name, timeout=(6,45))
        response.raise_for_status()
        require(sha(response.content) == expected, "downloaded SLURP text differs from the pinned Git revision")
        path.write_bytes(response.content)
    output = cache / PREFIX
    offset = output.stat().st_size if output.exists() else 0
    end = plan["archive_prefix_bytes"]
    require(offset <= end, "cached archive exceeds the preregistered prefix")
    url = "https://zenodo.org/records/4274930/files/slurp_real.tar.gz"
    def chunk(lo):
        hi = min(lo + 2 * 1024**2, end) - 1
        for attempt in range(3):
            try:
                response = requests.get(url, params={"download":"1","trackrag_span":f"{lo}-{hi}"},
                                        headers={"Range": f"bytes={lo}-{hi}"}, timeout=(6,28))
                response.raise_for_status()
                require(response.status_code == 206 and response.headers.get("Content-Range") == f"bytes {lo}-{hi}/3918185662"
                        and len(response.content) == hi - lo + 1, "Zenodo did not serve the requested original byte range")
                return response.content
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    # Append only completed contiguous ranges; a stopped transfer can safely resume at the last byte.
    with ThreadPoolExecutor(max_workers=6) as workers, output.open("ab") as stream:
        for part in workers.map(chunk, range(offset, end, 2 * 1024**2)):
            stream.write(part)
            stream.flush()
    require(file_hash(output) == PREFIX_SHA, "archived byte prefix differs from the original pre-ASR source")
    return {"archive_bytes": output.stat().st_size, "prefix_sha256": PREFIX_SHA}


def metadata(cache, quotas):
    require(all(file_hash(cache / name) == expected for name,expected in TEXT_SHA.items()), "published SLURP text differs")
    table = json.loads((cache / "metadata.json").read_bytes())
    index = {}
    for split in ("train", "devel", "test"):
        for row in map(json.loads, (cache / (split + ".jsonl")).read_bytes().splitlines()):
            if row["scenario"] != "transport" or row["intent"] not in quotas:
                continue
            entities = sorted((e for e in row["entities"] if e["type"] == "place_name"), key=lambda e: min(e["span"]))
            places = [" ".join(row["tokens"][i]["surface"] for i in e["span"]) for e in entities]
            if not places:
                continue
            details = table[str(row["slurp_id"])]["recordings"]
            for recording in row["recordings"]:
                name = recording["file"]
                user = details[name]["usrid"].rsplit("-", 1)[-1]
                if not user.isdigit():
                    continue
                require(name not in index, "recording maps to multiple source rows")
                index[name] = {"source_split": split, "slurp_id": row["slurp_id"], "intent": row["intent"],
                               "speaker": user, "reference": row["sentence"], "places": places}
    return index


def choose(cache, flac, *, selection_name="selection.json", quotas=None, exclude=(), protocol_sha=PROTOCOL_SHA):
    plan = fixed_protocol()
    quotas = plan["quotas"] if quotas is None else quotas
    out = cache / selection_name
    require(not out.exists(), "the external study already has a frozen selection")
    archive = cache / PREFIX
    require(archive.stat().st_size == plan["archive_prefix_bytes"] and file_hash(archive) == PREFIX_SHA,
            "download the exact preregistered prefix")
    lookup = metadata(cache, quotas)
    source_dir = cache / "selected-local-audio"
    source_dir.mkdir(exist_ok=True)
    selected, counts = [], Counter()
    people = {r["speaker"] for r in exclude}
    sentences = {tuple(normalize(r["reference"])) for r in exclude}
    try:
        with tarfile.open(archive, "r|gz") as bundle:
            for member_idx, member in enumerate(bundle):
                name = PurePosixPath(member.name).name
                if not member.isfile() or name not in lookup:
                    continue
                r = lookup[name]
                terms = tuple(normalize(r["reference"]))
                if r["speaker"] in people or terms in sentences or counts[r["intent"]] >= quotas[r["intent"]]:
                    continue
                require(0 < member.size < 12 * 1024**2 and "slurp_real" in PurePosixPath(member.name).parts,
                        "unexpected original SLURP member")
                raw = bundle.extractfile(member).read()
                require(len(raw) == member.size and raw.startswith(b"fLaC"), "incomplete original FLAC")
                path = source_dir / name
                path.write_bytes(raw)
                converted = path.with_suffix(".wav")
                run = subprocess.run([str(flac), "--silent", "--decode", "--force", "--output-name=" + str(converted), str(path)],
                                     capture_output=True, timeout=20)
                selected.append({**r, "filename": name, "archive_member_index": member_idx, "flac_sha256": sha(raw),
                                 "microphone": "close_headset" if name.endswith("-headset.flac") else "distant",
                                 "converted_wav_sha256": file_hash(converted) if run.returncode == 0 else None,
                                 "conversion_error": None if run.returncode == 0 else "flac_decoder_failed"})
                people.add(r["speaker"])
                sentences.add(terms)
                counts[r["intent"]] += 1
                if all(counts[k] == n for k,n in quotas.items()):
                    break
    except (EOFError, tarfile.ReadError, zlib.error) as exc:
        # A fixed byte ceiling can cut the following member. Never skip a corrupt earlier region.
        if "end" not in str(exc).lower() and "truncated" not in str(exc).lower():
            raise
    local = {"protocol_sha256": protocol_sha, "selected_before_asr_utc": datetime.now(timezone.utc).isoformat(),
             "original_archive_prefix_sha256": file_hash(archive), "archive_prefix_bytes": archive.stat().st_size,
             "published_text_sha256": {n: file_hash(cache / n) for n in ("train.jsonl","devel.jsonl","test.jsonl","metadata.json")},
             "counts": dict(counts), "rows": selected}
    write_new(out, local)
    return {"selected": len(selected), "by_intent": dict(counts), "speakers": len({r["speaker"] for r in selected}),
            "selection_sha256": file_hash(out)}


def place_matches(places, transcript):
    hyp = normalize(transcript)
    found, cursor = [], 0
    for place in places:
        ref = normalize(place)
        require(bool(ref), "empty annotated destination span")
        index = next((i for i in range(cursor, len(hyp) - len(ref) + 1) if hyp[i:i+len(ref)] == ref), None)
        found.append(index is not None)
        if index is not None:
            cursor = index + len(ref)
    return found


def aggregate(rows):
    words = sum(r["reference_words"] for r in rows)
    edits = sum(r["word_edits"] for r in rows)
    places = sum(r["place_mentions"] for r in rows)
    recovered = sum(r["ordered_place_mentions_recovered"] for r in rows)
    return {"clips": len(rows), "speakers": len({r["speaker_slot"] for r in rows}), "recognition_errors": sum(bool(r["error"]) for r in rows),
            "reference_words": words, "word_edits": edits, "word_error_rate": edits / words if words else None,
            "exact_utterances": sum(r["word_exact"] for r in rows), "place_mentions": places,
            "ordered_place_mentions_recovered": recovered, "place_mention_recall": recovered / places if places else None,
            "clips_with_all_places": sum(r["ordered_place_mentions_recovered"] == r["place_mentions"] for r in rows)}


def score(cache, model, out, selection_sha):
    plan = fixed_protocol()
    require(not out.exists() and not (cache / "private-transcripts.json").exists(), "use a fresh scoring output")
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_selection = (cache / "selection.json").read_bytes()
    require(sha(raw_selection) == selection_sha, "private selection differs from the pre-ASR freeze")
    frozen = json.loads(raw_selection)
    require(frozen["protocol_sha256"] == PROTOCOL_SHA, "selection is from another protocol")
    require(frozen["published_text_sha256"] == TEXT_SHA and file_hash(cache / PREFIX) == frozen["original_archive_prefix_sha256"],
            "selection does not match the archived and published sources")
    require(len({r["speaker"] for r in frozen["rows"]}) == len(frozen["rows"]) > 0, "needs an independent, nonempty cohort")
    engine = LocalSpeechRecognizer(model)
    rows, private = [], []
    for slot, r in enumerate(frozen["rows"], 1):
        flac = cache / "selected-local-audio" / r["filename"]
        require(file_hash(flac) == r["flac_sha256"], "original audio changed since selection")
        if r["conversion_error"]:
            result = {"receipt": None, "error": {"code": r["conversion_error"], "message": "conversion failed"}}
        else:
            raw = flac.with_suffix(".wav").read_bytes()
            require(sha(raw) == r["converted_wav_sha256"], "WAV changed since selection")
            result = decode(engine, raw)
        text = result["receipt"]["transcript"] if result["receipt"] else ""
        measured = measure(r["reference"], result)
        terms = place_matches(r["places"], text)
        private.append({"source_file": r["filename"], "reference": r["reference"], "asr": result, "places": r["places"],
                        "places_recovered": terms})
        rows.append({"slurp_id": r["slurp_id"], "real_filename": r["filename"], "speaker_slot": slot,
                     "source_split": r["source_split"], "intent": r["intent"], "microphone": r["microphone"],
                     "real_flac_sha256": r["flac_sha256"], "recognizer_wav_sha256": r["converted_wav_sha256"],
                     "source_sentence_sha256": sha(r["reference"].encode()), "raw_transcript_sha256": sha(text.encode()),
                     "reference_words": measured["reference_words"], "word_edits": measured["word_edits"], "word_exact": measured["word_exact"],
                     "place_mentions": len(terms), "ordered_place_mentions_recovered": sum(terms),
                     "duration_s": result["receipt"]["duration_s"] if result["receipt"] else None,
                     "decode_s": result["receipt"]["elapsed_s"] if result["receipt"] else None,
                     "error": result["error"]["code"] if result["error"] else None})
    report = {"kind": "external_slurp_real_english_transport_diagnostic", "scored_utc": datetime.now(timezone.utc).isoformat(),
              "team_speech_study_clips_contributed": 0, "original_audio_uploaded_or_in_git": False,
              "protocol_sha256": PROTOCOL_SHA, "private_selection_sha256": file_hash(cache / "selection.json"),
              "selection_frozen_utc": frozen["selected_before_asr_utc"], "archive_prefix_sha256": frozen["original_archive_prefix_sha256"],
              "archive_prefix_bytes": frozen["archive_prefix_bytes"], "source_text_sha256": frozen["published_text_sha256"],
              "source_attribution": plan["dataset"], "audio_source": plan["audio_source"], "research_paper": plan["research_paper"],
              "audio_license": plan["input_licenses"]["real_audio"], "text_license": plan["input_licenses"]["text"], "text_revision": plan["text_revision"],
              "model": plan["asr"], "overall": aggregate(rows), "unfilled_slots": {k: n - sum(r["intent"] == k for r in rows)
              for k,n in plan["quotas"].items()}, "by_intent": {k: aggregate([r for r in rows if r["intent"] == k]) for k in plan["quotas"]},
              "by_microphone": {k: aggregate([r for r in rows if r["microphone"] == k]) for k in ("close_headset","distant")},
              "rows": rows, "limits": ["Small, prospectively bounded archive-order convenience sample.",
              "SLURP assistant transport prompts describe city travel and bookings. They have no paired EMT dashcam image or Gulf landmark task.",
              "Reference texts are the released prompts. Actual speech can depart from those words.",
              "This external check supplies no volunteer consent or recordings for the local three-speaker study."]}
    write_new(cache / "private-transcripts.json", private)
    write_new(out, report)
    return report["overall"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    a = actions.add_parser("fetch")
    a.add_argument("--cache", type=Path, required=True)
    a = actions.add_parser("select")
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--flac", type=Path, required=True)
    a = actions.add_parser("score")
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--model-dir", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--selection-sha", required=True, help="SHA256 printed by select before any ASR run")
    args = parser.parse_args()
    if args.action == "fetch":
        result = fetch(args.cache)
    elif args.action == "select":
        result = choose(args.cache, args.flac)
    else:
        result = score(args.cache, args.model_dir, args.out, args.selection_sha)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
