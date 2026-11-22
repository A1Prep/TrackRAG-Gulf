"""Compare one fixed Whisper Small CPU condition with Vosk on separate, frozen SLURP speakers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from tools import passenger_external_speech as external
from tools.passenger_speech_study import decode, measure, normalize, require, write_new
from trackrag.local_speech import LocalSpeechRecognizer, MODEL_ID as VOSK_ID, MODEL_CONTENT_SHA256 as VOSK_SHA
from trackrag.whisper_speech import WhisperSpeechRecognizer, MODEL_ID as WHISPER_ID, MODEL_CONTENT_SHA256 as WHISPER_SHA, OPTIONS


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/dropoff/passenger-asr-comparison-protocol-20261122.json"
PLAN_SHA = "11dfd3827b4a08b8c330b27937cd785f74d80e682e0392f841b0cf70a41d24a5"
ORIGINAL = ROOT / "experiments/dropoff/passenger-public-human-speech-20261122.json"
FRESH = "asr-fresh-selection.json"


def protocol():
    raw = PLAN.read_bytes().replace(b"\r\n", b"\n")
    require(external.sha(raw) == PLAN_SHA, "comparator protocol differs from freeze")
    plan = json.loads(raw)
    require(all(plan["candidate"][key] == value for key,value in OPTIONS.items()), "candidate settings differ from freeze")
    return plan


def original(cache, plan):
    public = ORIGINAL.read_bytes().replace(b"\r\n", b"\n")
    require(external.sha(public) == plan["comparative_development"]["original_receipt_sha256"], "original Vosk results changed")
    original_rows = json.loads((cache / "selection.json").read_bytes())["rows"]
    rows = json.loads(public)["rows"]
    require(len(original_rows) == len(rows) and all(
        src["filename"] == row["real_filename"] and src["flac_sha256"] == row["real_flac_sha256"]
        and src["converted_wav_sha256"] == row["recognizer_wav_sha256"] and external.sha(src["reference"].encode()) == row["source_sentence_sha256"]
        for src,row in zip(original_rows,rows)), "original development utterances or audio differ from the published run")
    return original_rows, rows


def fresh(cache, flac):
    plan = protocol()
    prior, _ = original(cache, plan)
    return external.choose(cache, flac, selection_name=FRESH, quotas=plan["fresh_external_cohort"]["quotas"],
                           exclude=prior, protocol_sha=PLAN_SHA)


def partitions(cache, sha):
    plan = protocol()
    prior, baseline = original(cache, plan)
    raw = (cache / FRESH).read_bytes()
    require(external.sha(raw) == sha, "fresh panel changed after freeze")
    frozen = json.loads(raw)
    require(frozen["protocol_sha256"] == PLAN_SHA and frozen["original_archive_prefix_sha256"] == external.PREFIX_SHA,
            "fresh panel cites the wrong source or protocol")
    later = frozen["rows"]
    combined = prior + later
    require(frozen["published_text_sha256"] == external.TEXT_SHA
            and all(external.file_hash(cache / name) == expected for name, expected in external.TEXT_SHA.items()),
            "original transcript or speaker metadata changed")
    source = external.metadata(cache, {r["intent"] for r in combined})
    for row in combined:
        official = source.get(row["filename"])
        require(official is not None and all(row[k] == official[k] for k in
                ("slurp_id", "speaker", "reference", "places", "intent", "source_split")),
                "cohort differs from the official recording, place, sentence or speaker metadata")
    require(len(combined) == len({r["speaker"] for r in combined}) == len({tuple(normalize(r["reference"])) for r in combined})
            == len({r["flac_sha256"] for r in combined}), "speakers, texts or audio overlap in either panel")
    return {"comparative_development": prior, "fresh_external": later}, baseline


def run(cache, backend, model, frozen_sha):
    groups, _ = partitions(cache, frozen_sha)
    out = cache / (backend + "-private-asr-comparison.json")
    require(not out.exists(), "prior model transcripts already exist; preserve the recorded condition")
    start = perf_counter()
    engine = WhisperSpeechRecognizer(model) if backend == "whisper" else LocalSpeechRecognizer(model)
    load_s = perf_counter() - start
    model_id, model_hash = (WHISPER_ID, WHISPER_SHA) if backend == "whisper" else (VOSK_ID, VOSK_SHA)
    outputs = {}
    for group, selected in groups.items():
        if backend == "vosk" and group == "comparative_development":
            continue  # Immutable original run is the paired baseline for this group.
        rows = []
        for slot, src in enumerate(selected, 1):
            path = cache / "selected-local-audio" / src["filename"]
            if src["conversion_error"]:
                receipt = {"receipt":None, "error":{"code":"flac_decoder_failed"}}
            else:
                raw = path.with_suffix(".wav").read_bytes()
                require(external.file_hash(path) == src["flac_sha256"] and external.sha(raw) == src["converted_wav_sha256"],
                        "audio changed after the pre-ASR selection")
                receipt = decode(engine, raw, expected_model=model_id, expected_digest=model_hash)
            text = receipt["receipt"]["transcript"] if receipt["receipt"] else ""
            measured = measure(src["reference"], receipt)
            recovered = external.place_matches(src["places"], text)
            rows.append({"speaker_slot": slot, "real_filename": src["filename"], "intent": src["intent"], "microphone": src["microphone"],
                         "slurp_id": src["slurp_id"], "recognizer_wav_sha256": src["converted_wav_sha256"],
                         "source_sentence_sha256": external.sha(src["reference"].encode()), "raw_transcript_sha256": external.sha(text.encode()),
                         "reference_words": measured["reference_words"], "word_edits": measured["word_edits"], "word_exact": measured["word_exact"],
                         "ordered_place_mentions_recovered": sum(recovered), "place_mentions": len(recovered),
                         "decode_s": receipt["receipt"]["elapsed_s"] if receipt["receipt"] else None,
                         "duration_s": receipt["receipt"]["duration_s"] if receipt["receipt"] else None,
                         "error": receipt["error"]["code"] if receipt["error"] else None, "private_asr": receipt})
        outputs[group] = rows
    peak = None
    try:
        import psutil
        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except ImportError:
        pass
    write_new(out, {"backend": backend, "model_id":model_id, "model_content_sha256":model_hash, "started_after_freeze": True,
                    "finished_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": PLAN_SHA, "fresh_selection_sha256": frozen_sha,
                    "load_s":load_s, "windows_peak_process_working_set_bytes":peak, "outputs":outputs})
    return {g: external.aggregate(rows) for g,rows in outputs.items()}


def metrics(rows):
    times = sorted(r["decode_s"] for r in rows if r["decode_s"] is not None)
    result = external.aggregate(rows)
    if times:
        from math import ceil
        result.update(median_decode_s=median(times), p95_decode_s=times[ceil(.95*len(times))-1], sum_decode_s=sum(times))
    return result


def compare(left, right):
    require(len(left) == len(right), "conditions returned different cohort sizes")
    paired = []
    for prev, current in zip(left, right):
        require(all(prev[k] == current[k] for k in ("real_filename", "recognizer_wav_sha256", "source_sentence_sha256",
                    "reference_words", "place_mentions", "slurp_id", "intent")), "paired conditions refer to different input or reference")
        paired.append({"filename": prev["real_filename"], "slurp_id":prev["slurp_id"],"intent":prev["intent"],"microphone":prev["microphone"],
                       "reference_words":prev["reference_words"],"audio_sha256":prev["recognizer_wav_sha256"],
                       "vosk_word_edits":prev["word_edits"],"whisper_word_edits":current["word_edits"],
                       "place_mentions":prev["place_mentions"],"vosk_places":prev["ordered_place_mentions_recovered"],
                       "whisper_places":current["ordered_place_mentions_recovered"],
                       "vosk_decode_s":prev["decode_s"],"whisper_decode_s":current["decode_s"],
                       "vosk_error":prev["error"],"whisper_error":current["error"],
                       "vosk_transcript_sha256":prev["raw_transcript_sha256"],"whisper_transcript_sha256":current["raw_transcript_sha256"]})
    return {"vosk":metrics(left), "whisper":metrics(right),
            "fewer_word_edits":sum(r["whisper_word_edits"] < r["vosk_word_edits"] for r in paired),
            "equal_word_edits":sum(r["whisper_word_edits"] == r["vosk_word_edits"] for r in paired),
            "more_word_edits":sum(r["whisper_word_edits"] > r["vosk_word_edits"] for r in paired),
            "per_intent":{k:{"vosk":metrics([r for r in left if r["intent"] == k]),"whisper":metrics([r for r in right if r["intent"] == k])}
                          for k in sorted({r["intent"] for r in left})}, "paired":paired}


def report(cache, frozen_sha, out):
    groups, prior = partitions(cache, frozen_sha)
    studies = {k:json.loads((cache / (k+"-private-asr-comparison.json")).read_bytes()) for k in ("vosk","whisper")}
    require(all(s["protocol_sha256"] == PLAN_SHA and s["fresh_selection_sha256"] == frozen_sha for s in studies.values()),
            "outcomes are from different freezes")
    data = {}
    left = {"comparative_development":prior, **studies["vosk"]["outputs"]}
    for name, selected in groups.items():
        right = studies["whisper"]["outputs"][name]
        require([r["filename"] for r in selected] == [r["real_filename"] for r in right], "comparison omitted or reordered clips")
        data[name] = compare(left[name], right)
    result = {"kind":"preregistered_paired_external_transport_asr","created_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":PLAN_SHA,
              "fresh_selection_sha256":frozen_sha,"distinct_speakers_over_both_groups":sum(len(v) for v in groups.values()),
              "source":"SLURP, Bastianelli et al. (2020). Audio CC BY-NC 4.0, text CC BY 4.0. https://zenodo.org/records/4274930",
              "team_speech_clips_contributed":0, "cpu_only":True, "runtime":{k:{key:s[key] for key in ("model_id","model_content_sha256","load_s","windows_peak_process_working_set_bytes")}
              for k,s in studies.items()},"groups":data}
    write_new(out,result)
    return {k:{backend:v[backend] for backend in ("vosk","whisper")} for k,v in data.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest="action",required=True)
    for name in ("freeze","run","report"):
        a=actions.add_parser(name)
        a.add_argument("--cache",type=Path,required=True)
        if name=="freeze": a.add_argument("--flac",type=Path,required=True)
        else: a.add_argument("--fresh-sha",required=True)
        if name=="run":
            a.add_argument("--backend",choices=("vosk","whisper"),required=True)
            a.add_argument("--model-dir",type=Path,required=True)
        if name=="report": a.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.action=="freeze": result=fresh(args.cache,args.flac)
    elif args.action=="run": result=run(args.cache,args.backend,args.model_dir,args.fresh_sha)
    else: result=report(args.cache,args.fresh_sha,args.out)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
