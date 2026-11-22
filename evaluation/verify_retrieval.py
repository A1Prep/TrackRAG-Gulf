"""Check retrieval contracts and temporal isolation; no answer model is called."""

import argparse
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .compare import _md
from .run_audit import audit_run, fingerprint
from .score import read_jsonl
from .temporal_checks import (
    FrameView, changed_future_observations, checked_frames, checked_observations,
    checked_ranking, replacement_frames, same_ranking,
)
from trackrag.__main__ import _adapter
from trackrag.emt import load_annotated_tracks, open_recording
from trackrag.questions import load_questions, validate_questions
from trackrag.review import _validate_questions, review_recording


def _error(error):
    return {"error_type": type(error).__name__, "detail": str(error)[:2000]}


def _ranking_kind(result):
    if "intervals" not in result:
        return "invalid"
    if not result["intervals"]:
        return "empty"
    if all(row["track_id"] is None and row["score"] == 0 for row in result["intervals"]):
        return "zero_score_scene_context"
    return "retrieved"


def _rank(question, index, rank_evidence, source, observations, frames):
    ranked = None
    try:
        supplied = deepcopy(question)
        ranked = rank_evidence(supplied, index, k=3)
        # Materialize before checking mutation: adapters may return a generator.
        ranked = list(ranked)
        if supplied != question:
            raise ValueError("Retriever modified the supplied question")
        return {"intervals": checked_ranking(ranked, question, source, observations, frames)}
    except Exception as error:
        failure = _error(error)
        if ranked is not None:
            # Invalid values (including NaN) cannot enter the JSON interval field.
            # Keep a bounded diagnostic rather than silently dropping the output.
            try:
                representation = repr(ranked)
                failure["returned_intervals_repr"] = representation[:8000]
                failure["returned_intervals_repr_truncated"] = len(representation) > 8000
            except Exception:
                failure["returned_intervals_repr"] = "<representation unavailable>"
        return failure


def _batch(recording, rows, questions, build_index, rank_evidence, source, original_rows, frames, *, repeat=False):
    try:
        index = build_index(recording, deepcopy(rows))
    except Exception as error:
        return {q["question_id"]: {"first": _error(error), "repeat": _error(error)} for q in questions}
    output = {}
    for question in questions:
        first = _rank(question, index, rank_evidence, source, original_rows, frames)
        second = _rank(question, index, rank_evidence, source, original_rows, frames) if repeat else first
        output[question["question_id"]] = {"first": first, "repeat": second}
    return output


def _comparison(expected, actual, tolerance, *, control=False):
    if "intervals" not in expected or "intervals" not in actual:
        return {"status": "inconclusive" if control else "failed", "reason": "Adapter error",
                "expected": expected, "actual": actual}
    equal = same_ranking(expected["intervals"], actual["intervals"], tolerance)
    if not equal:
        return {"status": "inconclusive" if control else "failed",
                "reason": "Control is unstable or wrapper-dependent" if control else "Raw intervals or scores changed",
                "expected": expected, "actual": actual}
    return {"status": "stable_empty" if not expected["intervals"] else "passed"}


def check_recording(recording, questions, observations, source, build_index, rank_evidence, workspace, tolerance=1e-6):
    """Falsification checks conditional on supplied observations, not a leakage proof."""
    if source not in {"annotated", "predicted"}:
        raise ValueError("Invalid track source")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Score tolerance must be finite and nonnegative")
    questions = deepcopy(list(questions))
    _validate_questions(recording, questions)
    if not questions:
        raise ValueError("No questions were supplied")
    frames = checked_frames(recording)
    rows = checked_observations(observations, frames, recording.recording_id, source)
    # Baseline receives complete ordinary inputs. Do not silently truncate it.
    original = _batch(recording, rows, questions, build_index, rank_evidence, source, rows, frames, repeat=True)
    rebuilt = _batch(recording, rows, questions, build_index, rank_evidence, source, rows, frames)
    last = frames[-1]["frame_id"]
    cases = {}
    for question in questions:
        qid = question["question_id"]
        base = original[qid]["first"]
        cases[qid] = {"question": question, "baseline": base, "ranking_kind": _ranking_kind(base), "checks": {
            "repeat_control": _comparison(base, original[qid]["repeat"], tolerance, control=True),
            "rebuild_control": _comparison(base, rebuilt[qid]["first"], tolerance, control=True),
        }}
        if "intervals" not in base:
            cases[qid]["checks"]["raw_contract"] = {"status": "failed", **base}
        else:
            cases[qid]["checks"]["raw_contract"] = {"status": "passed"}
        if question["query_frame"] == last:
            cases[qid]["checks"]["future_tests"] = {"status": "not_applicable", "reason": "No frames after this query"}
    groups = defaultdict(list)
    for question in questions:
        if question["query_frame"] < last:
            groups[question["query_frame"]].append(question)
    for cutoff, group in sorted(groups.items()):
        # The pass-through proxy uses the same public protocol as all variants.
        # Running the same query sequence catches proxy and query-order effects.
        proxy = _batch(FrameView(recording.recording_id, frames), rows, group,
                       build_index, rank_evidence, source, rows, frames)
        eligible = []
        for question in group:
            qid = question["question_id"]
            case = cases[qid]
            case["checks"]["wrapper_control"] = _comparison(case["baseline"], proxy[qid]["first"], tolerance, control=True)
            controls = [v["status"] for k, v in case["checks"].items() if k.endswith("control")]
            if "intervals" in case["baseline"] and all(s in {"passed", "stable_empty"} for s in controls):
                eligible.append(question)
            else:
                case["checks"]["future_tests"] = {"status": "inconclusive", "reason": "Matched controls did not agree"}
        if not eligible:
            continue
        changed = changed_future_observations(rows, frames, cutoff, recording.recording_id, source)
        replacements = replacement_frames(frames, cutoff, Path(workspace) / str(cutoff))
        variants = {
            "future_observations": (FrameView(recording.recording_id, frames), changed),
            "prefix_only": (FrameView(recording.recording_id, frames, cutoff=cutoff),
                            [row for row in rows if row["frame_id"] <= cutoff]),
            "future_pixels": (FrameView(recording.recording_id, frames, replacements=replacements), rows),
        }
        for name, (view, supplied_rows) in variants.items():
            values = _batch(view, supplied_rows, group, build_index, rank_evidence, source, rows, frames)
            for question in eligible:
                qid = question["question_id"]
                cases[qid]["checks"][name] = _comparison(proxy[qid]["first"], values[qid]["first"], tolerance)
    counts = Counter(value["status"] for case in cases.values() for value in case["checks"].values())
    temporal_positive = sum(case["checks"].get(name, {}).get("status") == "passed"
                            for case in cases.values() for name in ("future_observations", "prefix_only", "future_pixels"))
    empty = sum(case["checks"].get(name, {}).get("status") == "stable_empty"
                for case in cases.values() for name in ("future_observations", "prefix_only", "future_pixels"))
    status = "failed" if counts["failed"] else "inconclusive" if counts["inconclusive"] or not temporal_positive else "passed_with_gaps" if empty or counts["not_applicable"] else "passed"
    content = [{"frame_id": f["frame_id"], "timestamp_s": f["timestamp_s"], "width": f["width"], "height": f["height"],
                "image_sha256": fingerprint(f["image_path"])} for f in frames]
    return {
        "recording_id": recording.recording_id, "recording_adapter": f"{type(recording).__module__}.{type(recording).__name__}",
        "source": source, "status": status, "questions": list(cases.values()), "check_counts": dict(counts),
        "question_count": len(questions), "frame_count": len(frames), "observation_count": len(rows),
        "nonempty_rankings": sum(bool(c["baseline"].get("intervals")) for c in cases.values()),
        "ranking_kind_counts": dict(Counter(c["ranking_kind"] for c in cases.values())),
        "context_temporal_comparisons": sum(c["ranking_kind"] == "zero_score_scene_context" and c["checks"].get(name, {}).get("status") == "passed"
                                             for c in cases.values() for name in ("future_observations", "prefix_only", "future_pixels")),
        "positive_temporal_comparisons": temporal_positive, "empty_temporal_comparisons": empty,
        "input_content_sha256": sha256(json.dumps({"frames": content, "observations": rows}, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }, rows


def markdown_report(report):
    lines = ["# Retrieval integration checks", "", f"Result: **{report['status']}**. Track source: **{report['source']}**.", "", report["scope"], "",
             f"Question loader: `{report['question_loader']}`. Retrieval module: `{report['retrieval_module']}`.", "",
             "| Recording | Status | Questions | Nonempty rankings | Zero-score scene contexts | Nonempty temporal checks (context subset) | Stable-empty checks |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["recordings"]:
        lines.append(f"| {_md(item['recording_id'])} | {item['status']} | {item.get('question_count',0)} | {item.get('nonempty_rankings',0)} | {item.get('ranking_kind_counts',{}).get('zero_score_scene_context',0)} | {item.get('positive_temporal_comparisons',0)} ({item.get('context_temporal_comparisons',0)}) | {item.get('empty_temporal_comparisons',0)} |")
    lines += ["", f"Detector adapter called: **{report['detector_adapter_called']}**. Observation origin: {_md(report['observation_origin'])}.", ""]
    for item in report["recordings"]:
        lines += [f"## {_md(item['recording_id'])}", ""]
        if item.get("error"):
            lines.append(_md(item["error"]))
        if item.get("coverage_note"):
            lines.append(item["coverage_note"])
        if item.get("run_directory"):
            lines.append(f"Evidence export: `{item['run_directory']}` (answers are not requested).")
        for case in item.get("questions", []):
            findings = [f"{key}: {value['status']}" for key, value in case["checks"].items() if value["status"] != "passed"]
            if case.get("ranking_kind") == "zero_score_scene_context":
                findings.insert(0, "zero-score scene context; semantic relevance and answerability unassessed")
            if findings:
                lines.append(f"- {_md(case['question']['question_id'])}: " + "; ".join(findings))
        lines.append("")
    lines += ["## Limits", "", *[f"- {note}" for note in report["limitations"]], ""]
    return "\n".join(lines)


def verify_suite(data_root, question_paths, source, output_dir, *, retrieval_module="trackrag.retrieval", predicted_paths=(), tolerance=1e-6):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Output already exists; choose a new directory")
    if source not in {"annotated", "predicted"}:
        raise ValueError("Invalid track source")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Score tolerance must be finite and nonnegative")
    questions = validate_questions([q for path in question_paths for q in load_questions(path)])
    if not questions:
        raise ValueError("The question files are empty")
    if predicted_paths and source != "predicted":
        raise ValueError("Saved prediction files require source predicted")
    grouped = defaultdict(list)
    for question in questions:
        grouped[question["recording_id"]].append(question)
    saved = defaultdict(list)
    for path in predicted_paths:
        for row in read_jsonl(path):
            if row.get("source") != "predicted" or row.get("recording_id") not in grouped:
                raise ValueError("Prediction file has the wrong source or a recording outside the selected cohort")
            saved[row["recording_id"]].append(row)
    build = _adapter(retrieval_module, "build_index", 4)
    rank = _adapter(retrieval_module, "rank_evidence", 4)
    detector = _adapter("trackrag.vision", "predict_tracks", 1) if source == "predicted" and not predicted_paths else None
    origin = "saved predictions; generator provenance not verified" if predicted_paths else "prediction adapter" if detector else "Task 2 annotated observations"
    report = {
        "version": 1, "scope": "Contract and temporal falsification checks on supplied observations. No answer model or gold key is used. This is not retrieval accuracy or proof that future data was never accessed.",
        "source": source, "question_loader": "trackrag.questions.load_questions", "retrieval_module": retrieval_module,
        "score_tolerance": tolerance, "observation_origin": origin, "detector_adapter_called": False,
        "question_inputs": [{"path": str(Path(p).resolve()), "sha256": fingerprint(p)} for p in question_paths],
        "prediction_inputs": [{"path": str(Path(p).resolve()), "sha256": fingerprint(p)} for p in predicted_paths],
        "recordings": [], "two_recordings_exercised": False,
        "limitations": [
            "These checks do not measure detection, tracking, retrieval or answer accuracy. Gold scoring stays separate.",
            "Zero-score scene contexts are counted separately. A nonempty fallback or passing temporal control does not establish a relevant match or answer.",
            "Only the shared Recording protocol is proxied. Wrapper-dependent or unstable controls are inconclusive, not proof of leakage.",
            "Passing perturbations cannot prove absence of leakage. Hidden/global caches can mask changes; use isolated or correctly invalidated model caches.",
            "Saved or frozen observations may already contain future-dependent detector/tracker output. Task 1 causality is not tested.",
            "Stable-empty outputs and cutoffs with no future frames remain explicit and do not count as positive temporal comparisons.",
            "The runner does not call an answerer. A selected retrieval or detector adapter can load models or contact services; the explicit TF-IDF module runs locally.",
            "Synthetic future images/observations are test controls, not detector predictions or measured driving performance.",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".retrieval-check-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        for position, (recording_id, cohort) in enumerate(grouped.items(), 1):
            item = None
            try:
                recording = open_recording(data_root, recording_id)
                if recording.recording_id != recording_id:
                    raise ValueError("Reader returned a different recording")
                _validate_questions(recording, cohort)
                if predicted_paths:
                    observations = saved[recording_id]
                elif detector:
                    report["detector_adapter_called"] = True
                    observations = list(detector(recording))
                else:
                    observations = load_annotated_tracks(recording)
                item, rows = check_recording(recording, cohort, observations, source, build, rank,
                                             staging / "counterfactual-inputs" / str(position), tolerance)
                if predicted_paths:
                    item["saved_prediction_coverage"] = "rows_present" if rows else "unverified_empty"
                    if not rows and item["status"] != "failed":
                        item["status"] = "inconclusive"
                        item["coverage_note"] = "No saved rows: a zero-detection recording and a missing export cannot be distinguished"
                if all("intervals" in case["baseline"] for case in item["questions"]):
                    relative = f"runs/recording-{position:02d}"
                    captured = {case["question"]["question_id"]: case["baseline"]["intervals"] for case in item["questions"]}
                    review_recording(recording, cohort, source=source,
                                     load_tracks=lambda rec: deepcopy(rows), build_index=lambda rec, obs: None,
                                     rank_evidence=lambda q, idx, k=3: deepcopy(captured[q["question_id"]]),
                                     output_dir=staging / relative,
                                     run_metadata={"adapter_module": retrieval_module, "answerer": "none",
                                                   "mode": "replay of original captured retrieval for evidence export",
                                                   "observation_origin": origin, "detector_adapter_called": bool(detector)})
                    item["run_directory"] = relative
                    audit_run(staging / relative, {q["question_id"]: q for q in cohort})
                report["recordings"].append(item)
            except Exception as error:
                item = item or {"recording_id": recording_id}
                item.update(status="failed", error=_error(error))
                report["recordings"].append(item)
        states = [r["status"] for r in report["recordings"]]
        report["status"] = "failed" if "failed" in states else "inconclusive" if "inconclusive" in states else "passed_with_gaps" if "passed_with_gaps" in states else "passed"
        report["two_recordings_exercised"] = len(report["recordings"]) >= 2 and all("frame_count" in r for r in report["recordings"])
        (staging / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "report.md").write_text(markdown_report(report), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError("Output was created during verification")
        staging.rename(output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--questions", required=True, nargs="+", type=Path)
    parser.add_argument("--source", required=True, choices=("annotated", "predicted"))
    parser.add_argument("--retrieval-module", default="trackrag.retrieval")
    parser.add_argument("--predicted-tracks", nargs="+", type=Path)
    parser.add_argument("--score-tolerance", type=float, default=1e-6)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_suite(args.data_root, args.questions, args.source, args.out,
                              retrieval_module=args.retrieval_module, predicted_paths=args.predicted_tracks or (),
                              tolerance=args.score_tolerance)
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved retrieval checks to {args.out / 'report.md'}; result: {report['status']}")
    return 1 if report["status"] == "failed" else 2 if report["status"] == "inconclusive" else 0


if __name__ == "__main__":
    raise SystemExit(main())
