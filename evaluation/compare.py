"""Compare saved multi-recording experiments. This module makes no model calls."""

import argparse
import html
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from .run_audit import audit_run, fingerprint, local_asset, read_object, validate_question
from .score import read_jsonl, score_results


def _path(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest paths must be nonempty strings")
    return (base / value).resolve()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _fields(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Unknown fields or invalid object in {name}")


def compare_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_object(manifest_path)
    _fields(manifest, {"version", "name", "purpose", "questions", "manual_question_ids", "gold",
                       "temporal_iou", "box_iou", "variants", "reference_variant", "notes"}, "comparison manifest")
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("Comparison manifest version must be 1")
    name = _text(manifest["name"], "Comparison name")
    purpose = manifest["purpose"]
    if purpose not in {"development", "evaluation"}:
        raise ValueError("Purpose must be development or evaluation; this declaration does not certify a held-out split")
    notes = manifest.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("Comparison notes must be a list of strings")
    base, expected, files = manifest_path.parent, {}, []

    def input_file(path, role):
        files.append({"path": str(path), "role": role, "sha256": fingerprint(path)})

    input_file(manifest_path, "manifest")
    question_paths = manifest["questions"]
    if not isinstance(question_paths, list) or not question_paths:
        raise ValueError("At least one shared question file is required")
    for value in question_paths:
        path = _path(base, value)
        input_file(path, "questions")
        for question in read_jsonl(path):
            validate_question(question)
            qid = question["question_id"]
            if qid in expected:
                raise ValueError("Question IDs must be unique across all recordings and files")
            expected[qid] = question
    if not expected:
        raise ValueError("The shared question set is empty")
    diagnostics = manifest.get("manual_question_ids", [])
    if (not isinstance(diagnostics, list) or any(not isinstance(q, str) or q not in expected for q in diagnostics)
            or len(set(diagnostics)) != len(diagnostics)):
        raise ValueError("Manual diagnostic IDs must be unique members of the shared question set")
    diagnostics = set(diagnostics)
    ordinary = set(expected) - diagnostics
    if not ordinary:
        raise ValueError("At least one ordinary evaluation question is required")
    gold = []
    gold_paths = manifest.get("gold", [])
    if not isinstance(gold_paths, list):
        raise ValueError("Gold must be a list of answer-key paths")
    for value in gold_paths:
        path = _path(base, value)
        input_file(path, "gold")
        gold.extend(read_jsonl(path))
    if gold_paths:
        ids = [row["question_id"] for row in gold]
        if len(set(ids)) != len(ids) or set(ids) != ordinary:
            raise ValueError("Retrieval gold must cover every ordinary question exactly once, excluding manual diagnostics")
        for row in gold:
            question = expected[row["question_id"]]
            if (row["recording_id"], row["query_frame"]) != (question["recording_id"], question["query_frame"]):
                raise ValueError("Answer key recording/cutoff differs from shared questions")
    temporal_iou, box_iou = manifest.get("temporal_iou", 0.5), manifest.get("box_iou", 0.5)
    if (isinstance(temporal_iou, bool) or isinstance(box_iou, bool)
            or not 0 < temporal_iou <= 1 or not 0 < box_iou <= 1):
        raise ValueError("Scoring thresholds must be greater than zero and at most one")
    specifications = manifest["variants"]
    if not isinstance(specifications, list) or len(specifications) < 2:
        raise ValueError("A comparison requires at least two variants")
    variants, seen_variants = [], set()
    for specification in specifications:
        _fields(specification, {"id", "label", "runs"}, "variant")
        identity = _text(specification["id"], "Variant ID")
        if identity in seen_variants or not re.fullmatch(r"[a-zA-Z0-9_-]+", identity):
            raise ValueError("Variant IDs must be unique and contain only letters, digits, underscores or hyphens")
        seen_variants.add(identity)
        entries = specification["runs"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("Each variant requires at least one saved run")
        results, reviews, provenance, predicted, annotated = {}, {}, [], [], []
        seen_review_ids, review_bindings = set(), []
        recordings, sources = set(), set()
        for entry in entries:
            _fields(entry, {"directory", "annotated_tracks", "review"}, "run entry")
            run = audit_run(_path(base, entry["directory"]), expected)
            if run["recording_id"] in recordings:
                raise ValueError("Use one run per recording in each variant")
            recordings.add(run["recording_id"])
            sources.add(run["source"])
            results.update(run["results"])
            files.extend(run["files"])
            provenance.append(run["provenance"])
            if run["source"] == "predicted":
                predicted.extend(run["tracks"])
                needs_association = any(
                    item["interval"]["track_id"] is not None
                    for qid, result in run["results"].items() if qid in ordinary
                    for item in result["evidence"]
                )
                if gold_paths and needs_association and not entry.get("annotated_tracks"):
                    raise ValueError("Each predicted run with tracked evidence requires its own annotated reference file for scoring")
            if entry.get("annotated_tracks"):
                path = _path(base, entry["annotated_tracks"])
                rows = read_jsonl(path)
                if any(r["recording_id"] != run["recording_id"] or r["source"] != "annotated" for r in rows):
                    raise ValueError("Reference tracks must be annotated observations from the same recording")
                annotated.extend(rows)
                input_file(path, "annotated_tracks")
            review = entry.get("review")
            if review is None:
                continue
            _fields(review, {"path", "results_sha256", "artifacts"}, "review binding")
            if review["results_sha256"] != fingerprint(run["directory"] / "results.jsonl"):
                raise ValueError("Review is bound to a different results file; review the current run")
            binding_scope = "results_only"
            if review.get("artifacts"):
                snapshot_path = _path(base, review["artifacts"])
                snapshot = read_object(snapshot_path)
                if type(snapshot.get("version")) is not int or snapshot["version"] != 1 or not isinstance(snapshot.get("files"), list):
                    raise ValueError("Invalid review artifact snapshot")
                actual = {item["path"]: item["sha256"] for item in run["files"]}
                bound = {}
                for item in snapshot["files"]:
                    asset = local_asset(run["directory"], item["path"])
                    if str(asset) in bound:
                        raise ValueError("Duplicate file in review artifact snapshot")
                    bound[str(asset)] = item["sha256"]
                if bound != actual:
                    raise ValueError("Review artifact snapshot differs from current run files; review the changed artifacts")
                input_file(snapshot_path, "review_artifact_snapshot")
                binding_scope = "results_and_audited_artifacts"
            review_bindings.append({"recording_id": run["recording_id"], "scope": binding_scope})
            path = _path(base, review["path"])
            input_file(path, "reviews")
            for row in read_jsonl(path):
                qid, grade = row["question_id"], row["grade"]
                if qid not in expected or expected[qid]["recording_id"] != run["recording_id"] or qid in seen_review_ids:
                    raise ValueError("Review question is unknown, duplicated or belongs to a different recording")
                seen_review_ids.add(qid)
                if grade is None:
                    if not isinstance(row.get("note"), str) or not isinstance(row.get("reviewer"), str):
                        raise ValueError("Pending review fields must be strings")
                    continue
                if grade not in {"pass", "qualified", "fail", "manual"}:
                    raise ValueError("Unknown review grade")
                if grade == "manual" and qid not in diagnostics:
                    raise ValueError("Manual exclusions must be declared before comparing variants")
                _text(row["note"], "Review note")
                _text(row["reviewer"], "Reviewer label")
                delivered = results.get(qid, {}).get("answer", {}).get("answer_status", "missing")
                if grade != "fail" and delivered not in {"answered", "insufficient_evidence"}:
                    raise ValueError("Missing, failed or unrequested answers cannot receive a successful review grade")
                reviews[qid] = {key: row[key] for key in ("grade", "note", "reviewer")}
        if len(sources) != 1:
            raise ValueError("A variant must use one track source; compare annotated and predicted as separate variants")
        counts = Counter(results.get(qid, {}).get("answer", {}).get("answer_status", "missing") for qid in expected)
        grades = Counter(reviews.get(qid, {}).get("grade", "unreviewed") for qid in ordinary)
        coverage = sum(grades[g] for g in ("pass", "qualified", "fail"))
        retrieval = None
        if gold_paths:
            retrieval = score_results(
                [row for qid, row in results.items() if qid in ordinary], gold,
                temporal_iou=temporal_iou, box_iou=box_iou,
                predicted_tracks=predicted, annotated_tracks=annotated,
            )
        cases = []
        for qid, question in expected.items():
            result = results.get(qid)
            cases.append({
                "question": question, "diagnostic": qid in diagnostics,
                "answer": result["answer"] if result else {"answer_status": "missing", "answer": None, "cited_evidence_ids": []},
                "review": reviews.get(qid),
            })
        variants.append({
            "id": identity, "label": _text(specification.get("label", identity), "Variant label"),
            "source": next(iter(sources)), "recordings_with_runs": sorted(recordings),
            "status_counts": {s: counts[s] for s in ("answered", "insufficient_evidence", "error", "not_requested", "missing")},
            "review_counts": {g: grades[g] for g in ("pass", "qualified", "fail", "unreviewed")},
            "ordinary_reviewed": coverage, "ordinary_total": len(ordinary),
            "clean_pass_fraction": grades["pass"] / len(ordinary) if coverage == len(ordinary) else None,
            "diagnostics": [{"question_id": qid, "review": reviews.get(qid)} for qid in expected if qid in diagnostics],
            "reviewer_labels": sorted({row["reviewer"] for row in reviews.values()}),
            "review_bindings": review_bindings,
            "retrieval": retrieval, "runs": provenance, "questions": cases,
        })
    reference_id = manifest.get("reference_variant", variants[0]["id"])
    if reference_id not in seen_variants:
        raise ValueError("Reference variant is unknown")
    reference = next(v for v in variants if v["id"] == reference_id)
    paired = []
    reference_reviews = {q["question"]["question_id"]: q["review"] for q in reference["questions"]}
    for variant in variants:
        if variant is reference:
            continue
        gains, losses, paired_count = [], [], 0
        for case in variant["questions"]:
            qid, review = case["question"]["question_id"], case["review"]
            other = reference_reviews[qid]
            if qid in diagnostics or review is None or other is None:
                continue
            paired_count += 1
            if review["grade"] == "pass" and other["grade"] != "pass":
                gains.append(qid)
            if review["grade"] != "pass" and other["grade"] == "pass":
                losses.append(qid)
        paired.append({"reference": reference_id, "variant": variant["id"], "reviewed_pairs": paired_count,
                       "clean_pass_gains": gains, "clean_pass_losses": losses})
    # Include hashes for precisely the files audited, not the current source tree
    # or a manifest's unverified claim about which checkpoint produced the run.
    unique_files = {row["path"]: row for row in files}
    return {
        "version": 1, "name": name, "purpose": purpose, "notes": notes,
        "scope": "Offline comparison of saved outputs; no models run. Reviews are supplied judgments, not automatic factual verification. Paired differences do not establish a causal training effect or a held-out split.",
        "review_binding_scope": "Every review binding authenticates results.jsonl bytes. Optional artifact snapshots also bind all audited run files to the exported review packet. Neither form proves that a reviewer actually inspected the material; legacy result-only bindings do not authenticate prior image bytes.",
        "expected_questions": len(expected), "ordinary_questions": len(ordinary),
        "manual_question_ids": [qid for qid in expected if qid in diagnostics],
        "variants": variants, "paired_comparisons": paired, "input_files": list(unique_files.values()),
    }


def _md(value):
    value = html.escape(str(value))
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("[", "\\[").replace("]", "\\]").replace("`", "\\`").replace("\r", " ").replace("\n", " ")


def markdown_report(report):
    lines = [f"# {_md(report['name'])}", "", report["scope"], "",
             f"Declared purpose: **{report['purpose']}**. Questions: **{report['expected_questions']}**, including **{len(report['manual_question_ids'])}** manual diagnostics declared in this manifest.", "",
             report["review_binding_scope"], ""]
    lines.extend(f"- {_md(note)}" for note in report["notes"])
    lines += ["", "| Variant | Source | Clean pass | Qualified | Fail | Unreviewed | Provider/model errors | Missing results |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in report["variants"]:
        r, s = variant["review_counts"], variant["status_counts"]
        lines.append(f"| {_md(variant['label'])} | {variant['source']} | {r['pass']} | {r['qualified']} | {r['fail']} | {r['unreviewed']} | {s['error']} | {s['missing']} |")
    lines += ["", "Clean-pass counts use all ordinary questions. Incomplete review coverage produces no clean-pass fraction. Error/missing counts describe all questions and can overlap review failures; the columns are not additive. A correct answer status does not establish factual correctness.", ""]
    for variant in report["variants"]:
        lines += [f"## {_md(variant['label'])}", "", f"Reviewed ordinary questions: {variant['ordinary_reviewed']}/{variant['ordinary_total']}.",
                  "Reviewer labels: " + "; ".join(_md(x) for x in variant["reviewer_labels"]), ""]
        status = variant["status_counts"]
        lines.append("Delivered statuses: " + "; ".join(f"{key}={status[key]}" for key in status) + ".")
        for run in variant["runs"]:
            directory = Path(run["directory"])
            target = directory / "report.html"
            if not target.is_file() or not target.resolve().is_relative_to(directory):
                target = directory / "results.jsonl"
            lines.append(f"- [Open {_md(run['recording_id'])} saved output](<{target.as_posix()}>)")
        models = sorted({(r.get("model") or "not recorded", r.get("reported_model") or "not recorded")
                         for run in variant["runs"] for r in run["provider_receipts"]})
        lines.append("Provider IDs (requested → reported): " + ("; ".join(f"{_md(a)} → {_md(b)}" for a, b in models) or "No provider receipts available."))
        if variant["retrieval"] is None:
            lines.append("Retrieval metrics: not scored because no separate retrieval answer key was supplied.")
        else:
            score = variant["retrieval"]
            lines.append(f"Retrieval top-1: {score['counts']['top1_correct']}/{score['counts']['answerable']}; top-3: {score['counts']['top3_correct']}/{score['counts']['answerable']}.")
        lines += ["", "| Question | Prompt | Delivered status | Review | Finding |", "| --- | --- | --- | --- | --- |"]
        for case in variant["questions"]:
            qid = case["question"]["question_id"]
            review = case["review"]
            grade = review["grade"] if review else "unreviewed"
            if case["diagnostic"]:
                grade += " (manual diagnostic)"
            lines.append(f"| {_md(qid)} | {_md(case['question']['question'])} | {case['answer']['answer_status']} | {_md(grade)} | {_md(review['note'] if review else 'No content review supplied.')} |")
        lines.append("")
    lines += ["## Paired review changes", ""]
    for pair in report["paired_comparisons"]:
        lines.append(f"- {_md(pair['reference'])} → {_md(pair['variant'])}: {pair['reviewed_pairs']} reviewed pairs; {len(pair['clean_pass_gains'])} clean-pass gains; {len(pair['clean_pass_losses'])} clean-pass losses.")
    return "\n".join(lines) + "\n"


def write_bundle(output_dir, contents):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Output already exists; choose a new directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        for name, content in contents.items():
            (staging / name).write_text(content, encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError("Output was created during evaluation; choose a new directory")
        staging.rename(output_dir)


def write_comparison(manifest_path, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Comparison output already exists; choose a new directory")
    report = compare_manifest(manifest_path)
    write_bundle(output_dir, {
        "summary.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "report.md": markdown_report(report),
    })
    return output_dir / "report.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="A new output directory")
    args = parser.parse_args(argv)
    try:
        path = write_comparison(args.manifest, args.out)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved offline comparison to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
